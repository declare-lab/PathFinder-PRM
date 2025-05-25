from __future__ import annotations

from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from .abstract_model import prm
from ..utils.model_utils import remove_step_prefix
from ..utils.log_utils import get_logger
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from accelerate.data_loader import DataLoaderShard

logger = get_logger(__name__)

import socket, logging
RANK   = Accelerator().local_process_index
HOST   = socket.gethostname().split('.')[0]

def dbg(msg: str, lvl=logging.INFO):
    """
    Emit rank-tagged debug lines only if PRM_DEBUG=1
    """
    if os.getenv("PRM_DEBUG", "0") != "1":
        return
    _msg = f"[{HOST} rank{RANK}] {msg}"
    logger.log(lvl, _msg)

import json, os, io
LOG_PATH = "./prm_bench.txt"

if os.path.exists(LOG_PATH) and os.environ.get("PRM_LOG_RESET", "1") == "1":
    open(LOG_PATH, "w", encoding="utf-8").close()  
def _log(payload):
    with io.open(LOG_PATH, "a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")

class MyDiscriminativePRM(prm):
    def __init__(
        self,
        pretrained: str,
        validity_threshold: float = 0.5,
    ) -> None:
        super().__init__(validity_threshold=validity_threshold)

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True, attn_implementation='flash_attention_2'   # NOTE: please remove this argument if flash_attention_2 is not supported on your system
        ) 

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer pad_token was None, set to eos_token: %s (ID: %d)", self.tokenizer.eos_token, self.tokenizer.eos_token_id)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained, torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True, attn_implementation='flash_attention_2'
        )

        self.accelerator = Accelerator()
        self.model = self.accelerator.prepare(self.model)
        self.model.eval() 

        self.step_separator: str = "<extra>"
        try:
            sep_tokens = self.tokenizer.encode(self.step_separator, add_special_tokens=False)
            if not sep_tokens:
                raise ValueError(f"Tokenizer encoded step_separator '{self.step_separator}' to an empty list.")
            self.step_sep_id: int = sep_tokens[0]
        except Exception as e:
            logger.error(f"Failed to properly encode step_separator '{self.step_separator}'. It might not be a single token or present in the vocabulary. Error: {e}")
            # Fallback to converting token to ID, which might yield UNK if not setup correctly
            self.step_sep_id: int = self.tokenizer.convert_tokens_to_ids(self.step_separator)

        self.good_id: int = self.tokenizer.convert_tokens_to_ids("<+>")
        self.bad_id: int = self.tokenizer.convert_tokens_to_ids("<->")

        # Critical checks for special tokens
        if self.good_id == self.tokenizer.unk_token_id or self.bad_id == self.tokenizer.unk_token_id:
            msg = "<+> or <-> tokens are UNK. Model predictions will be meaningless. Check tokenizer and model vocabulary."
            logger.error(msg)
            raise ValueError(msg)
        if self.step_sep_id == self.tokenizer.unk_token_id:
            msg = f"Step separator token '{self.step_separator}' is UNK. This is critical for finding evaluation points."
            logger.error(msg)
            raise ValueError(msg)

        logger.info(
            "DiscPRM initialised | good_id=%d bad_id=%d sep_id=%d pad_id=%d",
            self.good_id,
            self.bad_id,
            self.step_sep_id,
            self.tokenizer.pad_token_id
        )

        self.system_prompt = (
            "You are a Math Teacher. Given a question and a student's "
            "solution, evaluate the mathematical correctness and logic "
            "consistency of the current step."
        )

    def getitem_function(self, meta_data: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
        """
        Processes one problem instance from PRMBench.
        To maintain compatibility with DataCollatorForSupervisedDataset, this function
        must return a dictionary with an 'input_ids' key containing a 1D tensor.
        We will pass the raw question and steps text to be processed in the `respond` method.
        The 'input_ids' returned here will be a placeholder (e.g., tokenized question)
        and will be ignored by the `respond` method's main logic for step evaluation.
        """
        sample = meta_data[index]
        q_text: str = sample["question"]
        steps_list: List[str] = sample.get("steps", [])
        sample_idx: str = sample["idx"]

        # Create a placeholder input_ids (e.g., tokenized question) to satisfy the collator
        # This specific input_ids tensor will not be directly used for step evaluation in respond,
        # as respond will re-tokenize per step.
        placeholder_input_ids = self.tokenizer.encode(
            q_text,
            return_tensors=None, # Get list of IDs
            add_special_tokens=True,
            truncation=True, # Avoid overly long placeholders if questions are huge
            max_length=512 # Arbitrary reasonable max_length for placeholder
        )
        if placeholder_input_ids is None: placeholder_input_ids = []

        return {
            "idx": sample_idx,
            "input_ids": torch.tensor(placeholder_input_ids, dtype=torch.long), # Must be a 1D tensor
            "raw_question_text": q_text,    # Pass raw data for respond method
            "raw_list_of_steps": steps_list # Pass raw data for respond method
        }

    def respond(self, dataloader) -> List[Dict[str, Any]]:
        self.model.eval()
        device = self.accelerator.device
        good_id = self.good_id
        bad_id = self.bad_id
        sep_id = self.step_sep_id
        threshold = self.validity_threshold
        pad_id = self.tokenizer.pad_token_id

        if not isinstance(dataloader, DataLoaderShard):
            prepared_dataloader = self.accelerator.prepare(dataloader)
        else:
            prepared_dataloader = dataloader
            
        try:
            total_items = len(prepared_dataloader)   
        except AttributeError:
            total_items = len(prepared_dataloader)
            logger.warning("Could not determine total items from dataloader.dataset, using len(dataloader).")

        def _process_batch(step_items: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not step_items:
                return {}

            # Pad batch
            padded = torch.nn.utils.rnn.pad_sequence(
                [it["ids"] for it in step_items], batch_first=True, padding_value=pad_id
            )
            attn = (padded != pad_id).long()
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(input_ids=padded.to(device), attention_mask=attn.to(device)).logits
            
            # Compute probabilities
            lse = torch.logsumexp(logits, dim=-1)
            p_plus = torch.exp(logits[..., good_id] - lse)
            p_minus = torch.exp(logits[..., bad_id] - lse)
            
            # Move to CPU for processing
            p_plus_cpu = p_plus.cpu()
            p_minus_cpu = p_minus.cpu()
            
            # Score each item
            mask_extra = (padded == sep_id)
            batch_results: Dict[str, Any] = {}

            for i, item in enumerate(step_items):
                pid = item["pid"]
                current_mask_extra = mask_extra[i]
                pos = current_mask_extra.nonzero(as_tuple=False).squeeze(-1) - 1
                actual_seq_len = p_plus_cpu.size(1)
                pos = pos[pos < actual_seq_len]

                p_plus_item = p_plus_cpu[i, pos]
                p_minus_item = p_minus_cpu[i, pos]

                if "_log" not in item: 
                    item["_log"] = {}
                item["_log"].update(
                    p_plus=[float(x) for x in p_plus_item],
                    p_minus=[float(x) for x in p_minus_item],
                )
                _log(item["_log"])

                score = float(p_plus_item.mean()) if p_plus_item.numel() > 0 else 0.0
                label = bool((p_plus_item >= threshold).all()) if p_plus_item.numel() > 0 else (threshold == 0.0)
                
                rec = batch_results.setdefault(pid, {"step_level_validity_scores": [], "step_level_validity_labels": []})
                rec["step_level_validity_scores"].append(score)
                rec["step_level_validity_labels"].append(label)
                
            return batch_results
        
        rank = self.accelerator.process_index
        bar = tqdm(
            total=total_items,
            desc=f"Rank {rank}",
            position=rank,
            leave=False,
            dynamic_ncols=True
        )
        all_problem_results = [] 

        for batch_idx, batch in enumerate(prepared_dataloader):
            print(f"[Rank {rank}] entering batch {batch_idx}, {batch['idx']}")
            self.accelerator.wait_for_everyone()

            if batch is None: 
                continue
                
            prob_ids   = batch["idx"] 
            questions  = batch["raw_question_text"] 
            step_lists = batch["raw_list_of_steps"]
            current_batch_size = len(prob_ids)
            
            batch_step_items_to_process: List[Dict[str, Any]] = []

            for i in range(current_batch_size):
                pid = prob_ids[i]
                steps = step_lists[i]
                q_text_single = questions[i]
                
                if not steps:
                    if hasattr(prepared_dataloader.dataset, "store_results"):
                        prepared_dataloader.dataset.store_results({
                            "idx": pid, "scores": {"step_level_validity_scores": [], "step_level_validity_labels": []}
                        })
                    else:
                        all_problem_results.append({
                            "idx": pid, "scores": {"step_level_validity_scores": [], "step_level_validity_labels": []}
                        })
                    continue

                for s_idx, step_text in enumerate(steps):
                    prev_steps_text = "\n\n".join(remove_step_prefix(x) for x in steps[:s_idx])
                    current_step_text = remove_step_prefix(step_text)
                    prefix = f"{prev_steps_text}\n\n" if prev_steps_text else ""
                    assistant_content = (
                        prefix +
                        f"Current step: {current_step_text} "
                        f"Math reasoning: {self.step_separator}, "
                        f"Consistency: {self.step_separator}"
                    )
                    messages = [
                        {"role": "user", "content": f"{self.system_prompt}\n\n Question: {q_text_single}"},
                        {"role": "assistant", "content": assistant_content}
                    ]
                    
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    prompt_ids_list = self.tokenizer.encode(
                        prompt_text, return_tensors=None, add_special_tokens=True
                    ) or []
                    
                    batch_step_items_to_process.append({
                        "pid": pid, "sidx": s_idx,
                        "ids": torch.tensor(prompt_ids_list, dtype=torch.long),
                        "_log": {"problem_id": pid, "step_idx": s_idx, "prompt_len": len(prompt_ids_list)},
                    })
            
            if batch_step_items_to_process:
                aggregated_results = _process_batch(batch_step_items_to_process)
                for pid_res, scores_data in aggregated_results.items():
                    if hasattr(prepared_dataloader.dataset, "store_results"):
                        prepared_dataloader.dataset.store_results({"idx": pid_res, "scores": scores_data})
                    else:
                        all_problem_results.append({"idx": pid_res, "scores": scores_data})
            
            if bar is not None: 
                bar.update(current_batch_size)
            print(f"[Rank {rank}] finished batch {batch_idx}, {batch['idx']}")

        rank = self.accelerator.process_index
        self.accelerator.wait_for_everyone()
