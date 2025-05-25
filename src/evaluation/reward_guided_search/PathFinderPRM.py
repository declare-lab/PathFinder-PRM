import torch
import torch.nn.functional as F

from copy import *
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_config import logger

class PathFinderPRM:
    def __init__(self, model_path, model_init_kwargs = None):
        # Load the model and tokenizer
        logger.info(f"Loading model from {model_path}")

        if model_init_kwargs == None:
            model_init_kwargs = {
                "torch_dtype":torch.bfloat16,
                "attn_implementation":"flash_attention_2",
                "device_map":"auto"
            }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_init_kwargs
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )
        logger.info(f"PathFinderPRM loaded successfully")
        self.pos_token_id = self.tokenizer.encode("<+>")[0]
        self.neg_token_id = self.tokenizer.encode("<->")[0]


    def inference_single(self, sample_input, logging=False):

        message_ids = self.tokenizer.apply_chat_template(
            sample_input,
            tokenize=True,
            return_dict=True,
            return_tensors='pt'
            ).to(self.model.device)

        mask_token_id = self.tokenizer.encode("<extra>")[0]
        token_masks = (message_ids['input_ids'] == mask_token_id)

        shifted_mask = torch.cat(
            [
                token_masks[:, 1:], 
                torch.zeros(token_masks.size(0), 1, dtype=torch.bool, device=self.model.device)
            ], 
            dim=1
            )

        with torch.no_grad():
            outputs = self.model(**message_ids)

        allowed_token_ids = torch.tensor([self.pos_token_id, self.neg_token_id], device=outputs.logits.device)  # shape: (2,)

        masked_logits = outputs.logits[shifted_mask][:, allowed_token_ids]
        predicted_indices = masked_logits.argmax(dim=-1)
        predicted_tokens = allowed_token_ids[predicted_indices]

        decoded_tokens = [self.tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in predicted_tokens]

        if logging:
            logger.info("Decoded Labels: ", decoded_tokens)

        if '<->' in decoded_tokens:
            return -1

        new_messages = sample_input.copy()

        asst_response = new_messages[-1]['content']

        for pred in decoded_tokens:
            asst_response = asst_response.replace("<extra>", pred, 1)

        asst_response += ', Correctness: <extra>'

        new_messages[-1]['content'] = asst_response
        
        new_message_ids = self.tokenizer.apply_chat_template(
            new_messages,
            tokenize=True,
            return_dict=True,
            return_tensors='pt'
            ).to(self.model.device)

        token_masks = (new_message_ids['input_ids'] == mask_token_id)

        shifted_mask = torch.cat(
            [
                token_masks[:, 1:], 
                torch.zeros(token_masks.size(0), 1, dtype=torch.bool, device=self.model.device)
            ], 
            dim=1
            )

        with torch.no_grad():
            outputs = self.model(**new_message_ids)

        masked_logits = outputs.logits[shifted_mask]

        restricted_logits = masked_logits[:, [self.pos_token_id,self.neg_token_id]]   

        probs_pos_neg  = F.softmax(restricted_logits, dim=-1)

        return probs_pos_neg[0][0].cpu().item()