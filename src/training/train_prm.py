import random
import sys
import colorlog
import datasets
import numpy as np
import torch
import transformers

from accelerate.state import PartialState
from utils.configs import DataArguments, H4ArgumentParser
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorForSeq2Seq
)

from transformers.trainer_utils import EvalPrediction, get_last_checkpoint
from trl import (
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from utils import get_datasets

tqdm.pandas()

logger = colorlog.getLogger(__name__)

if __name__ == "__main__":
    parser = H4ArgumentParser((DataArguments, ModelConfig, TrainingArguments))
    data_args, model_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    fmt_string = '%(log_color)s %(asctime)s - %(levelname)s - %(message)s'
    log_colors = {
            'DEBUG': 'white',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'purple'
            }
    log_level = training_args.get_process_log_level()
    colorlog.basicConfig(
        log_colors=log_colors, 
        format=fmt_string, 
        handlers=[colorlog.StreamHandler(sys.stdout)],
        level = log_level
    )
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")


    ################
    # Model & Tokenizer
    ################
    # MODEL
    logger.info("*** Loading pretrained model and tokenizer ***")

    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        low_cpu_mem_usage=False,
    )
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_fast=True, 
        trust_remote_code=model_args.trust_remote_code
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 32000
    assert tokenizer.chat_template is not None, "Needs chat template!"

    # Label and Mask Tokens 
    GOOD_TOKEN='<+>'
    BAD_TOKEN='<->'
    SEPERATOR_TOKEN='<extra>'

    ################
    # Dataset
    ################
    logger.info("*** Loading datasets ***")
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        # columns_to_keep=["inputs", 'labels'],
        columns_to_keep=['labels', 'input_ids', 'attention_mask'],
    )
    
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    logger.info(f"Column Names: {column_names}")

    train_dataset = raw_datasets['train'] 
    eval_dataset = raw_datasets['test'] 
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)

    ################
    # Evaluation Metric
    ################

    correct_preds = 0
    total_preds = 0

    POS_ID = tokenizer.convert_tokens_to_ids(GOOD_TOKEN)
    NEG_ID = tokenizer.convert_tokens_to_ids(BAD_TOKEN)

    def batch_compute_token_accuracy(eval_pred: EvalPrediction, compute_result: bool):
        global correct_preds, total_preds
        IGNORE_INDEX = -100

        # Unpack predictions and labels
        logits = eval_pred.predictions  # shape: (batch_size, seq_len, vocab_size)
        labels = eval_pred.label_ids    # shape: (batch_size, seq_len)

        # Get predicted token IDs via argmax
        pos_logits = logits[..., POS_ID]  # (batch_size, seq_len)
        neg_logits = logits[..., NEG_ID]  # (batch_size, seq_len)

        # pred_ids = torch.argmax(torch.tensor(logits), dim=-1)  # shape: (batch_size, seq_len)
        pred_ids = torch.where(pos_logits > neg_logits, POS_ID, NEG_ID)
        label_ids = torch.tensor(labels)

        # Mask to select only supervised tokens (e.g., <+> positions)
        mask = label_ids != IGNORE_INDEX

        correct = (pred_ids == label_ids) & mask

        # Accumulate total correct and total considered
        correct_preds += correct.sum().item()
        total_preds += mask.sum().item()

        if compute_result:
            token_accuracy = correct_preds / total_preds if total_preds > 0 else 0.0

            # Reset for next eval round
            correct_preds = 0
            total_preds = 0

            return {"token_accuracy": token_accuracy}
        else:
            return {}

    
    class PrintMetricsCallback(TrainerCallback):
        def on_evaluate(self, args:TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if state.is_world_process_zero:
                logs = state.log_history[-1]
                if "eval_token_accuracy" in logs:
                    logger.critical(f"Token Accuracy: {logs['eval_token_accuracy']}")

    logger.critical(tokenizer.model_max_length)

    ################
    # Initialize the Trainer
    ################

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=batch_compute_token_accuracy,
        )


    ###############
    # Training loop
    ###############
    logger.info("*** Training ***")
    checkpoint = None
    # Check for last checkpoint
    if training_args.resume_from_checkpoint is not None:
        checkpoint = get_last_checkpoint(training_args.output_dir) if isinstance(training_args.resume_from_checkpoint, bool) else training_args.resume_from_checkpoint
        if checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training at {checkpoint=}.")
        else:
            logger.error(f"Failed to load last checkpoint at {checkpoint=}. Start training from scratch")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


    ##########
    # Evaluate
    ##########
    torch.cuda.empty_cache()
    if training_args.do_eval:
        logger.info("*** Evaluating ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    logger.info("*** Evaluating complete ***")
