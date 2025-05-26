from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "./models/Qwen2.5-Math-7B-Instruct-updated",
    use_fast = True
)

raw_datasets = load_dataset("declare-lab/PathFinder-600K")
column_names = list(raw_datasets["train"].features)

def batch_preprocess(batch):
    input_convs = batch['inputs']
    label_convs = batch['labels']

    input_enc = tokenizer.apply_chat_template(
        input_convs,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        padding=False,
        max_length=tokenizer.model_max_length,
        return_dict=True
    )

    label_enc = tokenizer.apply_chat_template(
        label_convs,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        padding=False,
        max_length=tokenizer.model_max_length,
        return_dict=True
    )

    input_ids = input_enc["input_ids"]
    label_ids = label_enc["input_ids"]

    labels = [
        [-100 if i == j else j for i, j in zip(in_ids, lbl_ids)]
        for in_ids, lbl_ids in zip(input_ids, label_ids)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": input_enc["attention_mask"],
        "labels": labels
    }

train_dataset = raw_datasets["train"].map(
    batch_preprocess, 
    batched=True, 
    remove_columns=column_names,
    num_proc=None,
    batch_size=256
    )

eval_dataset = raw_datasets["test"].map(
    batch_preprocess, 
    batched=True, 
    remove_columns=column_names,
    num_proc=None,
    batch_size=256
    )

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": eval_dataset
})

save_path = "./train_data/PathFinder_600k_tokenized_qwen_25_7B"

# Save the DatasetDict to disk
dataset_dict.save_to_disk(save_path)

print(f"Combined dataset saved to: {save_path}")
