# Training

This directory contains the training scripts for **PathFinder-PRM**. Configuration files for training hyperparameters and Accelerate/DeepSpeed integration are located in the `training_configs` directory. These scripts are optimized for training with `Qwen/Qwen2.5-7B-Instruct` as the base model.

---

## 1. Preparing the Model and Tokenizer

Before starting training, the model and tokenizer must be customized to include the special tokens `<+>`, `<->`, and `<extra>`. This also requires adjusting the model's embedding layer accordingly.

To perform this setup, run:

```bash
python prepare_model.py
```

---


## 2. Downloading and Processing the Training Dataset

The PathFinder training dataset, **PathFinder-600K**, is available on Hugging Face with the dataset ID: [`declare-lab/PathFinder-600K`](https://huggingface.co/datasets/declare-lab/PathFinder-600K).

To download and preprocess the dataset:

```bash
python prepare_train_dataset.py
```

---

## 3. Configuring Training Hyperparameters

Before launching training, you should review and adjust the configuration file, `training_configs/prm_train.yaml`.

Common parameters to update include:
* `output_dir`: Path to save model checkpoints and logs.
* `dataset_path`: Path to the preprocessed dataset.
* Other training hyperparameters like learning rate, batch size, and number of epochs.

We use **DeepSpeed** for memory optimization during training. You can refer to `training_configs/deepspeed_zero2.yaml` which contains the DeepSpeed configuration used for training `PathFinder-PRM-7B`. Modify it as needed based on your hardware setup.

---

## 4. Launching Training

Once all configurations are set, start training with:

```bash
bash train_prm.sh
```