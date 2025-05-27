# Data Generation for PathFinder-PRM

This directory contains the scripts to replicate our described data generation process. The final result is a 400K-sample dataset enriched with three-dimensional step-level labels.

## Overview

The PathFinder-PRM dataset combines two sources:

- **PRM800K**: Human-annotated math reasoning dataset 
- **RLHFlow Mistral Data**: Automated annotations from Mistral-7B

Each sample gets three-dimensional categorical scores:

- **Score A (Math)**: Mathematical reasoning accuracy (0/1)
- **Score B (Consistency)**: Logical consistency with prior steps (0/1)
- **Score C (Correctness)**: Step correctness and optimality (0/1)

Scores of -1 correspond to invalid samples.
## Prerequisites

### Required Dependencies

```bash
pip install torch transformers vllm datasets tqdm
```

### Required Model Access

- DeepSeek-R1-Distill-Qwen-32B (for score annotation)
- Sufficient GPU memory for VLLM inference (recommended: 4 or more GPUs)

### Required Data Files

#### For PRM800K:

Download these files from [OpenAI PRM800K](https://github.com/openai/prm800k) and place in `./dataset/`:

```
dataset/
├── phase1_train.jsonl
├── phase1_test.jsonl
├── phase2_train.jsonl
└── phase2_test.jsonl
```

#### For Mistral Data:

Download the RLHFlow Mistral dataset from [huggingface](https://huggingface.co/datasets/RLHFlow/Mistral-PRM-Data) and place as:

```
dataset/
└── mistral_dataset.json
```

## Data labelling steps

### Step 1: Process PRM800K Data

```bash
# Load and process the PRM800K dataset
python load_data.py
```

This script:

- Loads the 4 PRM800K jsonl files
- Converts them to a unified format with context, question, prev_steps, curr_step
- Saves to `./dataset/processed_dataset.json`

### Step 2: Process Mistral Data

```bash
# Load and process the Mistral dataset
python load_data_mistral.py dataset/mistral_dataset.json
```

This script:

- Processes the Mistral conversation format
- Extracts questions, steps, and Mistral scores
- Saves to `./dataset/processed_mistral_dataset.json`

### Step 3: Generate 3D Labels for PRM800K

```bash
# Run inference to generate categorical scores for PRM800K data
python run_inference.py
```

This script:

- Uses DeepSeek-R1-Distill-Qwen-32B to annotate score_A, score_B, score_C
- For human_score=1: assigns (1,1,1)
- For human_score=0: assigns (1,1,0)
- For human_score=-1: runs LLM inference with consistency checks
- Saves to `./dataset/Disc_PRM_ds.json`

**Note**: This step requires significant GPU resources and may take several hours.

### Step 4: Generate 3D Labels for Mistral Data

```bash
# Run inference to generate categorical scores for Mistral data  
python run_inference_mistral.py
```

This script:

- Runs LLM inference on sampled Mistral data (150K each of score 0/1)
- Implements disagreement filtering between Mistral scores and model predictions
- Saves to `./dataset/processed_mistral_dataset_scored.json`

**Note**: This step requires significant GPU resources and may take several hours.

### Step 5: Clean Mistral Data

```bash
# Clean step labels from Mistral data
python clean_mistral.py
```

This script:

- Removes "Step X:" prefixes from prev_steps and curr_step fields
- Saves cleaned data to `./dataset/processed_mistral_dataset_scored_cleaned.json`

### Step 6: Build Final Training Dataset

```bash
# Combine PRM800K and Mistral data into final training format
python build_dataset.py
```

This script:

- Loads both processed datasets
- Converts to training format with hierarchical prediction structure
- Creates two samples per record:
    1. Math reasoning + Consistency prediction
    2. Final correctness prediction conditioned on error labels
- Saves to `./dataset/Disc_PRM_full_dataset/`

### Step 7: Build Ablation Dataset (Optional)

```bash
# Build PRM800K-only dataset for ablation studies
python build_dataset_ablation.py  
```

This creates a dataset using only PRM800K data for comparison. It is possible to generate data with other ablations by modifying build_dataset_ablation.py according the ablation you would like to test model training with.

## Configuration Notes

### Hardware Requirements

- The inference scripts are configured for 8x GPU setup
- Adjust `tensor_parallel_size` in the inference scripts based on your hardware
- Modify `BATCH_SIZE` and memory settings as needed

### Key Configuration Variables

In `run_inference.py` and `run_inference_mistral.py`:

```python
BATCH_SIZE = 1024  # Adjust based on GPU memory
tensor_parallel_size = 8  # Number of GPUs
gpu_memory_utilization = 0.97  # GPU memory usage
```
## Output Format

The final training dataset contains records with:

```json
{
  "labels": [
    {
      "role": "user", 
      "content": "System prompt + Question: [problem]"
    },
    {
      "role": "assistant", 
      "content": "[context] Math reasoning: <+>, Consistency: <+>, Correctness: <+>"
    }
  ],
  "inputs": [
    {
      "role": "user",
      "content": "System prompt + Question: [problem]" 
    },
    {
      "role": "assistant",
      "content": "[context] Math reasoning: <extra>, Consistency: <extra>"
    }
  ]
}
```
## Troubleshooting

### Common Issues

1. **GPU Memory Errors**: Reduce `BATCH_SIZE` or `max_num_batched_tokens`
2. **Score Extraction Failures**: Check regex patterns in inference scripts
3. **Missing Files**: Ensure all prerequisite data files are downloaded

## Citation

If you use this data generation pipeline, please cite:

```bibtex
@misc{pala2025errortypingsmarterrewards,
      title={Error Typing for Smarter Rewards: Improving Process Reward Models with Error-Aware Hierarchical Supervision}, 
      author={Tej Deep Pala and Panshul Sharma and Amir Zadeh and Chuan Li and Soujanya Poria},
      year={2025},
      eprint={2505.19706},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.19706}, 
}
```
