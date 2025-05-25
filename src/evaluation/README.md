# PathFinder-PRM Evaluation

This directory contains the evaluation setup for evaluating PathFinder-PRM. PRMBench evaluates Process Reward Models across multiple dimensions: Simplicity, Soundness, and Sensitivity.

## 1. PRMBench

PRMBench evaluates fine-grained error detection capabilities of Process Reward Models across 11 different error categories.

### Prerequisites
We recommend running the evaluation with the equivalent of 1 H100 GPU.
#### Dependencies

```bash
cd src/evaluation/PRMBench
pip install -r requirements.txt
```

**Key dependencies:**

- `accelerate>=1.1.1` (for distributed evaluation)
- `deepspeed` (for memory optimization)
- `transformers` (for model loading)
- `torch` with CUDA support

Please install these according to your system requirements.

### Model Access

The PathFinder-PRM model is available on HuggingFace at `declare-lab/PathFinder-PRM-7B`.
### Quick Start

#### 1. Basic Evaluation Run

```bash
cd src/evaluation
chmod +x prm_bench.sh
./prm_bench.sh
```

This will:
- Load PathFinder-PRM-7B model
- Run evaluation on PRMBench dataset
- Save results to `./log/disc_prm_4gpu_output.jsonl`

#### 2. Extract Results

```bash
cd src/evaluation
python results_prm_bench.py
```

This will output:

- Detailed scores by category (Simplicity, Soundness, Sensitivity)
- Overall PRMScore
- LaTeX-formatted results for easy copying

### Configuration Guide

#### Model Configuration (`disc_prm_eval.yaml`)

```yaml
model_args:
  model: disc_PRM                                    # Model identifier
  model_args: pretrained=declare-lab/PathFinder-PRM-7B  # Model path/name
  batch_size: 1                                      # Batch size per GPU
  num_workers: 0                                     # DataLoader workers

task_args:
  task_name: prmtest_classified                      # PRMBench task

save_to_ckpt:
  prmtest_classified: ./log/ckpt/disc_prm_4gpu_resume.jsonl  # Checkpoint path

script_args:
  verbosity: INFO                                    # Logging level
  output_path: ./log/disc_prm_4gpu_output.jsonl     # Results output path
```

#### Key Parameters to Modify:

**Model Selection:**

```yaml
model_args: pretrained=your-model-path-or-name
```

**Batch Size Optimization:**

```yaml
batch_size: 2  # Increase if you have sufficient GPU memory
```

**Output Paths:**

```yaml
output_path: ./log/your_custom_output.jsonl
save_to_ckpt:
  prmtest_classified: ./log/ckpt/your_checkpoint.jsonl
```

#### Hardware Configuration (`deepspeed_run_1.yaml`)

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: /path/to/zero3_inference.json
num_processes: 1    # Adjust based on your GPU count
```

#### Multi-GPU Setup:

```yaml
num_processes: 4    # For 4 GPUs
machine_rank: 0     # Keep as 0 for single machine
```

#### ZeR0-3 inference config (`zero3_inference.json`)

The DeepSpeed ZeRO-3 configuration optimizes memory usage:

```json
{
  "zero_optimization": {
    "stage": 3,                              # ZeRO-3 for maximum memory savings
    "stage3_max_live_parameters": 5e7,       # Adjust based on model size
    "stage3_max_reuse_distance": 5e7,        # Memory reuse distance
    "offload_param": {
      "device": "cpu",                       # Offload parameters to CPU
      "pin_memory": true
    }
  },
  "train_micro_batch_size_per_gpu": 1        # Micro-batch size
}
```

```
## Citation

If you use this evaluation setup, please cite:

```bibtex

```