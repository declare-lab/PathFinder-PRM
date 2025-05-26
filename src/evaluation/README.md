# PathFinder-PRM Evaluation

This directory contains the evaluation setup for evaluating PathFinder-PRM.

## 1. PRMBench

**PRMBench** evaluates fine-grained error detection capabilities of Process Reward Models across multiple dimensions: Simplicity, Soundness, and Sensitivity, with 11 total error categories.

### Quick Start

#### 1. Basic Evaluation Run

```bash
chmod +x prm_bench_eval.sh
./prm_bench_eval.sh
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

### PRMBench Configuration Guide

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

## 2. ProcessBench

**ProcessBench** is a benchmark designed to evaluate language models' ability to identify errors in mathematical reasoning processes. It comprises 3,400 test cases, primarily sourced from different math reasoning benchmarks.

### Running ProcessBench Evaluation
To run ProcessBench Evaluation on PathFinder-PRM, you can use the following commands:

```bash
bash process_bench_eval.sh
```

This will run the evaluations on PathFinder-PRM for all 4 subsets of ProcessBench and save the outputs and results under `./ProcessBench/code/outputs/PathFinder-PRM-7B/`

## 3. Reward-Guided Greedy Search

Reward-Guided Greedy Search is an inference-time procedure for measuring how well a Process Reward Model (PRM) can guide step-by-step generation toward high-quality outputs. At each decoding step, the policy model generates 8 candidate continuations; the PRM then scores each partial trace according to its expected downstream correctness, and the highest-scoring candidate is greedily selected. By repeating this “propose → score → select” cycle until completion, Reward-Guided Greedy Search provides a clear metric of a PRM's performance in guiding the policy model towards correct solutions. In our setup, we use `Qwen/Qwen2.5-7B-Instruct` as the policy model. We conducted this evaluation across several widely recognized math reasoning benchmarks, including AIME24, AMC23, MATH, Olympiad Bench, College MATH, and Minerva MATH.

### Running evaluation
To run Reward-Guided Greedy Search and evaluate the final generations, run the following command:

```bash
bash reward_guided_search_eval.sh
```

This will run the evaluations on PathFinder-PRM for all 6 math benchmarks and save the outputs and results under `./reward_guided_search/outputs/PathFinder-PRM-7B/`

## Acknowledgements

This evaluation setup builds upon the work of several open-source repositories. We gratefully acknowledge the authors and contributors of the following projects:

- [R-PRM](https://github.com/NJUNLP/R-PRM/tree/main): Evaluation Setup for Reward-Guided Greedy Search  
- [ProcessBench](https://github.com/QwenLM/ProcessBench): Benchmark and Evaluation Setup for ProcessBench
- [PRMBench](https://github.com/ssmisya/PRMBench/tree/main): Benchmark and Evaluation Setup for PRMBench
- [Math-Verify](https://github.com/huggingface/Math-Verify): Tool to evaluate math responses

Please refer to their respective repositories for license and contribution details.


## Citation

If you found our evaluation setup helpful, please cite:

```bibtex
@misc{pala2025errortypingsmarterrewards,
      title={Error Typing for Smarter Rewards: Improving Process Reward Models with Error-Aware Hierarchical Supervision}, 
      author={Tej Deep Pala and Panshul Sharma and Amir Zadeh and Chuan Li and Soujanya Poria},
      year={2025},
      eprint={2505.12345},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.12345}, 
}
```
