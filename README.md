<h1 align="center"> Towards Efficient Constraint Handling in Neural Solvers for Routing Problems </h1>

<p align="center">
<a href="https://iclr.cc/"><img alt="License" src="https://img.shields.io/static/v1?label=ICLR'26&message=Rio de Janeiro&color=purple&style=flat-square"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://openreview.net/forum?id=raDFGuQxvD"><img alt="License" src="https://img.shields.io/static/v1?label=OpenReview&message=Paper&color=blue&style=flat-square"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://github.com/jieyibi/CaR-constraint/blob/master/README.md"><img 
alt="License" src="https://img.shields.io/static/v1?label=License&message=MIT&color=rose&style=flat-square"></a>
</p>

---

Hi there! Thanks for your attention to our work!🤝

This is the PyTorch code for the **Construct-and-Refine (CaR)** framework for neural routing solvers.

CaR is the first general and efficient constraint-handling framework for neural routing solvers based on explicit learning-based feasibility refinement. Unlike prior construction-search hybrids that target reducing optimality gaps through heavy improvements yet still struggle with hard constraints, CaR achieves efficient constraint handling by designing a joint training framework that guides the construction module to generate diverse and high-quality solutions well-suited for a lightweight improvement process, e.g., 10 steps versus 5k steps in prior work. Moreover, CaR presents the first use of construction-improvement-shared representation, enabling potential knowledge sharing across paradigms by unifying the encoder, especially in more complex constrained scenarios.

For more details, please see our paper [Towards Efficient Constraint Handling in Neural Solvers for Routing Problems](https://openreview.net/forum?id=raDFGuQxvD), which has been accepted at ICLR 2026😊. If you find our work useful, please cite:

```
@inproceedings{
    bi2026towards,
    title={Towards Efficient Constraint Handling in Neural Solvers for Routing Problems},
    author={Bi, Jieyi and Cao, Zhiguang and Zhou, Jianan and Song, Wen and Wu, Yaoxin and Zhang, Jie and Ma, Yining and Wu, Cathy},
    booktitle={International Conference on Learning Representations},
    year={2026}
}
```

## Overview

<p align="center">
  <img src="plot/overall.png" alt="CaR Framework Overview" width="800"/>
</p>

---
## How to play with CaR
You could follow the command below to clone our repo and set up the running environment. 

```bash
git clone git@github.com:jieyibi/CaR-constraint.git
cd CaR-constraint
conda create -n car python=3.12
conda activate car
pip3 install torch torchvision torchaudio # Note: Please refer to your CUDA version and the official website of PyTorch!
pip install matplotlib tqdm pytz scikit-learn tensorflow tensorboard_logger pandas wandb

# Decompress dataset files (if needed)
cd data/SOP
gunzip sop50_uniform_variant1.pkl.gz sop50_uniform_variant2.pkl.gz
cd ../..
```


---



## Usage

<details>
    <summary><strong>Train</strong></summary>

```shell
# Default: --problem=TSPTW --problem_size=50 --pomo_size=50

# 1. CaR (Construction + Refinement)
python train.py --problem={PROBLEM} --problem_size={PROBLEM_SIZE} --improve_steps={IMPROVE_STEPS}

# Note: 
# 1. If you want to resume, please add arguments: `--checkpoint` and `--load_optimizer`
# 2. Key parameters:
#    --improve_steps: number of improvement steps during training (default: 5)
#    --validation_improve_steps: number of improvement steps during validation (default: 20)
#    --unified_encoder: whether to use shared encoder for construction and improvement (default: True)
#    --improvement_method: "rm_n_insert" or "kopt" (default: "rm_n_insert")
#    --select_top_k: number of solutions to select for improvement (default: 10)

# 2. Construction only (baseline)
python train.py --problem={PROBLEM} --problem_size={PROBLEM_SIZE} --improve_steps=0
```
</details> 

<details>
    <summary><strong>Evaluation</strong></summary>

```shell
# Default: --problem=TSPTW --problem_size=50

# 1. CaR (Construction + Refinement)

# If you want to evaluate on your own dataset,
python test.py --test_dataset={TEST_DATASET} --checkpoint={MODEL_PATH} --improve_steps={IMPROVE_STEPS} --validation_improve_steps={VAL_IMPROVE_STEPS}
# Optional: add `--opt_sol_path` to calculate optimality gap.

# If you want to evaluate on the provided dataset,
python test.py --problem={PROBLEM} --problem_size={PROBLEM_SIZE} --checkpoint={MODEL_PATH} --improve_steps={IMPROVE_STEPS} --validation_improve_steps={VAL_IMPROVE_STEPS}

# 2. Construction only

# If you want to evaluate on your own dataset,
python test.py --test_dataset={TEST_DATASET} --checkpoint={MODEL_PATH} --improve_steps=0
# Optional: add `--opt_sol_path` to calculate optimality gap.

# If you want to evaluate on the provided dataset,
python test.py --problem={PROBLEM} --problem_size={PROBLEM_SIZE} --checkpoint={MODEL_PATH} --improve_steps=0


# Please set your own `--test_batch_size` based on your GPU memory constraint.
```

</details> 

<details>
    <summary><strong>Generate data</strong></summary>

For evaluation, you can use our [provided datasets](https://github.com/jieyibi/CaR-constraint/tree/main/data) or generate data by running the following command:

```shell
# Default: --problem="TSPTW" --problem_size=50 --num_samples=10000 --hardness="hard"
python generate_data.py --problem={PROBLEM} --problem_size={PROBLEM_SIZE} --num_samples={NUM_SAMPLES} --hardness={HARDNESS}
```

Supported problems: `CVRP`, `TSPTW`, `TSPDL`, `VRPBLTW`, `SOP`

For SOP, you can specify:
- `--sop_variant`: `1` or `2` (default: `1`, see appendix for detailed settings)

</details>  


<details>
    <summary><strong>Baseline</strong></summary>

#### 1. LKH3 

```shell
# Default: --problem="TSPTW" --datasets="../data/TSPTW/tsptw50_hard.pkl"
python baselines/LKH_baseline.py --problem={PROBLEM} --datasets={DATASET_PATH} -n=10000 -runs=1 -max_trials=10000
```


#### 2. OR-Tools
```shell
# Default: --problem="TSPTW" --datasets="../data/TSPTW/tsptw50_hard.pkl"
python baselines/OR-Tools_baseline.py --problem={PROBLEM} --datasets={DATASET_PATH} -n=10000 -timelimit=200
# Optional arguments: `--cal_gap --opt_sol_path={OPTIMAL_SOLUTION_PATH}`
```


#### 3. HGS
```shell
# Default: --problem="CVRP" --datasets="../data/CVRP/cvrp50_uniform.pkl" -n=1000 -max_iteration=20000
python baselines/HGS_baseline.py --problem={PROBLEM} --datasets={DATASET_PATH} -n=1000 -max_iteration=20000
```


#### 4. Greedy
##### 4.1 Greedy-L
```shell
# Default: --problem="TSPTW" --datasets="../data/TSPTW/tsptw50_hard.pkl"
python baselines/greedy_parallel.py --problem={PROBLEM} --datasets={DATASET_PATH} --heuristics="length"
# Optional arguments: `--cal_gap --optimal_solution_path={OPTIMAL_SOLUTION_PATH}`
```

##### 4.2 Greedy-C
```shell
# Default: --problem="TSPTW" --datasets="../data/TSPTW/tsptw50_hard.pkl" 
python baselines/greedy_parallel.py --problem={PROBLEM} --datasets={DATASET_PATH} --heuristics="constraint"
# Optional arguments: `--cal_gap --optimal_solution_path={OPTIMAL_SOLUTION_PATH}`
```

</details>



---

## Supported Problems

- **CVRP**: Capacitated Vehicle Routing Problem
- **TSPTW**: Traveling Salesman Problem with Time Windows
- **TSPDL**: Traveling Salesman Problem with Deadlines
- **VRPBLTW**: Vehicle Routing Problem with Backhauls and Time Windows
- **SOP**: Sequential Ordering Problem

---

## Acknowledgments
The code and the framework are based on our previous repo: [PIP-constraint](https://github.com/jieyibi/PIP-constraint).
If you find it useful, please cite:
```
@inproceedings{
    bi2024learning,
    title={Learning to Handle Complex Constraints for Vehicle Routing Problems},
    author={Bi, Jieyi and Ma, Yining and Zhou, Jianan and Song, Wen and Cao, 
    Zhiguang and Wu, Yaoxin and Zhang, Jie},
    booktitle = {Advances in Neural Information Processing Systems},
    year={2024}
}
```