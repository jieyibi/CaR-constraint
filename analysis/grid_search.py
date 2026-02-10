"""
Grid search script for VRP/TSPTW(-like) experiments using `test.py`.

Usage:
    python grid_search.py

What it does:
    - Iterates over combinations of (S, p, t, seed) specified via CLI arguments:
        * You pass comma-separated lists (e.g., --S_list 1,2,4) to define the grid.
    - For each (S, p, t, seed), it:
        * Builds and runs `test.py` with the corresponding CLI arguments, including `--problem=<problem>`
        * Parses stdout to extract:
            - NO-AUG / AUGMENTATION scores and gaps
            - Solution / instance infeasibility rates
            - Evaluation time
        * Appends one row of metrics into the CSV file `output_csv`

How to customize:
    - Pass different `--S_list`, `--p_list`, `--t_list`, and `--seed_list` values from the command line.
    - Change `--gpu_id` to select which GPU to use.
    - Change `problem` (default: "TSPTW") to run on a different problem type supported by `test.py`.
    - Change `output_csv` to a different file name if you want to log to another CSV.
    - Make sure that `test.py` is in the same directory (or adjust the command path).
"""

import argparse
import itertools
import subprocess
import re
import csv
import os

def estimate_batch_size(s, p):
    sp = s * p
    return 10000 // sp

def run_grid_search(args):
    problem = args.problem
    gpu_id = args.gpu_id

    # Build value lists for grid search from comma-separated CLI strings
    S_values = [int(x) for x in args.S_list.split(",") if x.strip()]
    p_values = [int(x) for x in args.p_list.split(",") if x.strip()]
    t_values = [int(x) for x in args.t_list.split(",") if x.strip()]
    seeds = [int(x) for x in args.seed_list.split(",") if x.strip()]

    # Regular expressions
    no_aug_pattern = re.compile(r"NO-AUG SCORE:\s*([\d\.]+),\s*Gap:\s*([-]?\d*\.?\d+)")
    aug_pattern = re.compile(r"AUGMENTATION SCORE:\s*([\d\.]+),\s*Gap:\s*([-]?\d*\.?\d+)")
    time_pattern = re.compile(r"Evaluation finished within\s*([\d\.]+)s")
    solution_infeasible_pattern = re.compile(r"Solution level Infeasible rate:\s*([\d\.]+)%")
    instance_infeasible_pattern = re.compile(r"Instance level Infeasible rate:\s*([\d\.]+)%")

    # Section flags in log output
    construction_flag = "*** Construction ***"
    improvement_flag = "*** Improvement ***"
    rc_flag = "*** Re-Construction w. mask ***"

    # Output CSV file
    output_csv = args.output_csv
    write_header = not os.path.exists(output_csv)

    with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "S", "p", "t", "test_batch_size",
                "construction_no_aug_score", "construction_no_aug_gap",
                "construction_aug_score", "construction_aug_gap",
                "construction_solution_inf", "construction_instance_inf",
                "improvement_no_aug_score", "improvement_no_aug_gap",
                "improvement_aug_score", "improvement_aug_gap",
                "improvement_solution_inf", "improvement_instance_inf",
                "reconstruction_no_aug_score", "reconstruction_no_aug_gap",
                "reconstruction_aug_score", "reconstruction_aug_gap",
                "reconstruction_solution_inf", "reconstruction_instance_inf",
                "eval_time",
            ])

        # Grid search over (S, p, t, seed)
        for s, p, t, seed in itertools.product(S_values, p_values, t_values, seeds):
            test_batch_size = estimate_batch_size(s, p)

            cmd = [
                "python", "../test.py",
                f"--problem={problem}",
                f"--validation_improve_steps={t}",
                f"--select_top_k_val={p}",
                f"--sample_size={s}",
                f"--test_batch_size={test_batch_size}",
                f"--gpu_id={gpu_id}",
                f"--eval_type=softmax",
                f"--seed={seed}",
            ]

            print("Running command:", " ".join(cmd))
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            lines = result.stdout.splitlines()

            # Initialize variables for metrics
            construction_no_aug_score = construction_no_aug_gap = None
            construction_aug_score = construction_aug_gap = None
            construction_solution_inf = construction_instance_inf = None

            improvement_no_aug_score = improvement_no_aug_gap = None
            improvement_aug_score = improvement_aug_gap = None
            improvement_solution_inf = improvement_instance_inf = None

            reconstruction_no_aug_score = reconstruction_no_aug_gap = None
            reconstruction_aug_score = reconstruction_aug_gap = None
            reconstruction_solution_inf = reconstruction_instance_inf = None

            eval_time = None

            in_construction_section = False
            in_improvement_section = False
            in_rc_section = False

            for line in lines:
                line_stripped = line.strip()

                match_time = time_pattern.search(line_stripped)
                if match_time and eval_time is None:
                    eval_time = float(match_time.group(1))

                if line_stripped.startswith(construction_flag):
                    in_construction_section = True
                    in_improvement_section = False
                    in_rc_section = False
                    continue

                if line_stripped.startswith(improvement_flag):
                    in_construction_section = False
                    in_improvement_section = True
                    in_rc_section = False
                    continue

                if line_stripped.startswith(rc_flag):
                    in_construction_section = False
                    in_improvement_section = False
                    in_rc_section = True
                    continue

                if not (in_construction_section or in_improvement_section or in_rc_section):
                    continue

                match_no_aug = no_aug_pattern.search(line_stripped)
                if match_no_aug:
                    score_val = float(match_no_aug.group(1))
                    gap_val = float(match_no_aug.group(2))
                    if in_construction_section:
                        construction_no_aug_score = score_val
                        construction_no_aug_gap = gap_val
                    elif in_improvement_section:
                        improvement_no_aug_score = score_val
                        improvement_no_aug_gap = gap_val
                    elif in_rc_section:
                        reconstruction_no_aug_score = score_val
                        reconstruction_no_aug_gap = gap_val
                    continue

                match_aug = aug_pattern.search(line_stripped)
                if match_aug:
                    score_val = float(match_aug.group(1))
                    gap_val = float(match_aug.group(2))
                    if in_construction_section:
                        construction_aug_score = score_val
                        construction_aug_gap = gap_val
                    elif in_improvement_section:
                        improvement_aug_score = score_val
                        improvement_aug_gap = gap_val
                    elif in_rc_section:
                        reconstruction_aug_score = score_val
                        reconstruction_aug_gap = gap_val
                    continue

                match_sol_inf = solution_infeasible_pattern.search(line_stripped)
                if match_sol_inf:
                    rate = float(match_sol_inf.group(1))
                    if in_construction_section:
                        construction_solution_inf = rate
                    elif in_improvement_section:
                        improvement_solution_inf = rate
                    elif in_rc_section:
                        reconstruction_solution_inf = rate
                    continue

                match_inst_inf = instance_infeasible_pattern.search(line_stripped)
                if match_inst_inf:
                    rate = float(match_inst_inf.group(1))
                    if in_construction_section:
                        construction_instance_inf = rate
                    elif in_improvement_section:
                        improvement_instance_inf = rate
                    elif in_rc_section:
                        reconstruction_instance_inf = rate
                    continue

            writer.writerow([
                s, p, t, test_batch_size,
                construction_no_aug_score, construction_no_aug_gap,
                construction_aug_score, construction_aug_gap,
                construction_solution_inf, construction_instance_inf,
                improvement_no_aug_score, improvement_no_aug_gap,
                improvement_aug_score, improvement_aug_gap,
                improvement_solution_inf, improvement_instance_inf,
                reconstruction_no_aug_score, reconstruction_no_aug_gap,
                reconstruction_aug_score, reconstruction_aug_gap,
                reconstruction_solution_inf, reconstruction_instance_inf,
                eval_time,
            ])

            print([
                s, p, t, test_batch_size,
                construction_no_aug_score, construction_no_aug_gap,
                construction_aug_score, construction_aug_gap,
                construction_solution_inf, construction_instance_inf,
                improvement_no_aug_score, improvement_no_aug_gap,
                improvement_aug_score, improvement_aug_gap,
                improvement_solution_inf, improvement_instance_inf,
                reconstruction_no_aug_score, reconstruction_no_aug_gap,
                reconstruction_aug_score, reconstruction_aug_gap,
                reconstruction_solution_inf, reconstruction_instance_inf,
                eval_time,
            ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grid search for VRP/TSPTW(-like) experiments using `test.py`")
    parser.add_argument("--problem", type=str, default="TSPTW")
    parser.add_argument("--S_list", type=str, default="1,2,4", help="Comma-separated list of S values, e.g. '1,2,4'")
    parser.add_argument("--p_list", type=str, default="1", help="Comma-separated list of p values")
    parser.add_argument("--t_list", type=str, default="5,10,20", help="Comma-separated list of t values")
    parser.add_argument("--seed_list", type=str, default="2024", help="Comma-separated list of seed values")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_csv", type=str, default="output.csv")
    args = parser.parse_args()

    print(f"Running grid search for {args.problem} with "
          f"S_list={args.S_list}, p_list={args.p_list}, t_list={args.t_list}, seed_list={args.seed_list}, "
          f"gpu_id={args.gpu_id}")
    print(f"Writing results to {args.output_csv}")

    run_grid_search(args)
