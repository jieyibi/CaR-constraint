import os, random, math, time
import argparse
import pprint as pp
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets")
    parser.add_argument('--problem', type=str, default="SOP", choices=["CVRP", "TSPTW", "TSPDL", "VRPBLTW", "SOP"])
    parser.add_argument('--limited_vehicle_number', type=int, default=0, choices=["adaptive", "plus1"])
    parser.add_argument('--problem_size', type=int, default=50)
    parser.add_argument('--pomo_size', type=int, default=50, help="the number of start node, should <= problem size")
    parser.add_argument('--hardness', type=str, default="hard", choices=["hard", "medium", "easy"], help="hardness level: hard/medium/easy for TSPTW and TSPDL (default: hard)")
    parser.add_argument('--sop_variant', type=int, default=1, choices=[1, 2], help="SOP variant: 1 or 2 (default: 1). Variant 1: precedence_ratio=0.2, geometric_conflict_ratio=0.3, precedence_balance_ratio=0. Variant 2: precedence_ratio=0.2, geometric_conflict_ratio=0.8, precedence_balance_ratio=0")
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--dir', type=str, default="./data")
    parser.add_argument('--no_cuda', action='store_true', default=True)
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()
    pp.pprint(vars(args))
    
    env_params = {
        "problem_size": args.problem_size,
        "pomo_size": args.pomo_size,
        "hardness": args.hardness,
        "sop_variant": args.sop_variant,
        "limited_vehicle_number": args.limited_vehicle_number,
    }
    seed_everything(args.seed)

    # set log & gpu
    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda', args.gpu_id)
        torch.cuda.set_device(args.gpu_id)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        args.device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
    print(">> SEED: {}, USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(args.seed, not args.no_cuda, args.gpu_id))

    envs = get_env(args.problem)
    for env in envs:
        env = env(**env_params)
        if args.problem == "TSPTW":
            dataset_path = os.path.join(args.dir, env.problem, f"{env.problem.lower()}{args.problem_size}_{args.hardness}.pkl",)
        elif args.problem == "TSPDL":
            dataset_path = os.path.join(args.dir, env.problem, f"{env.problem.lower()}{args.problem_size}_{args.hardness}.pkl",)
        elif args.problem == "SOP":
            dataset_path = os.path.join(args.dir, env.problem, f"{env.problem.lower()}{args.problem_size}_uniform_variant{args.sop_variant}.pkl",)
        elif args.problem == "CVRP" and args.limited_vehicle_number is not None:
            dataset_path = os.path.join(args.dir, env.problem, f"{env.problem.lower()}{args.problem_size}_uniform_LV{args.limited_vehicle_number}.pkl")
        else:
            dataset_path = os.path.join(args.dir, env.problem, f"{env.problem.lower()}{args.problem_size}_uniform.pkl")

        env.generate_dataset(args.num_samples, args.problem_size, dataset_path)
        # sanity check
        data = env.load_dataset(dataset_path, num_samples=args.num_samples, disable_print=False)
        for i in len(data):
            print(data[i][0][:20])
