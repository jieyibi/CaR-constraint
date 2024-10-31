import os, random, math, time
import argparse
import pprint as pp
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets")
    parser.add_argument('--problem', type=str, default="TSPTW", choices=["ALL", "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
                                                                       "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
                                                                       "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"])
    parser.add_argument('--limited_vehicle_number', type=int, default=6, choices=["adaptive", "plus1"])
    parser.add_argument('--problem_size', type=int, default=50)
    parser.add_argument('--pomo_size', type=int, default=50, help="the number of start node, should <= problem size")
    parser.add_argument('--tw_type', type=str, default="da_silva",choices=["da_silva", "cappart", "zhang"])
    parser.add_argument('--tw_duration', type=str, default="test", choices=["1020", "75100", "2550", "5075", "random"])
    parser.add_argument('--dl_percent', type=int, default=60, help="percentage of nodes that DL < total demand for TSPDL")
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=2025)
    # parser.add_argument('--dir', type=str, default="/home/jieyi")
    parser.add_argument('--dir', type=str, default="./data")
    parser.add_argument('--no_cuda', action='store_true', default=True)
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()
    pp.pprint(vars(args))
    env_params = {"problem_size": args.problem_size, "pomo_size": args.pomo_size, "tw_type": args.tw_type,
                  "dl_percent": args.dl_percent, "tw_duration": args.tw_duration, "limited_vehicle_number": args.limited_vehicle_number}
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
            dataset_path = os.path.join(args.dir, env.problem, "{}{}_{}_uniform_varyN.pkl".format(env.problem.lower(), args.problem_size, args.tw_type))
        elif args.problem == "TSPDL":
            dataset_path = os.path.join(args.dir, env.problem, "{}{}_q{}_uniform_fsb.pkl".format(env.problem.lower(), args.problem_size, args.dl_percent))
        elif args.problem in ["CVRP", "VRPTW"] and args.limited_vehicle_number is not None:
            dataset_path = os.path.join(args.dir, env.problem,
                                        "{}{}_uniform_LV{}.pkl".format(env.problem.lower(), args.problem_size, args.limited_vehicle_number))
        else:
            dataset_path = os.path.join(args.dir, env.problem, "{}{}_uniform.pkl".format(env.problem.lower(), args.problem_size))

        env.generate_dataset(args.num_samples, args.problem_size, dataset_path)
        # sanity check
        data = env.load_dataset(dataset_path, num_samples=args.num_samples, disable_print=False)
        print(data[0][0][:20])
        print(data[1][0][:20])
        print(data[2][0][:20])
        print(data[3][0][:20])
        # print(data[4][0][:20])
        # print(data[0][0])
        # print(data[1][0])
        # print(data[2][0])
        # print(data[3][0])
