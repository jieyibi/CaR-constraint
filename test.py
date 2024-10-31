import os, random, math, time
import pytz
import argparse
import pprint as pp
from datetime import datetime
import pandas as pd
from Tester import Tester
from utils import *


def args2dict(args):
    env_params = {"problem_size": args.problem_size, "pomo_size": args.pomo_size, "tw_type": args.tw_type,"dl_percent": args.dl_percent,
                  "tw_duration": args.tw_duration, "pomo_start":args.pomo_start, "limited_vehicle_number": args.limited_vehicle_number,
                  "pomo_feasible_start": args.pomo_feasible_start, "reverse": args.reverse, "random_delta_t": args.random_delta_t,}
    model_params = {"embedding_dim": args.embedding_dim, "sqrt_embedding_dim": args.sqrt_embedding_dim,
                    "encoder_layer_num": args.encoder_layer_num, "decoder_layer_num": args.decoder_layer_num,
                    "qkv_dim": args.qkv_dim, "head_num": args.head_num, "logit_clipping": args.logit_clipping,
                    "ff_hidden_dim": args.ff_hidden_dim,"eval_type": args.eval_type, "norm": args.norm,
                    "norm_loc": args.norm_loc, "problem":args.problem, "load_safety_layer":args.load_safety_layer,
                    "learn_safety_layer": args.load_safety_layer, "sl_pe_dropout": args.sl_pe_dropout,
                    "sl_pe_type": args.sl_pe_type, "partial_route_aggregation": args.partial_route_aggregation,
                    "trustworthy_start_route_length": args.trustworthy_start_route_length, "add_candidate_feature":args.add_candidate_feature,
                    "sl_encoder_layer_num": args.sl_encoder_layer_num, "topk_masked":args.topk_masked,
                    "tw_normalize": args.tw_normalize, "simulation_fsb": args.simulation_fsb,
                    "dual_decoder": args.dual_decoder, "fsb_decoder_weight": args.fsb_decoder_weight,
                    "inner_safety_layer": args.inner_safety_layer, "unvisited_graph": args.unvisited_graph,
                    "visited_one_hot": args.visited_one_hot,"double_head_decoder": args.double_head_decoder,"decision_boundary":args.decision_boundary,
                    "W_q_sl":args.W_q_sl, "W_out_sl": args.W_out_sl, "W_kv_sl": args.W_kv_sl, "detach_from_encoder": args.detach_from_encoder,
                    "use_ninf_mask_in_sl_MHA": args.use_ninf_mask_in_sl_MHA, "seperated_decoder_num":args.seperated_decoder_num}

    tester_params = {"checkpoint": args.checkpoint, "test_episodes": args.test_episodes, "test_batch_size": args.test_batch_size,
                     "sample_size": args.sample_size, "aug_factor": args.aug_factor, "aug_batch_size": args.aug_batch_size,
                     "test_set_path": args.test_set_path, "test_set_opt_sol_path": args.test_set_opt_sol_path,
                     "fsb_dist_only": args.fsb_dist_only, "use_sl_mask": args.use_sl_mask, "lazy_piggy_model": args.lazy_piggy_model}

    return env_params, model_params, tester_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Towards Unified Models for Routing Problems")
    # env_params
    parser.add_argument('--problem', type=str, default="TSPTW", choices=["Train_ALL", "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
                                                                             "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
                                                                             "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"])
    parser.add_argument('--limited_vehicle_number', type=str, default="adaptive", choices=["adaptive", "plus1"])
    parser.add_argument('--tw_type', type=str, default="zhang",choices=["da_silva", "cappart", "zhang"])
    parser.add_argument('--tw_duration', type=str, default="1020", choices=["1020", "75100", "2550", "5075", "random"])
    parser.add_argument('--dl_percent', type=int, default=75, help="percentage of nodes that DL < total demand")
    parser.add_argument('--problem_size', type=int, default=50)
    parser.add_argument('--pomo_size', type=int, default=1, help="the number of start node, should <= problem size")
    parser.add_argument('--pomo_start', type=bool, default=False)
    parser.add_argument('--pomo_feasible_start', type= bool, default=False)
    parser.add_argument('--reverse', type=bool, default=False)
    parser.add_argument('--random_delta_t', type=float, default=0)
    # model_params
    parser.add_argument('--model_type', type=str, default="Single", choices=["Single", "MTL", "MOE"])
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--sqrt_embedding_dim', type=float, default=128**(1/2))
    parser.add_argument('--encoder_layer_num', type=int, default=6, help="the number of MHA in encoder")
    parser.add_argument('--decoder_layer_num', type=int, default=1, help="the number of MHA in decoder")
    parser.add_argument('--qkv_dim', type=int, default=16)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--logit_clipping', type=float, default=10)
    parser.add_argument('--ff_hidden_dim', type=int, default=512)
    parser.add_argument('--eval_type', type=str, default="argmax", choices=["argmax", "softmax"])
    parser.add_argument('--norm', type=str, default="instance", choices=["batch", "batch_no_track", "instance", "layer", "rezero", "none"])
    parser.add_argument('--norm_loc', type=str, default="norm_last", choices=["norm_first", "norm_last"],
                        help="whether conduct normalization before MHA/FFN/MOE")
    parser.add_argument('--load_safety_layer', type=bool, default=False)
    parser.add_argument('--sl_pe_type', type=str, default="cos_sin", choices=["con_sin", "sin"])
    parser.add_argument('--sl_pe_dropout', type=float, default=0.0)
    parser.add_argument('--partial_route_aggregation', type=str, default="mean", choices=["mean", "sum", "last", "max", "first"])
    parser.add_argument('--trustworthy_start_route_length',type=int, default=1)
    parser.add_argument('--sl_encoder_layer_num', type=int, default=1)
    parser.add_argument('--add_candidate_feature', type=bool, default=True)
    parser.add_argument('--tw_normalize', type=bool, default=True)
    parser.add_argument('--dual_decoder', type=bool, default=False)
    parser.add_argument('--double_head_decoder', type=bool, default=False)
    parser.add_argument('--seperated_decoder_num', type=int, default=1)
    parser.add_argument('--W_q_sl', type=bool, default=False)
    parser.add_argument('--W_out_sl', type=bool, default=False)
    parser.add_argument('--W_kv_sl', type=bool, default=False)
    parser.add_argument('--lazy_piggy_model', type=bool, default=False)
    parser.add_argument('--load_which_piggy', type=str, default="train_fsb_bsf", choices=["last_epoch", "train_fsb_bsf", "train_infsb_bsf", "train_accuracy_bsf"])
    parser.add_argument('--detach_from_encoder', type=bool, default=False)
    parser.add_argument('--use_ninf_mask_in_sl_MHA', type=bool, default= False)
    parser.add_argument("--fsb_decoder_weight", type=float, default=0.5)
    parser.add_argument('--topk_masked', type=int, default=3)

    parser.add_argument('--simulation_fsb', type=bool, default=False)
    parser.add_argument('--use_sl_mask', type=bool, default=False)
    parser.add_argument('--decision_boundary', type=float, default=0.5)
    parser.add_argument("--inner_safety_layer", type=bool, default=False)
    parser.add_argument("--unvisited_graph", type=bool, default=False)
    parser.add_argument("--visited_one_hot", type=bool, default=False)

    # tester_params
    # parser.add_argument("--checkpoint", type=str, default="")
    # parser.add_argument("--checkpoint", type=str, default="")
    # parser.add_argument("--checkpoint", type=str, default="")
    # parser.add_argument("--checkpoint", type=str, default="")
    # parser.add_argument("--checkpoint", type=str, default="")
    # parser.add_argument("--checkpoint", type=str, default="")
    # parser.add_argument("--checkpoint", type=str, default="")
    # parser.add_argument("--checkpoint", type=str, default="")
    # parser.add_argument("--checkpoint", type=str, default="")
    # parser.add_argument("--checkpoint", type=str, default="")
    # parser.add_argument("--checkpoint", type=str, default="")
    # parser.add_argument("--checkpoint", type=str, default="")


    #TSPTW
    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240406_224235_TSPTW50_da_silva_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240406_223935_TSPTW50_da_silva_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_simulation/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240425_182109_TSPTW50_da_silva_TimeoutReward_noPOMOStartSample_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_200_1000_50_fsb_m0/epoch-10000.pt")

    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240326_142328_TSPTW50_addInfsbCalulation_zhang_addTimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_1020/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240326_142844_TSPTW50_addInfsbCalulation_zhang_addTimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_1020_simulation/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240423_220308_TSPTW50_1020_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_200_1000_50_fsb_ignorem0_reverse/epoch-10000.pt")

    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240403_225040_TSPTW50_addInfsbCalulation_zhang_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240403_225148_TSPTW50_addInfsbCalulation_zhang_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_simulation/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240425_170437_TSPTW50_5075_TimeoutReward_noPOMOStartSample_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_200_1000_50_fsb_m0/epoch-10000.pt")

    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240409_133808_TSPTW100_da_silva_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240409_134013_TSPTW100_da_silva_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_simulation/epoch-3500.pt")
    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240423_211641_TSPTW100_da_silva_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_100_1000_20_last50_fsb_m0/epoch-10000.pt")

    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240328_151551_TSPTW100_zhang_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_1020/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240328_151359_TSPTW100_zhang_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_1020_simulation/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240423_211423_TSPTW100_1020_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_100_1000_20_last50_fsb_m0/epoch-10000.pt")

    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240406_170430_TSPTW100_zhang_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240320_132255_TSPTW100_addInfsbCalulation_random_addTimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty_twNormalized_simulation/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="/home/jieyi/Routing-Anything-main/pretrained/20240423_210915_TSPTW100_5075_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_100_1000_20_last50_fsb_m0/epoch-10000.pt")

    # TSPDL
    # parser.add_argument("--checkpoint", type=str, default="pretrained/20240430_172651_TSPDL50q90_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="pretrained/20240430_172549_TSPDL50q90_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_simulation/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="pretrained/20240508_124219_TSPDL50q90_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_200_1000_50_fsb_m0/epoch-10000.pt")

    # parser.add_argument("--checkpoint", type=str, default="pretrained/20240430_170010_TSPDL100q90_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_100_1000_20_50_fsb_ignorem0_reverse/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="pretrained/20240430_164234_TSPDL100q90_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="")

    # parser.add_argument("--checkpoint", type=str, default="pretrained/20240403_231158_TSPDL50q75_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="pretrained/20240403_231930_TSPDL50q75_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_simulation/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="pretrained/20240407_231459_TSPDL50q75_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_200_1000_50_fsb/epoch-10000.pt")

    # parser.add_argument("--checkpoint", type=str, default="pretrained/20240404_144954_TSPDL100q75_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty/epoch-10000.pt")
    # parser.add_argument("--checkpoint", type=str, default="pretrained/20240403_234213_TSPDL100q75_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty_simulation/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="pretrained/20240423_211958_TSPDL100q75_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_100_1000_20_last50_fsb_m0/epoch-10000.pt")

    parser.add_argument('--lazy_checkpoint', type=str, default=None)
    # parser.add_argument('--lazy_checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/results/20240423_211958_TSPDL100q75_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_100_1000_20_last50_fsb_m0/fsb_accuracy_bsf.pt")
    # parser.add_argument('--checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/pretrained/20240422_224942_TSPTW50_da_silva_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_200_1000_50_fsb/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/pretrained/20240425_182109_TSPTW50_da_silva_TimeoutReward_noPOMOStartSample_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_200_1000_50_fsb_m0/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/pretrained/20240406_223935_TSPTW50_da_silva_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_simulation/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/pretrained/20240406_224235_TSPTW50_da_silva_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/pretrained/20240326_142328_TSPTW50_addInfsbCalulation_zhang_addTimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_1020/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/pretrained/20240326_142844_TSPTW50_addInfsbCalulation_zhang_addTimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_1020_simulation/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/pretrained/20240423_220308_TSPTW50_1020_TimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_200_1000_50_fsb_ignorem0_reverse/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/pretrained/20240422_165055_TSPTW50_1020_zhang_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_200_1000_50_fsb_m0/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/pretrained/20240425_170437_TSPTW50_5075_TimeoutReward_noPOMOStartSample_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_200_1000_50_fsb_m0/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/pretrained/20240422_225442_TSPTW50_5075_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_Wq+Wout+Wkv_noNinfMask_noDetach_LabelBalanceLoss_200_1000_50_fsb/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/pretrained/20240403_225148_TSPTW50_addInfsbCalulation_zhang_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_simulation/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/pretrained/20240403_225040_TSPTW50_addInfsbCalulation_zhang_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/results/20240330_132317_TSPTW50_zhang_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_1020_Wq+Wout+Wkv_noNinfMasknoDetachLabelBalanceLossLoadepoch1000/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/results/20240330_190207_TSPTW50_zhang_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_1020_Wq+Wout+Wkv_noNinfMasknoDetachnoLabelBalanceLossLoadepoch1000/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="/mnt/slurm_home/jybi/Routing-Anything-main/results/20240330_190602_TSPTW50_zhang_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_1020_Wq+Wout+Wkv_noNinfMasknoDetachnoLabelBalanceLossLoadepoch1500fsbBSF/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="results/20240326_142844_TSPTW50_addInfsbCalulation_zhang_addTimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_1020_simulation/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str, default="results/20240328_235834_TSPTW50_zhang_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_1020_Wq+Wout+Wkv_noNinfMask_noDetach_noLabelBalanceLoss_200_1000_50_fsbBSF/epoch-10000.pt")

    # parser.add_argument('--checkpoint', type=str, default="results/20240329_152746_TSPTW50_zhang_TimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_1020_simulationASfinetuneEpoch100_lr1e-5/epoch-100.pt")
    # parser.add_argument('--checkpoint', type=str,
    #                     default="results/20240323_145233_TSPTW50_addInfsbCalulation_random_addTimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_twNormalized_simulation/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str,
    #                     default="results/20240326_142328_TSPTW50_addInfsbCalulation_zhang_addTimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_RealtwNormalized_1020/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str,
    #                     default="/home/jieyi/Routing-Anything-main/results/20240104_223958_addInfsbCalulation_zhang_addTimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str,
    #                     default="/home/jieyi/Routing-Anything-main/results/20240306_204955_addInfsbCalulation_zhang_addTimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_tighterTW0p25_0p5/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str,
    #                     default="/home/jieyi/Routing-Anything-main/results/20240306_204955_addInfsbCalulation_zhang_addTimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_tighterTW0p25_0p5/trained_model_best.pt")
    # parser.add_argument('--checkpoint', type=str,
    #                     default="/home/jieyi/Routing-Anything-main/results/20240307_124249_addInfsbCalulation_zhang_addTimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_tighterTW0p10_0p20/epoch-10000.pt")
    # parser.add_argument('--checkpoint', type=str,
    #                     default="results/20240307_213657_addInfsbCalulation_zhang_addTimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty_twNormalized/epoch-10000.pt")


    # parser.add_argument('--checkpoint', type=str,
    #                     default="results/20240104_223958_addInfsbCalulation_zhang_addTimeoutReward_noPOMOStartSample50_fsbDistOnly_noInfsbDistPenalty/epoch-10000.pt",
    #                     help="load pretrained model to evaluate")
    # parser.add_argument('--checkpoint', type=str,
    #                     default="results/20231229_141742_add_infsb_calulation_zhang_add_timeout_reward/epoch-10000.pt",
    #                     help="load pretrained model to evaluate")
    # parser.add_argument('--checkpoint', type=str,
    #                     default="results/20240228_104401_addInfsbCalulation_zhang_addTimeoutReward_noPOMOStartSample100_fsbDistOnly_noInfsbDistPenalty/trained_model_best.pt")
    # parser.add_argument('--checkpoint', type=str, default="results/20240115_202030_addTimeoutReward_fsbDistOnly_noPOMOStartSample20_finetuneOnPretrainedPOMO_everyStep_maxTop3mask_lr5e-6_addTWend_labelInbalancedWeight/epoch-71.pt" )
    # parser.add_argument('--sl_checkpoint', type=str, default="results/20240115_202030_addTimeoutReward_fsbDistOnly_noPOMOStartSample20_finetuneOnPretrainedPOMO_everyStep_maxTop3mask_lr5e-6_addTWend_labelInbalancedWeight/safety_layer_epoch-71.pt")
    # parser.add_argument('--checkpoint', type=str, default="results/20240302_223250_addTimeoutReward_fsbDistOnly_noPOMOStartSample50_finetuneOnPretrainedPOMO_everyStep_maxTop3mask_lr5e-6_addTWend_labelInbalancedWeight_last/epoch-33.pt" )
    # parser.add_argument('--sl_checkpoint', type=str, default="results/20240302_223250_addTimeoutReward_fsbDistOnly_noPOMOStartSample50_finetuneOnPretrainedPOMO_everyStep_maxTop3mask_lr5e-6_addTWend_labelInbalancedWeight_last/safety_layer_epoch-33.pt")
    # parser.add_argument('--checkpoint', type=str, default="results/20240305_104850_addTimeoutReward_fsbDistOnly_noPOMOStartSample20_finetuneOnPretrainedPOMO_everyStep_maxTop3mask_lr5e-6_addTWend_labelInbalancedWeight/epoch-5.pt" )
    # parser.add_argument('--sl_checkpoint', type=str, default="results/20240305_104850_addTimeoutReward_fsbDistOnly_noPOMOStartSample20_finetuneOnPretrainedPOMO_everyStep_maxTop3mask_lr5e-6_addTWend_labelInbalancedWeight/safety_layer_epoch-5.pt")

    # parser.add_argument('--sl_checkpoint', type=str, default="results/20240112_024410_addTimeoutReward_noPOMOStartSample50_fsbDistOnly_finetuneOnPretrainedPOMO_slLoss_everyStep_maxTop3mask_lr1e-6_labelInbalancedWeight/epoch-4.pt", help="load pretrained sl to assist")
    parser.add_argument('--test_episodes', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--sample_size', type=int, default=1, help="only activate if eval_type is softmax")
    parser.add_argument('--aug_factor', type=int, default=8, choices=[1, 8], help="whether to use instance augmentation during evaluation")
    parser.add_argument('--aug_batch_size', type=int, default=1)
    # parser.add_argument('--test_set_path', type=str, default="data/TSPDL/tspdl50_q90_uniform_fsb.pkl", help="evaluate on default test dataset if None")
    # parser.add_argument('--test_set_opt_sol_path', type=str, default="data/TSPDL/lkh_tspdl50_q90_uniform_fsb.pkl", help="evaluate on default test dataset if None")
    # parser.add_argument('--test_set_path', type=str, default="data/CVRP/cvrp50_uniform_adaptive_2025.pkl", help="evaluate on default test dataset if None")
    # parser.add_argument('--test_set_opt_sol_path', type=str, default="data/CVRP/lkh_cvrp50_uniform_adaptive_2025.pkl", help="evaluate on default test dataset if None")
    # parser.add_argument('--test_set_path', type=str, default="data/TSPTW/tsptw50_zhang_uniform.pkl", help="evaluate on default test dataset if None")
    # parser.add_argument('--test_set_opt_sol_path', type=str, default="data/TSPTW/lkh_tsptw50_zhang_uniform.pkl", help="evaluate on default test dataset if None")
    # parser.add_argument('--test_set_path', type=str, default="data/TSPDL/TSPDL_benchmark_data/Todosejevic_nonorm", help="evaluate on default test dataset if None")
    # parser.add_argument('--test_set_opt_sol_path', type=str, default=None, help="evaluate on default test dataset if None")
    parser.add_argument('--test_set_path', type=str, default="/home/jieyi/Routing-Anything-main/data/TSPTW/TSPTW_benchmark_data/Dumas", help="evaluate on default test dataset if None")
    parser.add_argument('--test_set_opt_sol_path', type=str, default=None, help="evaluate on default test dataset if None")
    parser.add_argument('--fsb_dist_only', type=bool, default=True)

    # settings (e.g., GPU)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--occ_gpu', type=float, default=0., help="occumpy (X)% GPU memory in advance, please use sparingly.")

    args = parser.parse_args()
    pp.pprint(vars(args))
    env_params, model_params, tester_params = args2dict(args)
    seed_everything(args.seed)

    if args.aug_factor != 1:
        args.test_batch_size = args.aug_batch_size
        tester_params['test_batch_size'] = tester_params['aug_batch_size']

    # set log & gpu
    # torch.set_printoptions(threshold=1000000)
    # process_start_time = datetime.now(pytz.timezone("Asia/Singapore"))
    # args.log_path = os.path.join(args.log_dir, "Test", process_start_time.strftime("%Y%m%d_%H%M%S"))
    # if not os.path.exists(args.log_path):
    #     os.makedirs(args.log_path)
    if not args.no_cuda and torch.cuda.is_available():
        occumpy_mem(args) if args.occ_gpu != 0. else print(">> No occupation needed")
        args.device = torch.device('cuda', args.gpu_id)
        torch.cuda.set_device(args.gpu_id)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        args.device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
    print(">> USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(not args.no_cuda, args.gpu_id))

    # start training
    print(">> Start {} Testing ...".format(args.problem))
    tester = Tester(args=args, env_params=env_params, model_params=model_params, tester_params=tester_params)
    tester.run()
    print(">> Finish {} Testing ...".format(args.problem))


    # score, aug_score, sol_infeasible_rate, ins_infeasible_rate
    #
    # for file in os.listdir(args.test_set_path):
    #     instance_name = file.split(".pkl")[0]
    #     print(">> Start {} Testing {} ...".format(args.problem, instance_name))
    #     tester = Tester(args=args, env_params=env_params, model_params=model_params, tester_params=tester_params)
    #     scores, aug_scores, sol_infeasible_rate, ins_infeasible_rate = tester.run()
    #     print(">> Finish {} Testing ...".format(args.problem))
    #
    # df = pd.DataFrame(tensor_data.cpu().numpy())
    #
    # # 将DataFrame写入Excel文件
    # excel_file = "cvrp50_2025.xlsx"
    # df.to_excel(excel_file, index=False, header=False)