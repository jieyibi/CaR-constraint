import pytz
import argparse
import pprint as pp
from datetime import datetime
import wandb
from Trainer import Trainer
from utils import *
import torch.distributed as dist
import torch.multiprocessing as mp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1233'
    dist.init_process_group(rank=rank, world_size=world_size, backend="nccl")
    torch.cuda.set_device(rank)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def cleanup():
    dist.destroy_process_group()

def args2dict(args):
    env_params = {"problem_size": args.problem_size, "pomo_size": args.pomo_size,
                  "tw_type": args.tw_type, "tw_duration": args.tw_duration,
                  "dl_percent": args.dl_percent, "random_delta_t": args.random_delta_t,
                  "precedence_ratio": args.precedence_ratio, "geometric_conflict_ratio": args.geometric_conflict_ratio,
                  "precedence_balance_ratio": args.precedence_balance_ratio,
                  "val_dataset": args.val_dataset, "val_episodes": args.val_episodes,
                  "pomo_start": args.pomo_start, "pomo_feasible_start": args.pomo_feasible_start,
                  "k_max": args.k_max,
                  # reward shaping
                  "with_regular": args.with_regular, "with_bonus": args.with_bonus,
                  }

    tester_params = {"eval_only": args.eval_only, "test_episodes": args.test_episodes,
                     "test_batch_size": args.test_batch_size, "test_dataset": args.test_dataset,
                     "test_z_sample_size": args.test_z_sample_size, "test_pomo_size": args.test_pomo_size,
                     "sample_size": args.sample_size, "aux_mask": args.aux_mask, "is_lib": args.is_lib,
                     "refinement_history_path": args.refinement_history_path, "best_solution_path": args.best_solution_path,
                     'EAS_params': {
                         'enable': args.enable_eas,
                         'iterations': args.iterations,
                         'lr': 0.0001,
                         'lambda': 0.05,
                         'resample': True,
                         "iterations_impr": args.iterations_impr
                     },
                    }

    model_params = {"embedding_dim": args.embedding_dim, "sqrt_embedding_dim": args.sqrt_embedding_dim,
                    "encoder_layer_num": args.encoder_layer_num, "decoder_layer_num": args.decoder_layer_num,
                    "qkv_dim": args.qkv_dim, "head_num": args.head_num, "logit_clipping": args.logit_clipping,
                    "ff_hidden_dim": args.ff_hidden_dim, "eval_type": args.eval_type,
                    "norm": args.norm, "norm_loc": args.norm_loc, "problem": args.problem,
                    "dual_decoder": args.dual_decoder, "clean_cache": args.clean_cache,
                    "gumbel": args.gumbel,
                    # improvement
                    "improvement_only": args.improvement_only, "problem_size": args.problem_size,
                    "improvement_method": args.improvement_method, "rm_num": args.rm_num, "boundary": args.boundary,
                    "improve_steps": args.improve_steps, "aspect_num": args.aspect_num,
                    "with_infsb_feature": args.with_infsb_feature, "supplement_feature_dim": args.supplement_feature_dim,
                    "with_RNN": args.with_RNN, "with_explore_stat_feature":args.with_explore_stat_feature,
                    "k_max": args.k_max, "impr_encoder_start_idx": args.impr_encoder_start_idx,
                    "select_top_k": args.select_top_k, "unified_decoder": args.unified_decoder,
                    "unified_encoder": args.unified_encoder, "n2s_decoder": args.n2s_decoder, "v_range": args.v_range,
                    # PIP-D
                    "pip_decoder": args.pip_decoder, "generate_PI_mask": args.generate_PI_mask,
                    # TSPTW
                    "tw_normalize": args.tw_normalize,
                    # SOP
                    "pairwise_merge": args.pairwise_merge, "which_feature": args.which_feature, "succ_attention_bias": args.succ_attention_bias,
                    }

    optimizer_params = {"optimizer": {"lr": args.lr, "weight_decay": args.weight_decay},
                        "scheduler": {"milestones": args.milestones, "gamma": args.gamma}}

    trainer_params = {"epochs": args.epochs, "train_episodes": args.train_episodes, "accumulation_steps": args.accumulation_steps,
                      "train_batch_size": args.train_batch_size, "validation_interval": args.validation_interval,
                      "validation_batch_size": args.validation_batch_size, "val_pomo_size": args.val_pomo_size,
                      "model_save_interval": args.model_save_interval, "checkpoint": args.checkpoint, "baseline": args.baseline,
                      "load_optimizer": args.load_optimizer, "uncertainty_weight": args.uncertainty_weight,
                      "dynamic_coefficient": args.dynamic_coefficient, "coefficient": args.coefficient,
                      # constraints
                      "wo_node_penalty": args.wo_node_penalty, "wo_tour_penalty": args.wo_tour_penalty,
                      "soft_constrained": args.soft_constrained, "backhaul_mask": args.backhaul_mask,
                      "non_linear": args.non_linear, "non_linear_cons": args.non_linear_cons, "epsilon": args.epsilon,
                      "epsilon_base": args.epsilon_base, "epsilon_decay_beta": args.epsilon_decay_beta,
                      "out_reward": args.out_reward, "out_node_reward":args.out_node_reward,
                      "penalty_normalize": args.penalty_normalize,
                      "fsb_dist_only": args.fsb_dist_only, "fsb_reward_only":args.fsb_reward_only,
                      "infsb_dist_penalty": args.infsb_dist_penalty, "penalty_factor": args.penalty_factor,
                      "reward_gating": args.reward_gating, "constraint_number": args.constraint_number,
                      "subgradient": args.subgradient, "subgradient_lr": args.subgradient_lr,
                      # PIP
                      "generate_PI_mask": args.generate_PI_mask, "pip_step": args.pip_step,
                      "pip_decoder": args.pip_decoder, "use_real_PI_mask": args.use_real_PI_mask,
                      "lazy_pip_model": args.lazy_pip_model, "simulation_stop_epoch": args.simulation_stop_epoch,
                      "pip_update_interval": args.pip_update_interval, "pip_last_growup": args.pip_last_growup,
                      "pip_update_epoch": args.pip_update_epoch, "load_which_pip": args.load_which_pip,
                      "pip_checkpoint": args.pip_checkpoint,
                      # reward & loss
                      "bonus_for_construction": args.bonus_for_construction, "extra_bonus": args.extra_bonus, "extra_weight": args.extra_weight,
                      "diversity_loss": args.diversity_loss, "diversity_weight": args.diversity_weight, "probs_return": args.probs_return,
                      "imitation_learning": args.imitation_learning, "imitation_loss_weight": args.imitation_loss_weight,
                      # improvement
                      "improvement_only": args.improvement_only, "init_sol_strategy": args.init_sol_strategy,
                      "max_dummy_size": args.max_dummy_size, "improve_start_when_dummy_ok": args.improve_start_when_dummy_ok,
                      "val_init_sol_strategy": args.val_init_sol_strategy, "select_top_k_val": args.select_top_k_val,
                      "neighborhood_search": args.neighborhood_search, "k_unconfident": args.k_unconfident,
                      "improvement_method": args.improvement_method, "rm_num": args.rm_num, "insert_before": args.insert_before,
                      "improve_steps": args.improve_steps, "dummy_improve_steps": args.dummy_improve_steps,
                      "total_history": args.total_history, "dummy_improve_selected": args.dummy_improve_selected,
                      "stochastic_probability": args.stochastic_probability, "select_strategy": args.select_strategy,
                      "select_top_k": args.select_top_k, "diversity": args.diversity,
                      "validation_improve_steps": args.validation_improve_steps, "val_reconstruct_times": args.val_reconstruct_times,
                      "diversity_reward": args.diversity_reward, "repeats": args.repeats,
                      "seperate_obj_penalty": args.seperate_obj_penalty,
                      "reconstruct": args.reconstruct, "reconstruct_improve_bonus": args.reconstruct_improve_bonus,
                      "amp_training": args.amp_training,
                      }

    return env_params, model_params, optimizer_params, trainer_params, tester_params

def set_problem_defaults(args):
    problem = args.problem
    
    if problem == "CVRP":
        args.pomo_start = True
        args.supplement_feature_dim = 5
        args.improvement_method = "kopt"
        args.n2s_decoder = False
        args.test_batch_size = 1000
        args.test_episodes = 1000
    elif problem == "VRPBLTW":
        args.pomo_start = False
        args.supplement_feature_dim = 17
        args.improvement_method = "rm_n_insert"
        args.n2s_decoder = True
        args.aux_mask = True
        args.test_batch_size = 1000
        args.test_episodes = 1000
    elif problem == "TSPDL" or problem == "TSPTW":
        args.pomo_start = False
        args.supplement_feature_dim = 5
        args.improvement_method = "kopt"
        args.n2s_decoder = False
        args.test_batch_size = 3334
        args.test_episodes = 10000
        args.generate_PI_mask = True
        args.use_real_PI_mask = True
    elif problem == "SOP":
        args.pomo_start = False
        args.supplement_feature_dim = 5
        args.improvement_method = "rm_n_insert"
        args.n2s_decoder = True
        args.test_batch_size = 3334
        args.test_episodes = 10000


def main(rank, world_size, args, env_params, model_params, optimizer_params, trainer_params, tester_params):
    if args.wandb_logger and rank == 0:
        create_logger(filename="run_log", log_path=args.log_path)
        wandb.init(project="unified_solver", name=args.name, config={**env_params, **model_params, **optimizer_params, **trainer_params, **tester_params})
    seed_everything(args.seed)
    if args.multi_processing: setup(rank, world_size)
    trainer = Trainer(args, env_params, model_params, optimizer_params, trainer_params, tester_params, rank)
    if rank==0 and not args.eval_only: copy_all_src(args.log_path)
    if args.eval_only:
        with torch.inference_mode():
            with torch.amp.autocast("cuda"):
                trainer.test()
    else:
        trainer.train()
    if args.multi_processing: cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Towards Unified Models for Routing Problems")
    # env_params
    parser.add_argument('--problem', type=str, default="VRPBLTW", choices=["CVRP", "TSPTW", "TSPDL", "VRPBLTW", "SOP"])
    parser.add_argument('--tw_type', type=str, default="da_silva", choices=["da_silva", "cappart", "zhang", "random"])
    parser.add_argument('--tw_duration', type=str, default="1020", choices=["1020", "75100", "2550", "5075", "random", "curriculum"])
    parser.add_argument('--dl_percent', type=int, default=90, help="percentage of nodes that DL < total demand")
    parser.add_argument('--random_delta_t', type=float, default=0)
    # SOP-specific parameters
    parser.add_argument('--precedence_ratio', type=float, default=0.2, help="SOP: precedence ratio")
    parser.add_argument('--geometric_conflict_ratio', type=float, default=0.8, help="SOP: geometric conflict ratio")
    parser.add_argument('--precedence_balance_ratio', type=float, default=0.0, help="SOP: precedence balance ratio")
    parser.add_argument('--problem_size', type=int, default=100)
    parser.add_argument('--pomo_size', type=int, default=100, help="the number of start node, should <= problem size")
    parser.add_argument('--pomo_start', type=bool, default=True)
    parser.add_argument('--pomo_feasible_start', type= bool, default=False)
    parser.add_argument('--fsb_start_delay', type=int, default=10000)
    parser.add_argument('--val_dataset', type=str, nargs='+', default =None)
    parser.add_argument('--select_top_k_val', type=int, default=1)
    parser.add_argument('--select_top_k', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=1, help="number of repeats")
    parser.add_argument('--improve_steps', type=int, default=5)
    parser.add_argument('--validation_improve_steps', type=int, default= 20)
    parser.add_argument('--enable_eas', type=bool, default=False)
    parser.add_argument('--iterations', type=int, default=200, help='Number of iterations for EAS')
    parser.add_argument('--iterations_impr', type=int, default=20, help='Number of iterations for EAS')
    parser.add_argument('--best_solution_path', type=str, default=None)
    parser.add_argument('--refinement_history_path', type=str, default=None)

    # tester_params
    parser.add_argument('--eval_only', type=bool, default=True)
    parser.add_argument('--test_episodes', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument("--test_pomo_size", type=int, default=1)
    parser.add_argument('--test_dataset', type=str, default=None)
    parser.add_argument('--is_lib', type=bool, default=False)
    parser.add_argument('--test_z_sample_size', type=int, default=0)
    parser.add_argument('--eval_type', type=str, default="argmax", choices=["argmax", "softmax"])
    parser.add_argument('--sample_size', type=int, default = 1)
    parser.add_argument('--aux_mask', type=bool, default=False, help="only activates when problem == VRPBLTW")

    # load
    parser.add_argument('--gpu_id', type=str, default="3")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint file")
    parser.add_argument('--load_optimizer', type=bool, default=False)

    # model_params
    parser.add_argument('--model_type', type=str, default="SINGLE", choices=["SINGLE", "MTL", "MOE"])
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--sqrt_embedding_dim', type=float, default=128**(1/2))
    parser.add_argument('--encoder_layer_num', type=int, default=6, help="the number of MHA in encoder")
    parser.add_argument("--impr_encoder_start_idx", type=int, default=0)
    parser.add_argument('--decoder_layer_num', type=int, default=1, help="the number of MHA in decoder")
    parser.add_argument('--unified_encoder', type=bool, default=True)
    parser.add_argument('--unified_decoder', type=bool, default=False)
    parser.add_argument('--n2s_decoder', type=bool, default=False)
    parser.add_argument('--v_range', type=float, default=6.0, help='to control the entropy')
    parser.add_argument('--qkv_dim', type=int, default=16)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--logit_clipping', type=float, default=10)
    parser.add_argument('--ff_hidden_dim', type=int, default=512)
    parser.add_argument('--tw_normalize', type=bool, default=True)
    parser.add_argument('--norm', type=str, default="instance", choices=["batch", "batch_no_track", "instance", "layer", "rezero", "none"])
    parser.add_argument('--norm_loc', type=str, default="norm_last", choices=["norm_first", "norm_last"], help="whether conduct normalization before MHA/FFN")
    parser.add_argument('--dual_decoder', type=bool, default=False)
    parser.add_argument('--aspect_num', type=int, default=2, help="aspects of info, now includes features and positional info")
    parser.add_argument('--pairwise_merge', type=str, default="feature,attention", help='Comma-separated list of modes: feature,embedding,attention')
    parser.add_argument('--which_feature', type=str, default="both", choices=["succ", "prec", "both"], help="SOP: which feature to use")
    parser.add_argument('--succ_attention_bias', type=float, default=1.0, help="successor attention bias")
    # optimizer_params
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--milestones', type=int, nargs='+', default=[4501, ], help='when to decay lr')
    parser.add_argument('--gamma', type=float, default=0.1, help='new_lr = lr * gamma')
    parser.add_argument('--dynamic_coefficient', type=bool, default=False)
    parser.add_argument('--uncertainty_weight', type=bool, default=False)
    parser.add_argument("--amp_training", type=bool, default=True)

    # trainer_params
    parser.add_argument('--epochs', type=int, default=5000, help="total training epochs")
    parser.add_argument('--train_episodes', type=int, default=10000*2, help="the num. of training instances per epoch")
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=64*2)
    parser.add_argument('--validation_interval', type=int, default=500)
    parser.add_argument('--validation_batch_size', type=int, default=40)
    parser.add_argument('--val_episodes', type=int, default=1000)
    parser.add_argument("--val_pomo_size", type=int, default=1)
    parser.add_argument('--model_save_interval', type=int, default=50)

    # PIP
    parser.add_argument("--generate_PI_mask", type=bool, default=False)
    parser.add_argument('--use_real_PI_mask', type=bool, default=False, help="whether to use PI masking")
    parser.add_argument('--pip_step', type=int, default=1)
    parser.add_argument('--pip_decoder', action='store_true', default=False)
    parser.add_argument('--lazy_pip_model', type=bool, default=False)
    parser.add_argument('--simulation_stop_epoch', type=int, default=100) # use 100 when N=100
    parser.add_argument('--pip_update_interval', type=int, default=1000)
    parser.add_argument('--pip_update_epoch', type=int, default=20) # use 20 when N=100
    parser.add_argument('--pip_last_growup', type=int, default=50)
    parser.add_argument('--load_which_pip', type=str, default="train_fsb_bsf", choices=["last_epoch", "train_fsb_bsf", "train_infsb_bsf", "train_accuracy_bsf"])
    parser.add_argument('--pip_checkpoint', type=str, default=None)

    # constraints
    parser.add_argument('--soft_constrained', type=bool, default=False)
    parser.add_argument('--backhaul_mask', type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument('--non_linear', type=str, default=None, choices=[None, "fixed_epsilon", "decayed_epsilon", "step", "scalarization"])
    parser.add_argument('--epsilon', type=float, default=3.67)
    parser.add_argument('--epsilon_base', type=float, default=5.)
    parser.add_argument('--epsilon_decay_beta', type=float, default=0.001)
    parser.add_argument('--non_linear_cons', type=bool, default=False, help="enable non-linear reward function during construction")
    parser.add_argument('--out_reward', type=bool, default=True)
    parser.add_argument("--out_node_reward",type=bool,default=True)
    parser.add_argument("--penalty_normalize", type=bool, default=False)
    parser.add_argument('--fsb_dist_only', type=bool, default=True)
    parser.add_argument('--fsb_reward_only', type=bool, default=True) # activate only if no penalty
    parser.add_argument('--infsb_dist_penalty', type=bool, default=False)
    parser.add_argument('--penalty_factor', type=float, default=1.0)
    parser.add_argument('--reward_gating', type=bool, default=False)
    parser.add_argument('--subgradient', type=bool, default=False) # adaptive_primal_dual
    parser.add_argument('--subgradient_lr', type=float, default=0.1)
    parser.add_argument('--constraint_number', type = int, default=4)
    parser.add_argument('--gumbel', type=bool, default=False)
    parser.add_argument('--wo_node_penalty', type=bool, default=False, help="without node penalty")
    parser.add_argument('--wo_tour_penalty', type=bool, default=False, help="without tour penalty")
    # reward
    parser.add_argument('--baseline', type=str, choices=['group', "improve", "share"], default="group")# group reward: average rollout as baselines
    parser.add_argument('--bonus_for_construction', type=bool, default=False, help="reduce the advantage for negative samples and increase the advantage for positive samples (with good quality and can be improved)")
    parser.add_argument('--extra_bonus', type=bool, default=False,
                        help="add extra bonus for improving the solution (with good quality and can be improved)")
    parser.add_argument('--extra_weight', type=float, default=0.1)
    parser.add_argument('--diversity_loss', type=bool, default=True)
    parser.add_argument('--diversity_reward', type=bool, default=False)
    parser.add_argument('--diversity_weight', type=float, default=0.01)
    parser.add_argument('--probs_return', type=bool, default=False)
    parser.add_argument('--imitation_learning', type=bool, default=True)
    parser.add_argument('--imitation_loss_weight', type=float, default=1.)

    # improvement
    parser.add_argument('--improvement_only', type=bool, default=False)
    parser.add_argument('--improvement_method', type=str, default="kopt", choices=["rm_n_insert", "kopt", "all"])
    parser.add_argument('--boundary', type=float, default=0.5)
    parser.add_argument('--insert_before', type=bool, default=True)
    parser.add_argument('--rm_num', type=int, default=1)
    parser.add_argument('--coefficient', type=float, default=100)
    parser.add_argument('--reconstruct', type=bool, default=False)
    parser.add_argument('--reconstruct_improve_bonus', type=bool, default=False)
    parser.add_argument('--reconstruct_bonus_weight', type=float, default=1.)
    parser.add_argument('--neighborhood_search', type=bool, default=False)
    parser.add_argument('--k_unconfident', type=int, default=10)
    parser.add_argument('--init_sol_strategy', type=str, default="POMO", choices=["random", "greedy_feasible", "random_feasible", "POMO"])
    parser.add_argument('--val_init_sol_strategy', type=str, default="POMO", choices=["random", "greedy_feasible", "random_feasible", "POMO"])
    parser.add_argument('--POMO_checkpoint', type=str, default="results/20240831_221004_TSPTW50_rmPOMOstart_Soft_unifiedEnc_GroupBaseline_construction_only/epoch-5000.pt")
    parser.add_argument('--max_dummy_size', type=int, default=12)
    parser.add_argument('--improve_start_when_dummy_ok', type=bool, default=True)
    parser.add_argument('--dummy_improve_steps', type=int, default=0)
    parser.add_argument('--dummy_improve_selected', type=str, default="random", choices=["random", "topk"])

    parser.add_argument('--val_reconstruct_times', type=int, default=1)
    parser.add_argument('--select_strategy', type=str, default="quality", choices=["quality", "diversity", "quality_stochastic", "diversity_stochastic", "stochastic"])
    parser.add_argument('--stochastic_probability', type=float, default=0.5)
    parser.add_argument('--diversity', type=str, default="kendall_tau_distance", choices=["kendall_tau_distance", "jaccard_distance"])
    parser.add_argument('--total_history', type=int, default=3)
    parser.add_argument('--with_infsb_feature', type=bool, default=True)
    parser.add_argument('--supplement_feature_dim', type=int, default=5) # for cvrp:5; for tsptw:5; for vrpbltw: 17
    parser.add_argument('--with_explore_stat_feature', type=bool, default=True)
    parser.add_argument('--with_RNN', type=bool, default=True)
    parser.add_argument('--k_max', type=int, default=4)
    parser.add_argument('--with_regular', type=bool, default=False)
    parser.add_argument('--with_bonus', type=bool, default=False)
    parser.add_argument('--seperate_obj_penalty', type=bool, default=False)

    # settings (e.g., GPU)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--log_dir', type=str, default="./results")
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument("--multiple_gpu", type=bool, default=False)
    parser.add_argument('--occ_gpu', type=float, default=0., help="occupy (X)% GPU memory in advance, please use sparingly.")
    parser.add_argument('--tb_logger', type=bool, default=True)
    parser.add_argument('--wandb_logger', type=bool, default=True)
    parser.add_argument('--clean_cache', type=bool, default=False)
    parser.add_argument('--multi_processing', type=bool, default=False)

    args = parser.parse_args()
    set_problem_defaults(args)
    
    if args.eval_only:
        assert args.checkpoint is not None, "eval-only mode requires checkpoint!"
        args.load_optimizer = False
    if not args.eval_only: pp.pprint(vars(args))

    log_path = None
    note = "test"

    args.wandb_logger = False
    args.tb_logger = False
    args.train_batch_size //= args.world_size
    if args.n2s_decoder:
        args.insert_before = False # original n2s decoder inserts after the selected node

    env_params, model_params, optimizer_params, trainer_params, tester_params = args2dict(args)
    seed_everything(args.seed)

    # set log & gpu
    process_start_time = datetime.now(pytz.timezone("Asia/Singapore"))
    if log_path is not None:
        args.log_path = log_path
        args.name=log_path.split("/")[-1]
    else:
        name = process_start_time.strftime("%Y%m%d_%H%M%S")+note
        args.name = name
        args.log_path = os.path.join(args.log_dir, name)
    if not os.path.exists(args.log_path) and not args.eval_only:
        os.makedirs(args.log_path)

    if args.problem == "ALL" and args.model_type == "Single":
        assert False, "Cannot solve multiple problems with Single model, please use MOE instead."

    if not args.eval_only: print(">> Log Path: {}".format(args.log_path))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not args.no_cuda and torch.cuda.is_available():
        if args.multiple_gpu and torch.cuda.device_count() > 1:
            args.world_size = torch.cuda.device_count()
        else:
            occumpy_mem(args) if args.occ_gpu != 0. else print(">> No occupation needed")
        args.device = torch.device('cuda')
        torch.cuda.set_device(0)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        args.device = torch.device('cpu')

    print(">> USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(not args.no_cuda, args.gpu_id))

    torch.set_printoptions(threshold=1000000)

    # multi-processing
    if args.multi_processing:
        torch.multiprocessing.set_start_method('spawn')
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # start training
    if not args.eval_only: print(">> Start {} Training using {} Model ...".format(args.problem, args.model_type))
    if args.multi_processing:
        mp.spawn(main, args=(args.world_size, args, env_params, model_params, optimizer_params, trainer_params,tester_params), nprocs=args.world_size, join=True)
    else:
        main(0, args.world_size, args, env_params, model_params, optimizer_params, trainer_params,tester_params)
    if not args.eval_only: print(">> Finish {} Training using {} Model ...".format(args.problem, args.model_type))
