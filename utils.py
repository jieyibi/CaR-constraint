import os, sys
import time
import random
import math
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from scipy.stats import ttest_rel
import shutil
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset
import logging

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class TimeEstimator:
    def __init__(self):
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        print("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(count, total, elapsed_time_str, remain_time_str))

def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used


def occumpy_mem(args):
    """
        Occupy GPU memory in advance.
    """
    # torch.cuda.set_device(args.gpu_id)
    total, used = check_mem(args.gpu_id)
    total, used = int(total), int(used)
    block_mem = int((total-used) * args.occ_gpu)
    x = torch.cuda.FloatTensor(8, 128, block_mem)
    del x


def seed_everything(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_env(problem):
    from envs import TSPDLEnv, TSPTWEnv, CVRPEnv, OVRPEnv, VRPBEnv, VRPLEnv, VRPTWEnv, OVRPTWEnv, OVRPBEnv, OVRPLEnv, VRPBLEnv, VRPBTWEnv, VRPLTWEnv, OVRPBLEnv, OVRPBTWEnv, OVRPLTWEnv, VRPBLTWEnv, OVRPBLTWEnv
    training_problems = ['CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW']
    all_problems = {
        'TSPTW': TSPTWEnv.TSPTWEnv,
        'TSPDL': TSPDLEnv.TSPDLEnv,
        'CVRP': CVRPEnv.CVRPEnv,
        'OVRP': OVRPEnv.OVRPEnv,
        'VRPB': VRPBEnv.VRPBEnv,
        'VRPL': VRPLEnv.VRPLEnv,
        'VRPTW': VRPTWEnv.VRPTWEnv,
        'OVRPTW': OVRPTWEnv.OVRPTWEnv,
        'OVRPB': OVRPBEnv.OVRPBEnv,
        'OVRPL': OVRPLEnv.OVRPLEnv,
        'VRPBL': VRPBLEnv.VRPBLEnv,
        'VRPBTW': VRPBTWEnv.VRPBTWEnv,
        'VRPLTW': VRPLTWEnv.VRPLTWEnv,
        'OVRPBL': OVRPBLEnv.OVRPBLEnv,
        'OVRPBTW': OVRPBTWEnv.OVRPBTWEnv,
        'OVRPLTW': OVRPLTWEnv.OVRPLTWEnv,
        'VRPBLTW': VRPBLTWEnv.VRPBLTWEnv,
        'OVRPBLTW': OVRPBLTWEnv.OVRPBLTWEnv,
    }
    if problem == "Train_ALL":
        return [all_problems[i] for i in training_problems]
    elif problem == "ALL":
        return list(all_problems.values())
    else:
        return [all_problems[problem]]


def get_model(model_type):
    from models import MOEModel, SINGLEModel, MTLModel
    if model_type == "MTL":
        return MTLModel.MTLModel
    elif model_type == "MOE":
        return MOEModel.MOEModel
    else:
        return SINGLEModel.SINGLEModel


def get_opt_sol_path(dir, problem, size):
    all_opt_sol = {
        'CVRP': {50: 'hgs_cvrp50_uniform.pkl', 100: 'hgs_cvrp100_uniform.pkl'},
        'OVRP': {50: 'or_tools_200s_ovrp50_uniform.pkl', 100: 'lkh_ovrp100_uniform.pkl'},
        'VRPB': {50: 'or_tools_200s_vrpb50_uniform.pkl', 100: 'or_tools_400s_vrpb100_uniform.pkl'},
        'VRPL': {50: 'or_tools_200s_vrpl50_uniform.pkl', 100: 'lkh_vrpl100_uniform.pkl'},
        'VRPTW': {50: 'hgs_vrptw50_uniform.pkl', 100: 'hgs_vrptw100_uniform.pkl'},
        'OVRPTW': {50: 'or_tools_200s_ovrptw50_uniform.pkl', 100: 'or_tools_400s_ovrptw100_uniform.pkl'},
        'OVRPB': {50: 'or_tools_200s_ovrpb50_uniform.pkl', 100: 'or_tools_400s_ovrpb100_uniform.pkl'},
        'OVRPL': {50: 'or_tools_200s_ovrpl50_uniform.pkl', 100: 'or_tools_400s_ovrpl100_uniform.pkl'},
        'VRPBL': {50: 'or_tools_200s_vrpbl50_uniform.pkl', 100: 'or_tools_400s_vrpbl100_uniform.pkl'},
        'VRPBTW': {50: 'or_tools_200s_vrpbtw50_uniform.pkl', 100: 'or_tools_400s_vrpbtw100_uniform.pkl'},
        'VRPLTW': {50: 'or_tools_200s_vrpltw50_uniform.pkl', 100: 'or_tools_400s_vrpltw100_uniform.pkl'},
        'OVRPBL': {50: 'or_tools_200s_ovrpbl50_uniform.pkl', 100: 'or_tools_400s_ovrpbl100_uniform.pkl'},
        'OVRPBTW': {50: 'or_tools_200s_ovrpbtw50_uniform.pkl', 100: 'or_tools_400s_ovrpbtw100_uniform.pkl'},
        'OVRPLTW': {50: 'or_tools_200s_ovrpltw50_uniform.pkl', 100: 'or_tools_400s_ovrpltw100_uniform.pkl'},
        'VRPBLTW': {50: 'or_tools_200s_vrpbltw50_uniform.pkl', 100: 'or_tools_400s_vrpbltw100_uniform.pkl'},
        'OVRPBLTW': {50: 'or_tools_200s_ovrpbltw50_uniform.pkl', 100: 'or_tools_400s_ovrpbltw100_uniform.pkl'},
    }
    return os.path.join(dir, all_opt_sol[problem][size])


def num_param(model):
    nb_param = 0
    for param in model.parameters():
        nb_param += param.numel()
    print('Number of Parameters: {}'.format(nb_param))


def check_null_hypothesis(a, b):
    print(len(a), a)
    print(len(b), b)
    alpha_threshold = 0.05
    t, p = ttest_rel(a, b)  # Calc p value
    print(t, p)
    p_val = p / 2  # one-sided
    # assert t < 0, "T-statistic should be negative"
    print("p-value: {}".format(p_val))
    if p_val < alpha_threshold:
        print(">> Null hypothesis (two related or repeated samples have identical average values) is Rejected.")
    else:
        print(">> Null hypothesis (two related or repeated samples have identical average values) is Accepted.")

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename, disable_print=False):
    filedir = os.path.split(filename)[0]
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    if not disable_print:
        print(">> Save dataset to {}".format(filename))


def load_dataset(filename, disable_print=False):
    with open(check_extension(filename), 'rb') as f:
        data = pickle.load(f)
    if not disable_print:
        print(">> Load {} data ({}) from {}".format(len(data), type(data), filename))
    return data

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass  # This flush method is required for file-like object.


def create_logger(filename, log_path=None):
    if log_path and not os.path.exists(log_path):
        os.makedirs(log_path)

    file_mode = 'a' if os.path.isfile(os.path.join(log_path, filename)) else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    # Clear existing handlers
    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # Write to file
    fileout = logging.FileHandler(os.path.join(log_path,filename), mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # Write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # Redirect print to logging
    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
        Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True, disable_tqdm=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    os.makedirs(directory, exist_ok=True)
    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval, disable=disable_tqdm))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    # assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    if len(failed) != 0:
        "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def show(x, y, label, title, xdes, ydes, path, min_y=None, max_y=None, x_scale="linear", dpi=300):
    plt.style.use('fast')  # bmh, fivethirtyeight, Solarize_Light2
    plt.figure(figsize=(8, 8))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', 'lightpink', 'lightgreen', 'linen', 'slategray', 'darkviolet', 'darkcyan']

    assert len(x) == len(y)
    for i in range(len(x)):
        if i < len(label):
            # plt.scatter(x[i], y[i], color=colors[i], s=50)  # label=label[i]
            plt.plot(x[i], y[i], color=colors[i], label=label[i], linewidth=3)
        else:
            # plt.scatter(x[i], y[i], color=colors[i % len(label)])
            plt.plot(x[i], y[i], color=colors[i % len(label)], linewidth=3)

    plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    plt.xlabel(xdes, fontsize=24)
    plt.ylabel(ydes, fontsize=24)

    if min_y and max_y:
        plt.ylim((min_y, max_y))

    plt.title(title, fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='upper right', fontsize=16)
    plt.xscale(x_scale)
    # plt.margins(x=0)

    # plt.grid(True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close("all")


def loss_edges(y_pred_edges, y_edges, edge_cw, loss_type='CE',
               reduction='mean', gamma=2):
    """
    Loss function for edge predictions.

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
        edge_cw: Class weights for edges loss

    Returns:
        loss_edges: Value of loss function

    """
    # Edge loss
    if loss_type == 'CE':
        # y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        # y = y.permute(0, 3, 1, 2)  # B x voc_edges x V x V
        y_pred_edges = torch.log(y_pred_edges + 1e-8)
        loss_edges = nn.NLLLoss(edge_cw)(y_pred_edges, y_edges)
    elif loss_type == 'FL':
        # print(gamma)
        # y = y_pred_edges.permute(0, 3, 1, 2)  # B x voc_edges x V x V
        loss_edges = FocalLoss(weight=edge_cw, gamma=gamma, reduction=reduction)(y_pred_edges, y_edges)
    return loss_edges

class FocalLoss(nn.Module):
    """
    Focal Loss for edge predictions.
    """

    def __init__(self, weight=None,
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):

        prob = input_tensor
        log_prob = torch.log(prob + 1e-8)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


def copy_all_src(dst_root):

    # execution dir
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])

    # home dir setting
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    # make target directory
    dst_path = os.path.join(dst_root, 'src')

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for item in sys.modules.items():
        key, value = item

        if hasattr(value, '__file__') and value.__file__:
            src_abspath = os.path.abspath(value.__file__)

            if os.path.commonprefix([home_dir, src_abspath]) == home_dir:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

                if os.path.exists(dst_filepath):
                    split = list(os.path.splitext(dst_filepath))
                    split.insert(1, '({})')
                    filepath = ''.join(split)
                    post_index = 0

                    while os.path.exists(filepath.format(post_index)):
                        post_index += 1

                    dst_filepath = filepath.format(post_index)

                shutil.copy(src_abspath, dst_filepath)

def gather_tensor_and_concat(tensor):
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)

def get_solution_with_dummy_depot(solution, problem_size):
    # solution.size: (batch, pomo, solution)
    batch_size, pomo_size, _ = solution.size()
    dummy_size = solution.size(-1) - problem_size
    solution = solution.clone()
    solution[solution != 0] += (dummy_size - 1)
    solution[solution == 0] = torch.arange(0, dummy_size).repeat(batch_size, pomo_size, 1).view(-1)

    return solution

def remove_dummy_depot_from_solution(solution, problem_size):
    # solution.size: (batch, pomo, solution)
    batch_size, pomo_size, _ = solution.size()
    dummy_size = solution.size(-1) - problem_size
    solution = solution.clone()
    solution[solution < dummy_size] = 0
    solution[solution != 0] -= (dummy_size - 1)
    return solution

def rec2sol(rec):
    # input: rec (solution in linked list format)
    # reference: Ma, Yining, Zhiguang Cao, and Yeow Meng Chee. "Learning to search feasible and infeasible regions of routing problems with flexible neural k-opt." Advances in Neural Information Processing Systems 36 (2024).
    batch_size, seq_length = rec.size()
    visited_time = torch.zeros((batch_size, seq_length)).to(rec.device)
    pre = torch.zeros((batch_size), device=rec.device).long()
    for i in range(seq_length):
        visited_time[torch.arange(batch_size), rec[torch.arange(batch_size), pre]] = (i + 1)
        pre = rec[torch.arange(batch_size), pre]

    visited_time = visited_time % seq_length
    return visited_time.argsort()

def sol2rec(solution):
    # transform solution to linked list
    # solution.size: (batch, pomo, solution)
    batch_size, pomo_size, solution_size = solution.size()
    solution = solution.view(batch_size * pomo_size, -1)

    solution_pre = solution
    solution_post = torch.cat((solution[:, 1:], solution[:, :1]), 1)

    rec = solution.clone()
    rec.scatter_(1, solution_pre, solution_post)
    return rec.view(batch_size, pomo_size, -1)


def get_previous_nodes(rec, selected_points):
    """
    Get the previous node for each selected point in the linked list.

    Args:
        rec: Tensor, with shape (B, N), representing solutions for B instances, each solution is a linked list.
        selected_points: Tensor, with shape (B, 1), representing the index of a selected point in each instance.

    Returns:
        Tensor, with shape (B, 1), representing the index of the previous point for each selected point in each instance.
    """
    rec = rec.clone()
    batch_size, num_nodes = rec.size()
    selected_points = selected_points.squeeze(-1)  # Remove the last dimension, make selected_points shape (B,)

    # Create index mask, indicating which position in each instance is the selected point
    indices = torch.arange(num_nodes).expand(batch_size, num_nodes)
    mask0 = (rec == selected_points.unsqueeze(1))
    previous_nodes = torch.where(mask0, indices, torch.full_like(indices, -1))
    previous_nodes = torch.max(previous_nodes, dim=1, keepdim=True).values  # Get the index of the previous node

    return previous_nodes

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.obj = []
        self.context = []
        self.context2 = []
        self.feasible = []
        self.soft_feasible = []
        self.cum_demand = []
        self.partial_sum_wrt_route_plan = []
        self.out_node_penalty = []
        self.out_penalty = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.obj[:]
        del self.context[:]
        del self.context2[:]
        del self.feasible[:]
        del self.soft_feasible[:]
        del self.cum_demand[:]
        del self.partial_sum_wrt_route_plan[:]
        del self.out_node_penalty[:]
        del self.out_penalty[:]


def dummify(input, dummy_size):
    # input.shape: (batch_size, problem_size, 0/x)
    # output.shape: (batch_size, problem_size+dummy_size, 0/x)

    dummy = input[:, :1]
    dummy = dummy.repeat_interleave(dummy_size, dim=1)
    output = torch.cat([dummy, input], dim=1)

    return output

class metric_logger:
    def __init__(self, problem="CVRP", dual_decoder=False):
        self.dummy_size = AverageMeter()
        self.coefficient = AverageMeter()
        self.sigma1 = AverageMeter()
        self.sigma2 = AverageMeter()
        self.lambda_tw = AverageMeter()
        self.lambda_demand = AverageMeter()
        self.lambda_backhaul = AverageMeter()
        self.lambda_dl = AverageMeter()
        self.improve_metrics = {
            "current_score": AverageMeter(), # best during improvement
            "bsf_score": AverageMeter(), # best during improvement
            "epsilon_fsb_bsf_score": AverageMeter(), # best during improvement
            "improve_reward": AverageMeter(), # improve_reward
            "reg_reward": AverageMeter(),  # reg to jump
            "bonus_reward": AverageMeter(),  # bonus to explore e-feasible
            "loss": AverageMeter(),
            "large_model_loss": AverageMeter(),
            "lora_loss": AverageMeter(),
            "sol_infeasible_rate": AverageMeter(), # best during improvement
            "ins_infeasible_rate": AverageMeter(), # best during improvement
            "soft_sol_infeasible_rate": AverageMeter(), # best during improvement
            "soft_ins_infeasible_rate": AverageMeter(), # best during improvement
            "feasible_dist_mean": AverageMeter(), # best during improvement
            "feasible_dist_max_pomo_mean": AverageMeter(), # best during improvement
            "epsilon_feasible_dist_mean": AverageMeter(), # best during improvement
            "epsilon_feasible_dist_max_pomo_mean": AverageMeter(), # best during improvement
            "out": AverageMeter(),
            "out_nodes": AverageMeter(),
            "entropy": AverageMeter()
        }
        self.construct_metrics = {
            "score": AverageMeter(),
            "loss": AverageMeter(),
            "improvement_value": AverageMeter(),
            "construct_RL_loss": AverageMeter(),
            "diversity_loss": AverageMeter(),
            "is_improved": AverageMeter(),
            "imitation_loss": AverageMeter(),
            "sol_infeasible_rate": AverageMeter(),
            "ins_infeasible_rate": AverageMeter(),
            "feasible_dist_mean": AverageMeter(),
            "feasible_dist_max_pomo_mean": AverageMeter(),
            "out": AverageMeter(),
            "out_nodes": AverageMeter(),
        }
        self.reconstruct_metrics = {
            "score": AverageMeter(),
            "loss": AverageMeter(),
            "improvement_value": AverageMeter(),
            "construct_RL_loss": AverageMeter(),
            "diversity_loss": AverageMeter(),
            "sol_infeasible_rate": AverageMeter(),
            "ins_infeasible_rate": AverageMeter(),
            "feasible_dist_mean": AverageMeter(),
            "feasible_dist_max_pomo_mean": AverageMeter(),
            "out": AverageMeter(),
            "out_nodes": AverageMeter(),
        }
        if problem in ["OVRPBLTW", "OVRPLTW", "VRPBLTW"]:
            self.improve_metrics["dlout"] = AverageMeter()
            self.improve_metrics["dlout_nodes"] = AverageMeter()
            self.improve_metrics["capacity_out"] = AverageMeter()
            self.improve_metrics["capacity_out_nodes"] = AverageMeter()
            self.improve_metrics["backhaul_out"] = AverageMeter()
            self.improve_metrics["backhaul_out_nodes"] = AverageMeter()
            self.improve_metrics["tw_out"] = AverageMeter()
            self.improve_metrics["tw_out_nodes"] = AverageMeter()

            self.improve_metrics["tw_out_ratio"] = AverageMeter()
            self.improve_metrics["capacity_out_ratio"] = AverageMeter()
            self.improve_metrics["backhaul_out_ratio"] = AverageMeter()
            self.improve_metrics["dlout_ratio"] = AverageMeter()
            self.improve_metrics["out_ratio"] = AverageMeter()

            self.improve_metrics["cons_tw_out_ratio"] = AverageMeter()
            self.improve_metrics["cons_capacity_out_ratio"] = AverageMeter()
            self.improve_metrics["cons_backhaul_out_ratio"] = AverageMeter()
            self.improve_metrics["cons_dlout_ratio"] = AverageMeter()
            self.improve_metrics["cons_out_ratio"] = AverageMeter()

            self.construct_metrics["dlout"] = AverageMeter()
            self.construct_metrics["dlout_nodes"] = AverageMeter()
            self.construct_metrics["capacity_out"] = AverageMeter()
            self.construct_metrics["capacity_out_nodes"] = AverageMeter()

            self.reconstruct_metrics["dlout"] = AverageMeter()
            self.reconstruct_metrics["dlout_nodes"] = AverageMeter()
            self.reconstruct_metrics["capacity_out"] = AverageMeter()
            self.reconstruct_metrics["capacity_out_nodes"] = AverageMeter()
        if dual_decoder:
            self.construct_metrics["loss1"] = AverageMeter()
            self.construct_metrics["loss2"] = AverageMeter()


class val_metric_logger:
    def __init__(self, trainer):
        self.improve_metrics = {
            "no_aug_score": torch.zeros(0).to(trainer.device),
            "aug_score": torch.zeros(0).to(trainer.device),
            "sol_infeasible_rate": AverageMeter(),
            "ins_infeasible_rate": AverageMeter(),
            "no_aug_feasible": torch.zeros(0).to(trainer.device),
            "aug_feasible": torch.zeros(0).to(trainer.device),
            "no_aug_out": torch.zeros(0).to(trainer.device),
            "no_aug_out_nodes": torch.zeros(0).to(trainer.device),
            "aug_out": torch.zeros(0).to(trainer.device),
            "aug_out_nodes": torch.zeros(0).to(trainer.device),
            "no_aug_gap_list": 0.0,
            "aug_gap_list": 0.0,
        }
        self.construct_metrics = {
            "no_aug_score": torch.zeros(0).to(trainer.device),
            "aug_score": torch.zeros(0).to(trainer.device),
            "sol_infeasible_rate": AverageMeter(),
            "ins_infeasible_rate": AverageMeter(),
            "no_aug_feasible": torch.zeros(0).to(trainer.device),
            "aug_feasible": torch.zeros(0).to(trainer.device),
            "no_aug_out": torch.zeros(0).to(trainer.device),
            "no_aug_out_nodes": torch.zeros(0).to(trainer.device),
            "aug_out": torch.zeros(0).to(trainer.device),
            "aug_out_nodes": torch.zeros(0).to(trainer.device),
            "no_aug_gap_list": 0.0,
            "aug_gap_list": 0.0,
        }
        self.reconstruct_metrics = {
            "no_aug_score": torch.zeros(0).to(trainer.device),
            "aug_score": torch.zeros(0).to(trainer.device),
            "sol_infeasible_rate": AverageMeter(),
            "ins_infeasible_rate": AverageMeter(),
            "no_aug_feasible": torch.zeros(0).to(trainer.device),
            "aug_feasible": torch.zeros(0).to(trainer.device),
            "no_aug_gap_list": 0.0,
            "aug_gap_list": 0.0,
        }
        self.reconstruct_masked_metrics = {
            "no_aug_score": torch.zeros(0).to(trainer.device),
            "aug_score": torch.zeros(0).to(trainer.device),
            "sol_infeasible_rate": AverageMeter(),
            "ins_infeasible_rate": AverageMeter(),
            "no_aug_feasible": torch.zeros(0).to(trainer.device),
            "aug_feasible": torch.zeros(0).to(trainer.device),
            "no_aug_gap_list": 0.0,
            "aug_gap_list": 0.0,
        }
        if trainer.problem in ["OVRPBLTW", "OVRPLTW", "VRPBLTW"]:
            self.improve_metrics["no_aug_total_out_of_dl"] = torch.zeros(0).to(trainer.device)
            self.improve_metrics["no_aug_out_of_dl_nodes"] = torch.zeros(0).to(trainer.device)
            self.improve_metrics["no_aug_total_out_of_capacity"] = torch.zeros(0).to(trainer.device)
            self.improve_metrics["no_aug_out_of_capacity_nodes"] = torch.zeros(0).to(trainer.device)
            self.improve_metrics["aug_total_out_of_dl"] = torch.zeros(0).to(trainer.device)
            self.improve_metrics["aug_out_of_dl_nodes"] = torch.zeros(0).to(trainer.device)
            self.improve_metrics["aug_total_out_of_capacity"] = torch.zeros(0).to(trainer.device)
            self.improve_metrics["aug_out_of_capacity_nodes"] = torch.zeros(0).to(trainer.device)
            self.construct_metrics["no_aug_total_out_of_dl"] = torch.zeros(0).to(trainer.device)
            self.construct_metrics["no_aug_out_of_dl_nodes"] = torch.zeros(0).to(trainer.device)
            self.construct_metrics["no_aug_total_out_of_capacity"] = torch.zeros(0).to(trainer.device)
            self.construct_metrics["no_aug_out_of_capacity_nodes"] = torch.zeros(0).to(trainer.device)
            self.construct_metrics["aug_total_out_of_dl"] = torch.zeros(0).to(trainer.device)
            self.construct_metrics["aug_out_of_dl_nodes"] = torch.zeros(0).to(trainer.device)
            self.construct_metrics["aug_total_out_of_capacity"] = torch.zeros(0).to(trainer.device)
            self.construct_metrics["aug_out_of_capacity_nodes"] = torch.zeros(0).to(trainer.device)

    def _construct_tensor_update(self, key, value):
        self.construct_metrics[key] = torch.cat((self.construct_metrics[key], value), dim=0)

    def _improve_tensor_update(self, key, value):
        self.improve_metrics[key] = torch.cat((self.improve_metrics[key], value), dim=0)

    def _reconstruct_tensor_update(self, key, value):
        self.reconstruct_metrics[key] = torch.cat((self.reconstruct_metrics[key], value), dim=0)

    def _reconstruct_masked_tensor_update(self, key, value):
        self.reconstruct_masked_metrics[key] = torch.cat((self.reconstruct_masked_metrics[key], value), dim=0)

    def _log_output(self, trainer):
        # construction
        no_aug_score = self.construct_metrics["no_aug_score"]
        no_aug_feasible = self.construct_metrics["no_aug_feasible"]
        aug_score = self.construct_metrics["aug_score"]
        aug_feasible = self.construct_metrics["aug_feasible"]
        if trainer.trainer_params["fsb_dist_only"]:
            if trainer.rank == 0: print(">> Only feasible solutions are under consideration!")
            self.construct_metrics["no_aug_score_list"] = round(no_aug_score[no_aug_feasible.bool()].mean().item(), 4)
            self.construct_metrics["aug_score_list"] = round(aug_score[aug_feasible.bool()].mean().item(), 4)
        else:
            self.construct_metrics["no_aug_score_list"] = round(no_aug_score.mean().item(), 4)
            self.construct_metrics["aug_score_list"] = round(aug_score.mean().item(), 4)

        sol_infeasible_rate = self.construct_metrics["sol_infeasible_rate"]
        ins_infeasible_rate = self.construct_metrics["ins_infeasible_rate"]
        self.construct_metrics["sol_infeasible_rate_list"] = round(sol_infeasible_rate.avg.item() *100, 3)
        self.construct_metrics["ins_infeasible_rate_list"] = round(ins_infeasible_rate.avg.item() *100, 3)
        # improvement
        if trainer.trainer_params["improve_steps"] > 0.:
            no_aug_score = self.improve_metrics["no_aug_score"]
            no_aug_feasible = self.improve_metrics["no_aug_feasible"]
            aug_score = self.improve_metrics["aug_score"]
            aug_feasible = self.improve_metrics["aug_feasible"]
            if trainer.trainer_params["fsb_dist_only"]:
                self.improve_metrics["no_aug_score_list"] = round(no_aug_score[no_aug_feasible.bool()].mean().item(), 4)
                self.improve_metrics["aug_score_list"] = round(aug_score[aug_feasible.bool()].mean().item(), 4)
            else:
                self.improve_metrics["no_aug_score_list"] = round(no_aug_score.mean().item(), 4)
                self.improve_metrics["aug_score_list"] = round(aug_score.mean().item(), 4)

            sol_infeasible_rate = self.improve_metrics["sol_infeasible_rate"]
            ins_infeasible_rate = self.improve_metrics["ins_infeasible_rate"]
            self.improve_metrics["sol_infeasible_rate_list"] = round(sol_infeasible_rate.avg.item() *100, 3)
            self.improve_metrics["ins_infeasible_rate_list"] = round(ins_infeasible_rate.avg.item() *100, 3)
            if trainer.trainer_params["reconstruct"]:
                no_aug_score = self.reconstruct_metrics["no_aug_score"]
                no_aug_feasible = self.reconstruct_metrics["no_aug_feasible"]
                aug_score = self.reconstruct_metrics["aug_score"]
                aug_feasible = self.reconstruct_metrics["aug_feasible"]
                if trainer.trainer_params["fsb_dist_only"]:
                    if trainer.rank == 0: print(">> Only feasible solutions are under consideration!")
                    self.reconstruct_metrics["no_aug_score_list"] = round(no_aug_score[no_aug_feasible.bool()].mean().item(), 4)
                    self.reconstruct_metrics["aug_score_list"] = round(aug_score[aug_feasible.bool()].mean().item(), 4)
                else:
                    self.reconstruct_metrics["no_aug_score_list"] = round(no_aug_score.mean().item(), 4)
                    self.reconstruct_metrics["aug_score_list"] = round(aug_score.mean().item(), 4)
                sol_infeasible_rate = self.reconstruct_metrics["sol_infeasible_rate"]
                ins_infeasible_rate = self.reconstruct_metrics["ins_infeasible_rate"]
                self.reconstruct_metrics["sol_infeasible_rate_list"] = round(sol_infeasible_rate.avg.item() * 100, 3)
                self.reconstruct_metrics["ins_infeasible_rate_list"] = round(ins_infeasible_rate.avg.item() * 100, 3)
        # reconstruction
        if trainer.tester_params["aux_mask"]:
            no_aug_score = self.reconstruct_masked_metrics["no_aug_score"]
            no_aug_feasible = self.reconstruct_masked_metrics["no_aug_feasible"]
            aug_score = self.reconstruct_masked_metrics["aug_score"]
            aug_feasible = self.reconstruct_masked_metrics["aug_feasible"]
            if trainer.trainer_params["fsb_dist_only"]:
                if trainer.rank == 0: print(">> Only feasible solutions are under consideration!")
                self.reconstruct_masked_metrics["no_aug_score_list"] = round(no_aug_score[no_aug_feasible.bool()].mean().item(), 4)
                self.reconstruct_masked_metrics["aug_score_list"] = round(aug_score[aug_feasible.bool()].mean().item(), 4)
            else:
                self.reconstruct_masked_metrics["no_aug_score_list"] = round(no_aug_score.mean().item(), 4)
                self.reconstruct_masked_metrics["aug_score_list"] = round(aug_score.mean().item(), 4)
            sol_infeasible_rate = self.reconstruct_masked_metrics["sol_infeasible_rate"]
            ins_infeasible_rate = self.reconstruct_masked_metrics["ins_infeasible_rate"]
            self.reconstruct_masked_metrics["sol_infeasible_rate_list"] = round(sol_infeasible_rate.avg.item() *100, 3)
            self.reconstruct_masked_metrics["ins_infeasible_rate_list"] = round(ins_infeasible_rate.avg.item() *100, 3)

    def _calculate_gap(self, trainer, opt_sol):
        # construction
        no_aug_score = self.construct_metrics["no_aug_score"]
        no_aug_feasible = self.construct_metrics["no_aug_feasible"]
        aug_score = self.construct_metrics["aug_score"]
        aug_feasible = self.construct_metrics["aug_feasible"]
        if trainer.trainer_params["fsb_dist_only"]:
            gap = (no_aug_score[no_aug_feasible.bool()] - opt_sol[no_aug_feasible.bool()]) / opt_sol[no_aug_feasible.bool()] * 100
            aug_gap = (aug_score[aug_feasible.bool()] - opt_sol[aug_feasible.bool()]) / opt_sol[aug_feasible.bool()] * 100
        else:
            gap = (no_aug_score - opt_sol) / opt_sol * 100
            aug_gap = (aug_score - opt_sol) / opt_sol * 100
        self.construct_metrics["no_aug_gap_list"] = round(gap.mean().item(), 4)
        self.construct_metrics["aug_gap_list"] = round(aug_gap.mean().item(), 4)
        # improvement
        if trainer.trainer_params["improve_steps"] > 0.:
            no_aug_score = self.improve_metrics["no_aug_score"]
            no_aug_feasible = self.improve_metrics["no_aug_feasible"]
            aug_score = self.improve_metrics["aug_score"]
            aug_feasible = self.improve_metrics["aug_feasible"]
            if trainer.trainer_params["fsb_dist_only"]:
                gap = (no_aug_score[no_aug_feasible.bool()] - opt_sol[no_aug_feasible.bool()]) / opt_sol[no_aug_feasible.bool()] * 100
                aug_gap = (aug_score[aug_feasible.bool()] - opt_sol[aug_feasible.bool()]) / opt_sol[aug_feasible.bool()] * 100
            else:
                gap = (no_aug_score - opt_sol) / opt_sol * 100
                aug_gap = (aug_score - opt_sol) / opt_sol * 100

            self.improve_metrics["no_aug_gap_list"] = round(gap.mean().item(), 4)
            self.improve_metrics["aug_gap_list"] = round(aug_gap.mean().item(), 4)
            if trainer.trainer_params["reconstruct"]:
                no_aug_score = self.reconstruct_metrics["no_aug_score"]
                no_aug_feasible = self.reconstruct_metrics["no_aug_feasible"]
                aug_score = self.reconstruct_metrics["aug_score"]
                aug_feasible = self.reconstruct_metrics["aug_feasible"]
                if trainer.trainer_params["fsb_dist_only"]:
                    gap = (no_aug_score[no_aug_feasible.bool()] - opt_sol[no_aug_feasible.bool()]) / opt_sol[no_aug_feasible.bool()] * 100
                    aug_gap = (aug_score[aug_feasible.bool()] - opt_sol[aug_feasible.bool()]) / opt_sol[aug_feasible.bool()] * 100
                else:
                    gap = (no_aug_score - opt_sol) / opt_sol * 100
                    aug_gap = (aug_score - opt_sol) / opt_sol * 100
                self.reconstruct_metrics["no_aug_gap_list"] = round(gap.mean().item(), 4)
                self.reconstruct_metrics["aug_gap_list"] = round(aug_gap.mean().item(), 4)
        # reconstruction
        if trainer.tester_params["aux_mask"]:
            no_aug_score = self.reconstruct_masked_metrics["no_aug_score"]
            no_aug_feasible = self.reconstruct_masked_metrics["no_aug_feasible"]
            aug_score = self.reconstruct_masked_metrics["aug_score"]
            aug_feasible = self.reconstruct_masked_metrics["aug_feasible"]
            if trainer.trainer_params["fsb_dist_only"]:
                gap = (no_aug_score[no_aug_feasible.bool()] - opt_sol[no_aug_feasible.bool()]) / opt_sol[no_aug_feasible.bool()] * 100
                aug_gap = (aug_score[aug_feasible.bool()] - opt_sol[aug_feasible.bool()]) / opt_sol[aug_feasible.bool()] * 100
            else:
                gap = (no_aug_score - opt_sol) / opt_sol * 100
                aug_gap = (aug_score - opt_sol) / opt_sol * 100
            self.reconstruct_masked_metrics["no_aug_gap_list"] = round(gap.mean().item(), 4)
            self.reconstruct_masked_metrics["aug_gap_list"] = round(aug_gap.mean().item(), 4)


class ValidationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def kendall_tau_distance(sol):
    # 计算每对解之间的Kendall tau距离
    B, P, solution_size = sol.shape
    sol_expanded = sol.unsqueeze(2).expand(-1, -1, P, -1)  # B x P x P x solution_size
    sol_transposed = sol.unsqueeze(1).expand(-1, P, -1, -1)  # B x P x P x solution_size

    # 计算解之间的顺序变化
    pairwise_diff = (sol_expanded < sol_expanded.transpose(2, 3)) != (sol_transposed < sol_transposed.transpose(2, 3))

    # 计算每对解之间的交换次数
    kendall_tau_dist = pairwise_diff.float().sum(dim=-1).sum(dim=-1)  # B x P x P
    return kendall_tau_dist


def convert_to_edge_set(sol):
    # sol: B x P x solution_size
    B, P, solution_size = sol.shape
    # edges: B x P x (solution_size - 1) x 2
    edges = torch.zeros(B, P, solution_size - 1, 2, dtype=sol.dtype)

    u = sol[:, :, :-1]  # B x P x (solution_size - 1)
    v = sol[:, :, 1:]  # B x P x (solution_size - 1)
    edges[:, :, :, 0] = torch.min(u, v)  # B x P x (solution_size - 1) x 1
    edges[:, :, :, 1] = torch.max(u, v)  # B x P x (solution_size - 1) x 1

    return edges  # B x P x (solution_size - 1) x 2


def jaccard_distance(sol):
    # sol: B x P x solution_size
    edges = convert_to_edge_set(sol)  # edges: B x P x (solution_size - 1) x 2
    B, P, edge_count, _ = edges.shape

    # edges: B x P x (edge_count) x 2
    edges = edges.view(B, P, -1, 1).expand(-1, -1, -1, P)  # B x P x (edge_count*2) x P
    transposed_edges = edges.permute(0, 3, 2, 1)  # B x P x (edge_count*2) x P

    # intersection: B x P x P (表示每对解的交集大小)
    intersection = (edges == transposed_edges).all(dim=2).sum(dim=2).float()
    # union: B x P x P (表示每对解的并集大小)
    union = edge_count - intersection + (edges != transposed_edges).any(dim=2).sum(dim=2).float()

    # jaccard_dist: B x P x P (表示每对解的Jaccard距离)
    jaccard_dist = 1 - intersection / union
    return jaccard_dist

def calculate_diversity(sol, diversity):
    B, P, solution_size = sol.shape
    if diversity == "kendall_tau_distance":
        distance_matrix = kendall_tau_distance(sol)
    elif diversity == "jaccard_distance":
        distance_matrix = jaccard_distance(sol)
    else:
        raise NotImplementedError(f"Unknown diversity criterion: {diversity}")
    diversity_scores = distance_matrix.sum(dim=2) / (P - 1)
    return diversity_scores

def mixed_select(criterion, K, rnd_prob):
    batch_size, pomo_size = criterion.size()

    quality_k = int(K * (1 - rnd_prob))
    stochastic_k = K - quality_k

    # quality_k
    _, top_indices = torch.topk(criterion, quality_k, dim=1, largest=True, sorted=False)

    # stochastic_k
    all_indices = torch.arange(pomo_size).expand(batch_size, -1)
    random_indices = torch.stack([torch.randperm(pomo_size) for _ in range(batch_size)])
    remaining_indices = torch.zeros(batch_size, pomo_size, dtype=torch.bool)
    remaining_indices.scatter_(1, top_indices, True)
    remaining_indices = ~remaining_indices

    random_indices = torch.where(remaining_indices, random_indices, all_indices)
    random_indices = random_indices[:, :stochastic_k]

    # merge
    indices = torch.cat([top_indices, random_indices], dim=1)

    return indices


def select4improve(sol, reward, strategy, K, rnd_prob=1.0, diversity=None):
    batch_size, pomo_size, solution_size = sol.size()
    if K > pomo_size:
        raise ValueError
    elif K == pomo_size:
        select_sol = sol
        idx = torch.arange(K)[None, :].expand(batch_size, pomo_size)
    else:
        # "quality", "diversity", "quality_stochastic", "diversity_stochastic", "stochastic"
        if strategy == "quality":
            _, indices = torch.topk(reward, K, dim=1, largest=True, sorted=False)

        elif strategy == "stochastic":
            indices = torch.randint(0, pomo_size, (batch_size, K))

        elif strategy == "diversity":
            diversity_scores = calculate_diversity(sol, diversity)
            _, indices = torch.topk(diversity_scores, K, dim=1, largest=True, sorted=False)

        elif strategy == "quality_stochastic":
            indices = mixed_select(criterion=reward, K=K, rnd_prob=rnd_prob)

        elif strategy == "diversity_stochastic":
            diversity_scores = calculate_diversity(sol, diversity)
            indices = mixed_select(criterion=diversity_scores, K=K, rnd_prob=rnd_prob)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # select
        selected_solutions = torch.gather(sol, 1, indices.unsqueeze(-1).expand(-1, -1, solution_size))

        return selected_solutions, indices


    return select_sol, idx


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        # Initialize learnable parameters (uncertainties)
        self.sigma1 = nn.Parameter(torch.tensor(1.0))  # for task 1
        self.sigma2 = nn.Parameter(torch.tensor(0.1))  # for task 2

    def forward(self, loss1, loss2):
        # Compute the weighted loss
        weighted_loss = (1.0 / (2 * self.sigma1 ** 2)) * loss1 + \
                        (1.0 / (2 * self.sigma2 ** 2)) * loss2 + \
                        torch.log(self.sigma1) + torch.log(self.sigma2)
        return weighted_loss


def get_optimizer_step(optimizer):
    step_list = [state['step'] for state in optimizer.state.values() if 'step' in state]
    if step_list:
        return max(step_list)
    else:
        return 0
