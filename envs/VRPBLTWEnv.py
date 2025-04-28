from dataclasses import dataclass
import torch
import os, pickle
import numpy as np
from utils import *

__all__ = ['VRPBLTWEnv']

EPSILON_hardcoded = 7.105

EPSILON = {
    50: 3.67,
    100: 7.105,
}

EPSILON_TW = {
    20: 0.74,
    50: 1.85,
    100: 3.7,
    200: 7.4
}

EPSILON_C ={#epsilon
    20: 0.33,
    50: 0.625,
    100: 1.0,
    200: 1.429
}
EPSILON_L={
    50: 1.202,
    100: 2.405,
    200: 4.809
}

@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    node_service_time: torch.Tensor = None
    # shape: (batch, problem)
    node_tw_start: torch.Tensor = None
    # shape: (batch, problem)
    node_tw_end: torch.Tensor = None
    # shape: (batch, problem)
    prob_emb: torch.Tensor = None
    # shape: (num_training_prob)
    dummy_xy: torch.Tensor = None
    dummy_demand: torch.Tensor = None
    dummy_tw_end: torch.Tensor = None
    dummy_tw_start: torch.Tensor = None
    dummy_service_time: torch.Tensor = None


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    START_NODE: torch.Tensor = None
    PROBLEM: str = None
    # shape: (batch, pomo)
    selected_count: int = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    infeasible: torch.Tensor = None
    # shape: (batch, pomo)
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_time: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    open: torch.Tensor = None
    # shape: (batch, pomo)
    current_coord: torch.Tensor = None
    # shape: (batch, pomo, 2)


class VRPBLTWEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.problem = "VRPBLTW"
        self.env_params = env_params
        self.backhaul_ratio = 0.2
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.k_max = self.env_params['k_max'] if 'k_max' in env_params.keys() else None
        if 'pomo_start' in env_params.keys():
            self.pomo_size = env_params['pomo_size'] if env_params['pomo_start'] else env_params['train_z_sample_size']
        self.loc_scaler = env_params['loc_scaler'] if 'loc_scaler' in env_params.keys() else None
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in env_params.keys() else env_params['device']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.START_NODE = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.depot_node_service_time = None
        # shape: (batch, problem+1)
        self.depot_node_tw_start = None
        # shape: (batch, problem+1)
        self.depot_node_tw_end = None
        # shape: (batch, problem+1)
        self.speed = 1.0
        self.depot_start, self.depot_end = 0., 3.  # tw for depot [0, 3]

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        self.infeasible = None
        # shape: (batch, pomo)
        self.current_time = None
        # shape: (batch, pomo)
        self.length = None
        # shape: (batch, pomo)
        self.open = None
        # shape: (batch, pomo)
        self.current_coord = None
        # shape: (batch, pomo, 2)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size, rollout_size, problems=None, aug_factor=1):
        self.pomo_size = rollout_size
        if problems is not None:
            depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end = problems
        else:
            depot_xy, node_xy, node_demand, capacity, route_limit, service_time, tw_start, tw_end = self.get_random_problems(batch_size, self.problem_size, normalized=True)
            node_demand = node_demand / capacity.view(-1, 1)
        self.batch_size = depot_xy.size(0)
        route_limit = route_limit[:, None] if route_limit.dim() == 1 else route_limit

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = self.augment_xy_data_by_8_fold(depot_xy)
                node_xy = self.augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                route_limit = route_limit.repeat(8, 1)
                service_time = service_time.repeat(8, 1)
                tw_start = tw_start.repeat(8, 1)
                tw_end = tw_end.repeat(8, 1)
            else:
                raise NotImplementedError

        # reset pomo_size
        self.pomo_size = min(int(self.problem_size * (1 - self.backhaul_ratio)), self.pomo_size)
        self.START_NODE = torch.arange(start=1, end=self.problem_size+1)[None, :].expand(self.batch_size, -1).to(self.device)
        self.START_NODE = self.START_NODE[node_demand > 0].reshape(self.batch_size, -1)[:, :self.pomo_size]

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1)).to(self.device)
        depot_service_time = torch.zeros(size=(self.batch_size, 1)).to(self.device)
        depot_tw_start = torch.ones(size=(self.batch_size, 1)).to(self.device) * self.depot_start
        depot_tw_end = torch.ones(size=(self.batch_size, 1)).to(self.device) * self.depot_end
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)
        self.route_limit = route_limit
        # shape: (batch, 1)
        self.depot_node_service_time = torch.cat((depot_service_time, service_time), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_tw_start = torch.cat((depot_tw_start, tw_start), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_tw_end = torch.cat((depot_tw_end, tw_end), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_service_time = service_time
        self.reset_state.node_tw_start = tw_start
        self.reset_state.node_tw_end = tw_end
        self.reset_state.prob_emb = torch.FloatTensor([1, 0, 1, 1, 1]).unsqueeze(0).to(self.device)  # bit vector for [C, O, B, L, TW]

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.open = torch.zeros(self.batch_size, self.pomo_size).to(self.device)
        self.step_state.START_NODE = self.START_NODE
        self.step_state.PROBLEM = self.problem

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        self.timeout_list = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)
        self.out_of_dl_list = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)
        self.out_of_capacity_list = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)

        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1)).to(self.device)
        self.constraints_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1)).to(self.device)
        self.simulated_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1)).to(self.device)
        self.capacity_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1)).to(self.device)
        self.tw_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1)).to(self.device)
        self.duration_limit_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1)).to(self.device)

        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1)).to(self.device)
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        self.infeasible = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        # shape: (batch, pomo)
        self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.length = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.current_coord = self.depot_node_xy[:, :1, :]  # depot
        # shape: (batch, pomo, 2)

        self.dummy_xy = None
        self.dummy_demand = None
        self.dummy_size = None
        self.dummy_tw_end = None
        self.dummy_tw_start = None
        self.dummy_service_time = None

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.infeasible = self.infeasible
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected, out_reward = False, soft_constrained = False, backhaul_mask = "hard", penalty_normalize=True, generate_PI_mask=False,use_predicted_PI_mask=False, pip_step=1):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill loaded at the depot

        current_coord = self.depot_node_xy[torch.arange(self.batch_size)[:, None], selected]
        # shape: (batch, pomo, 2)
        new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)
        # shape: (batch, pomo)
        self.length = self.length + new_length
        self.length[self.at_the_depot] = 0  # reset the length of route at the depot
        self.current_coord = current_coord

        # Mask
        ####################################
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        # Only for VRPB: reset load to 0.
        # >> Old implementation
        #   a. if visit backhaul nodes in the first two POMO moves (i.e., depot -> backhaul, the route is mixed with backhauls and linehauls alternatively);
        #   b. if only backhaul nodes unserved, we relax the load to be 0 (i.e., the vehicle only visit backhauls nodes in the last few routes).
        # if self.selected_node_list.size(-1) == 1:  # POMO first move
        #     depot_backhaul = self.at_the_depot & (self.depot_node_demand[:, 1:self.pomo_size+1] < 0.)
        #     # shape: (batch, pomo)
        #     self.load[depot_backhaul] = 0.
        # else:
        # >> New implementation - Remove constraint a, the POMO start node should be a linehaul.
        unvisited_demand = demand_list + self.visited_ninf_flag
        # shape: (batch, pomo, problem+1)
        linehauls_unserved = torch.where(unvisited_demand > 0., True, False)
        reset_index = self.at_the_depot & (~linehauls_unserved.any(dim=-1))
        # shape: (batch, pomo)
        self.load[reset_index] = 0.

        # capacity constraint
        #   a. the remaining vehicle capacity >= the customer demands
        #   b. the remaining vehicle capacity <= the vehicle capacity (i.e., 1.0) [specified for backhaul; cannot ]
        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        if not soft_constrained:
            self.ninf_mask[demand_too_large] = float('-inf')
        exceed_capacity = self.load[:, :, None] - demand_list > 1.0 + round_error_epsilon
        if backhaul_mask == "hard":
            self.ninf_mask[exceed_capacity] = float('-inf')
            self.constraints_ninf_flag[demand_too_large] += 1.
            self.capacity_ninf_flag[demand_too_large] = float('-inf')
        elif backhaul_mask == "soft":
            self.constraints_ninf_flag[(exceed_capacity.int() + demand_too_large.int()) > 0] += 1.
            self.capacity_ninf_flag[(exceed_capacity.int() + demand_too_large.int()) > 0] = float('-inf')

        # duration limit constraint
        route_limit = self.route_limit[:, :, None].expand(self.batch_size, self.pomo_size, self.problem_size + 1)
        # shape: (batch, pomo, problem+1)
        # check route limit constraint: length + cur->next->depot <= route_limit
        route_too_large = self.length[:, :, None] + (self.current_coord[:, :, None, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) + \
                          (self.depot_node_xy[:, None, :1, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) > route_limit + round_error_epsilon
        # shape: (batch, pomo, problem+1)
        if not soft_constrained:
            self.ninf_mask[route_too_large] = float('-inf')
        self.constraints_ninf_flag[route_too_large] += 1.
        self.duration_limit_ninf_flag[route_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        # time window constraint
        #   current_time: the end time of serving the current node
        #   a. max(current_time + travel_time, tw_start) or current_time + travel_time <= tw_end
        #   b. vehicle should return to the depot: max(current_time + travel_time, tw_start) + service_time + dist(node, depot)/speed <= self.depot_end
        self.current_time = torch.max(self.current_time + new_length / self.speed, self.depot_node_tw_start[torch.arange(self.batch_size)[:, None], selected]) + self.depot_node_service_time[torch.arange(self.batch_size)[:, None], selected]
        self.current_time[self.at_the_depot] = 0
        # shape: (batch, pomo)
        arrival_time = torch.max(self.current_time[:, :, None] + (self.current_coord[:, :, None, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) / self.speed, self.depot_node_tw_start[:, None, :].expand(-1, self.pomo_size, -1))
        out_of_tw = arrival_time > self.depot_node_tw_end[:, None, :].expand(-1, self.pomo_size, -1) + round_error_epsilon
        # shape: (batch, pomo, problem+1)
        if not soft_constrained:
            self.ninf_mask[out_of_tw] = float('-inf')
        fail_return_depot = arrival_time + self.depot_node_service_time[:, None, :].expand(-1, self.pomo_size, -1) + (self.depot_node_xy[:, None, :1, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) / self.speed > self.depot_end + round_error_epsilon
        # shape: (batch, pomo, problem+1)
        if not soft_constrained:
            self.ninf_mask[fail_return_depot] = float('-inf')
        self.constraints_ninf_flag[(fail_return_depot.int()+out_of_tw.int()) > 0] += 1.
        self.tw_ninf_flag[(fail_return_depot.int()+out_of_tw.int()) > 0] = float('-inf')

        self.simulated_ninf_flag[self.constraints_ninf_flag>=1] = float('-inf')

        # calculate the penalty
        ####################################
        # out of time: timeout value of the selected node = (current time - service time) - tw_end
        total_timeout = self.current_time - self.depot_node_service_time[torch.arange(self.batch_size)[:, None], selected] - self.depot_node_tw_end[torch.arange(self.batch_size)[:, None], selected]
        total_timeout = torch.clamp(total_timeout - round_error_epsilon, min=0)# negative value means arrival time < tw_end, turn it into 0
        self.timeout_list = torch.cat((self.timeout_list, total_timeout[:, :, None]), dim=2) # shape: (batch, pomo, solution)
        if not soft_constrained:
            if (total_timeout!=0).any():
                print("out of tw")

        # out of duration limit: out_of_dl value = current_length - route_limit
        out_of_dl = torch.clamp(self.length - self.route_limit - round_error_epsilon, min=0)
        self.out_of_dl_list = torch.cat((self.out_of_dl_list, out_of_dl[:, :, None]), dim=2)
        if not soft_constrained:
            if (out_of_dl!=0).any():
                print("out of dl")

        # out of capacity: out_of_capacity value = -remaining capacity (i.e., remain negative capacity means already exceeding, positive means real remaining)
        out_of_capacity = torch.clamp(- self.load - round_error_epsilon, min=0)
        if backhaul_mask == "soft":
            # out of capacity value 2 = how much exceeds 1 (exceeding one means choosing too many backhaul nodes)
            out_of_capacity += torch.clamp(self.load - 1.0 - round_error_epsilon, min=0)
        if not soft_constrained:
            if (out_of_capacity!=0).any():
                print("out of capacity")
        self.out_of_capacity_list = torch.cat((self.out_of_capacity_list, out_of_capacity[:, :, None]), dim=2)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)
        # print((self.visited_ninf_flag==float('-inf')).float().sum(-1).mean())
        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.infeasible = self.infeasible
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        # returning values
        infeasible = 0.
        done = self.finished.all()
        if done:
            self.dummy_size = self.selected_node_list.size(-1) - self.problem_size
            total_timeout_reward = -self.timeout_list.sum(dim=-1)
            timeout_nodes_reward = -torch.where(self.timeout_list > round_error_epsilon, torch.ones_like(self.timeout_list), self.timeout_list).sum(-1).int()
            total_out_of_dl_reward = -self.out_of_dl_list.sum(dim=-1)
            out_of_dl_nodes_reward = -torch.where(self.out_of_dl_list > round_error_epsilon, torch.ones_like(self.out_of_dl_list),self.out_of_dl_list).sum(-1).int()
            total_out_of_capacity_reward = -self.out_of_capacity_list.sum(dim=-1)
            out_of_capacity_nodes_reward = -torch.where(self.out_of_capacity_list > round_error_epsilon, torch.ones_like(self.out_of_capacity_list), self.out_of_capacity_list).sum(-1).int()
            self.infeasible = (timeout_nodes_reward + out_of_dl_nodes_reward + out_of_capacity_nodes_reward != 0.)
            # shape: (batch, pomo)
            infeasible = self.infeasible
            reward = -self._get_travel_distance()  # note the minus sign!
            if out_reward:
                # shape: (batch, pomo)
                reward = [reward, total_timeout_reward, timeout_nodes_reward, total_out_of_dl_reward,
                          out_of_dl_nodes_reward, total_out_of_capacity_reward, out_of_capacity_nodes_reward]
        else:
            reward = None

        return self.step_state, reward, done, infeasible

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        if self.loc_scaler:
            segment_lengths = torch.round(segment_lengths * self.loc_scaler) / self.loc_scaler

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def generate_dataset(self, num_samples, problem_size, path):
        data = self.get_random_problems(num_samples, problem_size, normalized=False)
        dataset = [attr.cpu().tolist() for attr in data]
        filedir = os.path.split(path)[0]
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        with open(path, 'wb') as f:
            pickle.dump(list(zip(*dataset)), f, pickle.HIGHEST_PROTOCOL)
        print("Save VRPBLTW dataset to {}".format(path))

    def load_dataset(self, path, offset=0, num_samples=1000, disable_print=True):
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset+num_samples]
            if not disable_print:
                print(">> Load {} data ({}) from {}".format(len(data), type(data), path))
        depot_xy, node_xy, node_demand, capacity, route_limit, service_time, tw_start, tw_end = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data], [i[4] for i in data], [i[5] for i in data], [i[6] for i in data], [i[7] for i in data]
        depot_xy, node_xy, node_demand, capacity, route_limit, service_time, tw_start, tw_end = torch.Tensor(depot_xy), torch.Tensor(node_xy), torch.Tensor(node_demand), torch.Tensor(capacity), torch.Tensor(route_limit), torch.Tensor(service_time), torch.Tensor(tw_start), torch.Tensor(tw_end)
        node_demand = node_demand / capacity.view(-1, 1)
        data = (depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end)
        return data

    def get_random_problems(self, batch_size, problem_size, normalized=True):
        depot_xy = torch.rand(size=(batch_size, 1, 2))  # (batch, 1, 2)
        node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)

        if problem_size == 20:
            demand_scaler = 30
        elif problem_size == 50:
            demand_scaler = 40
        elif problem_size == 100:
            demand_scaler = 50
        elif problem_size == 200:
            demand_scaler = 70
        elif problem_size == 10:
            demand_scaler = 20
        else:
            raise NotImplementedError

        route_limit = torch.ones(batch_size) * 3.0

        # time windows (vehicle speed = 1.):
        #   1. The setting of "MTL for Routing Problem with Zero-Shot Generalization".
        """
        self.depot_start, self.depot_end = 0., 4.6.
        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * torch.rand(batch_size, problem_size)
        tw_length = b + (c - b) * torch.rand(batch_size, problem_size)
        c = (node_xy - depot_xy).norm(p=2, dim=-1)
        h_max = (self.depot_end - service_time - tw_length) / c * self.speed - 1
        tw_start = (1 + (h_max - 1) * torch.rand(batch_size, problem_size)) * c / self.speed
        tw_end = tw_start + tw_length
        """
        #   2. See "Learning to Delegate for Large-scale Vehicle Routing" in NeurIPS 2021.
        #   Note: this setting follows a similar procedure as in Solomon, and therefore is more realistic and harder.
        service_time = torch.ones(batch_size, problem_size) * 0.2
        travel_time = (node_xy - depot_xy).norm(p=2, dim=-1) / self.speed
        a, b = self.depot_start + travel_time, self.depot_end - travel_time - service_time
        time_centers = (a - b) * torch.rand(batch_size, problem_size) + b
        time_half_width = (service_time / 2 - self.depot_end / 3) * torch.rand(batch_size, problem_size) + self.depot_end / 3
        tw_start = torch.clamp(time_centers - time_half_width, min=self.depot_start, max=self.depot_end)
        tw_end = torch.clamp(time_centers + time_half_width, min=self.depot_start, max=self.depot_end)
        # shape: (batch, problem)

        # check tw constraint: feasible solution must exist (i.e., depot -> a random node -> depot must be valid).
        instance_invalid, round_error_epsilon = False, 0.00001
        total_time = torch.max(0 + (depot_xy - node_xy).norm(p=2, dim=-1) / self.speed, tw_start) + service_time + (node_xy - depot_xy).norm(p=2, dim=-1) / self.speed > self.depot_end + round_error_epsilon
        # (batch, problem)
        instance_invalid = total_time.any()

        if instance_invalid:
            print(">> Invalid instances, Re-generating ...")
            return self.get_random_problems(batch_size, problem_size, normalized=normalized)
        elif normalized:
            node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)  # (batch, problem)
            backhauls_index = torch.randperm(problem_size)[:int(problem_size * self.backhaul_ratio)]  # randomly select 20% customers as backhaul ones
            node_demand[:, backhauls_index] = -1 * node_demand[:, backhauls_index]
            return depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end
        else:
            node_demand = torch.Tensor(np.random.randint(1, 10, size=(batch_size, problem_size)))  # (unnormalized) shape: (batch, problem)
            backhauls_index = torch.randperm(problem_size)[:int(problem_size * self.backhaul_ratio)]
            node_demand[:, backhauls_index] = -1 * node_demand[:, backhauls_index]
            capacity = torch.Tensor(np.full(batch_size, demand_scaler))
            return depot_xy, node_xy, node_demand, capacity, route_limit, service_time, tw_start, tw_end

    def augment_xy_data_by_8_fold(self, xy_data):
        # xy_data.shape: (batch, N, 2)

        x = xy_data[:, :, [0]]
        y = xy_data[:, :, [1]]
        # x,y shape: (batch, N, 1)

        dat1 = torch.cat((x, y), dim=2)
        dat2 = torch.cat((1 - x, y), dim=2)
        dat3 = torch.cat((x, 1 - y), dim=2)
        dat4 = torch.cat((1 - x, 1 - y), dim=2)
        dat5 = torch.cat((y, x), dim=2)
        dat6 = torch.cat((1 - y, x), dim=2)
        dat7 = torch.cat((y, 1 - x), dim=2)
        dat8 = torch.cat((1 - y, 1 - x), dim=2)

        aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
        # shape: (8*batch, N, 2)

        return aug_xy_data

    def get_initial_solutions(self, strategy, k, max_dummy_size):
        raise NotImplementedError

    def preprocessing(self, rec):
        batch_size, seq_length = rec.size()
        assert seq_length < 1000
        k = batch_size // self.depot_node_xy.size(0)
        arange = torch.arange(batch_size)

        pre = torch.zeros(batch_size).long()
        route = torch.zeros(batch_size).long()
        route_plan_visited_time = torch.zeros((batch_size, seq_length)).long()
        cum_demand = torch.zeros((batch_size, seq_length))
        partial_demand_sum_wrt_route_plan = torch.zeros((batch_size, self.dummy_size))
        cum_distance = torch.zeros((batch_size, seq_length))
        travel_time_from_last_node = torch.zeros((batch_size, seq_length))
        partial_dist_sum_wrt_route_plan = torch.zeros((batch_size, self.dummy_size))
        arrival_time = torch.zeros((batch_size, seq_length))
        last_arrival_time = torch.zeros((batch_size, seq_length))
        partial_current_time_wrt_route_plan = torch.zeros((batch_size, self.dummy_size))

        for i in range(seq_length):
            next_ = rec[arange, pre]

            next_is_dummy_node = next_ < self.dummy_size
            route[next_is_dummy_node] += 1
            route_plan_visited_time[arange, next_] = (route % self.dummy_size) * int(1e3) + (i + 1) % seq_length

            new_cum_demand = partial_demand_sum_wrt_route_plan[arange, route % self.dummy_size] + self.dummy_demand[arange, next_]
            partial_demand_sum_wrt_route_plan[arange, route % self.dummy_size] = new_cum_demand.clone()
            cum_demand[arange, next_] = new_cum_demand * (~next_is_dummy_node)

            route_for_distance = torch.where(next_is_dummy_node.clone(), route - 1,route)  # if the next node is depot, then route - 1

            current_time = partial_current_time_wrt_route_plan[arange, route_for_distance % self.dummy_size]
            last_arrival_time[arange, next_] = arrival_time[arange, pre] + self.dummy_service_time[arange, pre]# time left last node
            travel_time = (self.dummy_xy[arange, pre] - self.dummy_xy[arange, next_]).norm(p=2, dim=1)
            new_current_time = torch.max(current_time + travel_time, self.dummy_tw_start[arange, next_])
            partial_current_time_wrt_route_plan[arange, route_for_distance % self.dummy_size] = new_current_time.clone() + self.dummy_service_time[arange, next_]
            arrival_time[arange, next_] = new_current_time * (~next_is_dummy_node)

            new_cum_distance = partial_dist_sum_wrt_route_plan[arange, route_for_distance % self.dummy_size] + travel_time.clone()
            partial_dist_sum_wrt_route_plan[arange,route_for_distance % self.dummy_size] = new_cum_distance.clone()
            cum_distance[arange, next_] = new_cum_distance * (~next_is_dummy_node)
            travel_time_from_last_node[arange, next_] = travel_time.clone()

            pre = next_.clone()

       # shape: (batch*k, problem_size)
        last_arrival_time[:, 0] = 0.
        arrival_time[:, 0] = 0.
        # check by: self.timestamps.squeeze(1) == arrival_time.sort()[0]
        route_plan_0x = (route_plan_visited_time // int(1e3))

        return (
            route_plan_0x,  # route plan 0xxxxx, belongs to which route
            (route_plan_visited_time % int(1e3)),  # visited time
            cum_demand.clone(),  # cum_demand (inclusive)
            partial_demand_sum_wrt_route_plan.clone(),# partial_sum_wrt_route_plan
            arrival_time.clone(),
            last_arrival_time.clone(),
            partial_current_time_wrt_route_plan.clone(),
            cum_distance.clone(),
            travel_time_from_last_node.clone(),
            partial_dist_sum_wrt_route_plan.clone()
        )

    def check_feasibility(self):
        raise NotImplementedError  # TODO: implement
        # assert (self.visited_ninf_flag == float('-inf')).all(), "not visiting all nodes!"
        # assert torch.gather(~self.infeasible, 1, select_idx).all(), "not valid tour!"

    def get_costs(self, rec, get_context=False, check_full_feasibility=False, out_reward=False, penalty_factor=1.0, penalty_normalize=False, seperate_obj_penalty=False, non_linear=None, wo_node_penalty=False, wo_tour_penalty =False):

        k = rec.size(0) // self.depot_node_xy.size(0)
        self.dummy_size = rec.size(1) - self.problem_size

        if self.dummy_xy is None:
            self.dummy_xy = dummify(self.depot_node_xy, self.dummy_size-1).repeat_interleave(k, 0)
            self.dummy_demand = dummify(self.depot_node_demand, self.dummy_size-1).repeat_interleave(k, 0)
            self.dummy_tw_end = dummify(self.depot_node_tw_end, self.dummy_size-1).repeat_interleave(k, 0)
            self.dummy_tw_start = dummify(self.depot_node_tw_start, self.dummy_size-1).repeat_interleave(k, 0)
            self.dummy_service_time = dummify(self.depot_node_service_time, self.dummy_size-1).repeat_interleave(k, 0)

        # check full feasibility if needed
        if get_context:
            context = self.preprocessing(rec)
            # output
            # 0 route_plan_0x,  # route plan 0xxxxx, belongs to which route
            # 1 (route_plan_visited_time % int(1e3)),  # visited time
            # 2 cum_demand,  # cum_demand (inclusive)
            # 3 partial_demand_sum_wrt_route_plan,# partial_sum_wrt_route_plan
            # 4 arrival_time,
            # 5 last_arrival_time,
            # 6 partial_current_time_wrt_route_plan,
            # 7 cum_distance,
            # 8 travel_time_from_last_node,
            # 9 partial_dist_sum_wrt_route_plan
        if check_full_feasibility:
            self.check_feasibility()

        coor_next = self.dummy_xy.gather(1, rec.long().unsqueeze(-1).expand(*rec.size(), 2))
        cost = (self.dummy_xy - coor_next).norm(p=2, dim=2).sum(1)

        # constraint violation
        # 1. time window: arrival time - tw end
        exceed_time_window = torch.clamp_min(context[4] - self.dummy_tw_end, 0.0)
        out_node_penalty = (exceed_time_window > 1e-5).sum(-1).unsqueeze(0)
        out_penalty = exceed_time_window.sum(-1).unsqueeze(0)
        # 2. capacity:
        # 2-1: cum_demand - 1.00001
        out_node_penalty = torch.cat([out_node_penalty, (context[2] > 1.00001).sum(-1).unsqueeze(0)], dim = 0)
        out_penalty = torch.cat([out_penalty, torch.clamp_min(context[2] - 1.00001, 0.0).sum(dim=-1).unsqueeze(0)],dim=0)
        # out_penalty = torch.cat([out_penalty, torch.clamp_min(context[3] - 1.00001, 0.0).sum(dim=-1).unsqueeze(0)], dim = 0)
        # 2-2: backhaul: -cum_demand
        out_node_penalty = torch.cat([out_node_penalty, (context[2] < -0.00001).sum(-1).unsqueeze(0)], dim = 0)
        out_penalty = torch.cat([out_penalty, torch.clamp_min(-context[2], 0.0).sum(dim=-1).unsqueeze(0)], dim = 0)
        # 3. duration limit: cum_distance - self.route_limit
        out_node_penalty = torch.cat([out_node_penalty, (context[7] > (self.route_limit[0,0] + 0.00001)).sum(-1).unsqueeze(0)], dim = 0)
        out_penalty = torch.cat([out_penalty, torch.clamp_min(context[7] - (self.route_limit[0,0] + 0.00001), 0.0).sum(dim=-1).unsqueeze(0)], dim = 0)
        # out_penalty = torch.cat([out_penalty, torch.clamp_min(context[9] - (self.route_limit.repeat_interleave(k, 0) + 0.00001), 0.0).sum(dim=-1).unsqueeze(0)], dim = 0)

        if out_reward:
            if isinstance(penalty_factor, torch.Tensor):
                if seperate_obj_penalty or non_linear=="step":
                    cost = [cost, (penalty_factor.unsqueeze(1) * (out_node_penalty + out_penalty)).sum(0)]
                elif non_linear=="scalarization":
                    penalty = (out_node_penalty + out_penalty).sum(0)
                    cost = ((penalty)/(cost+penalty)) * cost + ((cost)/(cost+penalty)) * penalty
                else:
                    cost = cost + (penalty_factor.unsqueeze(1) * (out_node_penalty + out_penalty)).sum(0)
            else:
                if seperate_obj_penalty or non_linear=="step":
                    cost = [cost, penalty_factor * (out_node_penalty.sum(0) + out_penalty.sum(0))]
                elif non_linear=="scalarization":
                    penalty = (out_node_penalty + out_penalty).sum(0)
                    cost = ((penalty)/(cost+penalty)) * cost + ((cost)/(cost+penalty)) * penalty
                else:
                    cost = cost + penalty_factor * (out_node_penalty.sum(0) + out_penalty.sum(0))

        # get context
        if get_context:
            return cost, context, out_penalty, out_node_penalty
        else:
            return cost, out_penalty, out_node_penalty

    def get_dynamic_feature(self, context, with_infsb_feature, tw_normalize=False):

        (route_plan_0x, visited_time, cum_demand, partial_sum_wrt_route_plan,
         arrival_time, last_arrival_time, partial_current_time_wrt_route_plan,
         cum_distance, travel_time_from_last_node, partial_dist_sum_wrt_route_plan) = context

        batch_size, seq_length = arrival_time.size()
        k = batch_size // self.depot_node_xy.size(0)
        # capacity
        demand = self.dummy_demand.unsqueeze(-1)
        is_depot = (demand == 0).float()
        cum_demand = cum_demand.unsqueeze(-1)
        route_total_demand_per_node = partial_sum_wrt_route_plan.gather(-1, route_plan_0x).unsqueeze(-1)
        infeasibility_indicator_after_visit_capacity = torch.clamp_min(cum_demand - 1.00001, 0.0) > 0
        infeasibility_indicator_before_visit_capacity = torch.clamp_min((cum_demand - demand) - 1.00001, 0.0) > 0
        # backhaul
        is_backhaul_node = (demand<0).float()
        infeasibility_indicator_after_visit_backhaul = torch.clamp_min(-cum_demand, 0.0) > 0
        infeasibility_indicator_before_visit_backhaul = (torch.clamp_min(-(cum_demand - demand), 0.0) > 0) * (infeasibility_indicator_after_visit_backhaul)
        # tw
        exceed_time_window = torch.clamp_min(arrival_time - self.dummy_tw_end, 0.0)
        route_total_tw_per_node = partial_current_time_wrt_route_plan.gather(-1, route_plan_0x).unsqueeze(-1)
        infeasibility_indicator_after_visit_tw = exceed_time_window > 0
        # duration limit
        cum_distance = cum_distance.unsqueeze(-1)
        route_total_distance_per_node = partial_dist_sum_wrt_route_plan.gather(-1, route_plan_0x).unsqueeze(-1)
        infeasibility_indicator_after_visit_distance = torch.clamp_min(cum_distance - (self.route_limit.repeat_interleave(k, 0).unsqueeze(-1)+0.00001), 0.0) > 0
        last_cum_distance = cum_distance - travel_time_from_last_node.unsqueeze(-1)
        infeasibility_indicator_before_visit_distance = torch.clamp_min(last_cum_distance - (self.route_limit.repeat_interleave(k, 0).unsqueeze(-1)+0.00001), 0.0) > 0

        to_actor = torch.cat((
            is_depot,
            # capacity (accumulative)
            cum_demand,
            route_total_demand_per_node - cum_demand,
            # backhaul (accumulative)
            is_backhaul_node,
            # tw (not accumulative: if the last visited node is infeasible, the current node can be either infeasible or feasible)
            arrival_time.unsqueeze(-1),
            exceed_time_window.unsqueeze(-1),
            last_arrival_time.unsqueeze(-1), # tw_start.unsqueeze(-1),
            route_total_tw_per_node - (arrival_time + self.dummy_service_time).unsqueeze(-1),
            # duration limit (accumulative)
            cum_distance,
            route_total_distance_per_node - cum_distance,
            # capacity (accumulative)
            infeasibility_indicator_before_visit_capacity.float(),
            infeasibility_indicator_after_visit_capacity.float(),
            # backhaul (accumulative)
            infeasibility_indicator_before_visit_backhaul.float(),
            infeasibility_indicator_after_visit_backhaul.float(),
            # tw (not accumulative: if the last visited node is infeasible, the current node can be either infeasible or feasible)
            infeasibility_indicator_after_visit_tw.unsqueeze(-1),
            # duration limit (accumulative)
            infeasibility_indicator_before_visit_distance.float(),
            infeasibility_indicator_after_visit_distance.float(),
        ), -1)  # the node features

        feature = torch.cat([self.dummy_xy, self.dummy_demand.unsqueeze(-1), self.dummy_tw_start.unsqueeze(-1), self.dummy_tw_end.unsqueeze(-1)], dim=-1)
        supplement_feature = to_actor
        if not with_infsb_feature:
            supplement_feature = to_actor[:, :, :10]
        feature = torch.cat((feature, supplement_feature), dim=-1)

        return visited_time, None, feature

    def improvement_step(self, rec, action, obj, feasible_history, t, improvement_method = "kopt", epsilon=EPSILON_hardcoded, weights=0, out_reward = False, penalty_factor=1., penalty_normalize=False, insert_before=True, seperate_obj_penalty=False, non_linear=None, n2s_decoder=False):

        total_history = feasible_history.size(1)
        if seperate_obj_penalty:
            obj, penalty = obj
            pre_penalty_bsf =  penalty[:, 1:].clone()  # batch_size, 3 (current, bsf, tsp_bsf)
        pre_bsf = obj[:, 1:].clone()  # batch_size, 3 (current, bsf, tsp_bsf)
        feasible_history = feasible_history.clone()  # bs, total_history

        # improvement
        if improvement_method == "kopt":
            next_state = self.k_opt(rec, action)
        elif improvement_method == "rm_n_insert":
            next_state = self.rm_n_insert(rec, action, insert_before=insert_before)
        else:
            raise NotImplementedError()

        next_obj, context, out_penalty, out_node_penalty = self.get_costs(next_state, get_context=True, out_reward=out_reward,
                                                                          penalty_factor=penalty_factor, penalty_normalize=penalty_normalize,
                                                                          seperate_obj_penalty=seperate_obj_penalty, non_linear=non_linear)

        # MDP step
        non_feasible_cost_total = out_penalty.sum(0)
        # print(">>>>>>>>>>>>>>>>>", non_feasible_cost_total.mean(), non_feasible_cost_total.max(), non_feasible_cost_total.min())
        feasible = non_feasible_cost_total <= 0.0
        soft_infeasible = (non_feasible_cost_total <= epsilon) & (non_feasible_cost_total > 0.)
        # print(">>>>>>>>>>>>>>>>>", soft_infeasible.sum(), soft_infeasible.size())

        now_obj = pre_bsf.clone()
        if seperate_obj_penalty and non_linear!="step":
            next_obj, next_penalty_obj = next_obj
            now_penalty_obj = pre_penalty_bsf.clone()
        elif non_linear == "step":
            next_obj, next_penalty_obj = next_obj

        if not out_reward:
            # only update feasible obj
            if seperate_obj_penalty and non_linear!="step": now_penalty_obj[feasible, 0] = next_penalty_obj[feasible].clone()
            now_obj[feasible, 0] = next_obj[feasible].clone()
        else:
            # update all obj, obj = cost + penalty
            if seperate_obj_penalty and non_linear!="step": now_penalty_obj[:, 0] = next_penalty_obj.clone()
            if non_linear is None:
                now_obj[:, 0] = next_obj.clone()
            elif non_linear in ["fixed_epsilon", "decayed_epsilon"]: # only have reward when penalty <= epsilon
                now_obj[soft_infeasible, 0] = next_obj[soft_infeasible].clone()
            elif non_linear == "step":
                raise NotImplementedError

        # only update epsilon feasible obj
        now_obj[soft_infeasible, 1] = next_obj[soft_infeasible].clone()
        now_bsf = torch.min(pre_bsf, now_obj)
        rewards = (pre_bsf - now_bsf)  # bs,2 (feasible_reward, epsilon-feasible_reward)
        if seperate_obj_penalty:
            # reward = (pre_obj_bsf - now_obj_bsf) + (pre_penalty_bsf - now_penalty_bsf)
            # original reward: (obj+penalty)_bsf - (obj+penalty)_now
            now_penalty_obj[soft_infeasible, 1] = next_penalty_obj[soft_infeasible].clone()
            now_penalty_bsf = torch.min(pre_penalty_bsf, now_penalty_obj)
            rewards += (pre_penalty_bsf - now_penalty_bsf)

        # feasible history step
        if n2s_decoder: # calculate the removal record
            info, reg = None, torch.zeros((action.size(0), 1))
            assert not self.env_params["with_regular"], "n2s decoder does not support regularization reward."
            feasible_history[:, 1:] = feasible_history[:, :total_history - 1].clone()
            action_removal= torch.zeros_like(feasible_history[:,0])
            action_removal[torch.arange(action.size(0)).unsqueeze(1), action[:, :1]] = 1.
            feasible_history[:, 0] = action_removal.clone()
            context2 = torch.cat(
            (
                feasible_history, # last three removal
                feasible_history.mean(1, keepdims=True) if t > (total_history-2) else feasible_history[:,:(t+1)].mean(1, keepdims=True), # note: slightly different from N2S due to shorter improvement steps; before/after?
            ),1 )  # (batch_size, 4, solution_size)
        else:
            feasible_history[:, 1:] = feasible_history[:, :total_history - 1].clone()
            feasible_history[:, 0] = feasible.clone()

            # compute the ES features
            feasible_history_pre = feasible_history[:, 1:]
            feasible_history_post = feasible_history[:, :total_history - 1]
            f_to_if = ((feasible_history_pre == True) & (feasible_history_post == False)).sum(1, True) / (total_history - 1)
            f_to_f = ((feasible_history_pre == True) & (feasible_history_post == True)).sum(1, True) / (total_history - 1)
            if_to_f = ((feasible_history_pre == False) & (feasible_history_post == True)).sum(1, True) / (total_history - 1)
            if_to_if = ((feasible_history_pre == False) & (feasible_history_post == False)).sum(1, True) / (total_history - 1)
            f_to_if_2 = f_to_if / (f_to_if + f_to_f + 1e-5)
            f_to_f_2 = f_to_f / (f_to_if + f_to_f + 1e-5)
            if_to_f_2 = if_to_f / (if_to_f + if_to_if + 1e-5)
            if_to_if_2 = if_to_if / (if_to_f + if_to_if + 1e-5)

            # update info to decoder
            active = (t >= (total_history - 2))
            context2 = torch.cat((
                (if_to_if * active),
                (if_to_if_2 * active),
                (f_to_f * active),
                (f_to_f_2 * active),
                (if_to_f * active),
                (if_to_f_2 * active),
                (f_to_if * active),
                (f_to_if_2 * active),
                feasible.unsqueeze(-1).float(),
            ), -1)  # 9 ES features

            info = (if_to_if, if_to_f, f_to_if, f_to_f, if_to_if_2, if_to_f_2, f_to_if_2, f_to_f_2)
            # update regulation
            reg = self.f(f_to_f_2) + self.f(if_to_if_2)

        reward = torch.cat((rewards[:, :1],  # reward
                            -1 * reg * weights * 0.05 * self.env_params["with_regular"],  # regulation, alpha = 0.05
                            rewards[:, 1:2] * 0.05 * self.env_params["with_bonus"],  # bonus, beta = 0.05
                            ), -1)

        if seperate_obj_penalty:
            obj_out = [torch.cat((next_obj[:, None], now_bsf), -1), torch.cat((next_penalty_obj[:, None], now_penalty_bsf), -1)]
        else:
            obj_out = torch.cat((next_obj[:, None], now_bsf), -1)

        out = (next_state,
               reward,
               obj_out,
               feasible_history,
               context,
               context2,
               info,
               out_penalty,
               out_node_penalty
               )

        return out

    def k_opt(self, rec, action):

        _, dummy_graph_size = rec.size()

        # action bs * (K_index, K_from, K_to)
        selected_index = action[:, :self.k_max]
        left = action[:, self.k_max:2 * self.k_max]
        right = action[:, 2 * self.k_max:]

        # prepare
        rec_next = rec.clone()
        right_nodes = rec.gather(1, selected_index)
        argsort = rec.argsort()

        # new rec
        rec_next.scatter_(1, left, right)
        cur = left[:, :1].clone()
        for i in range(dummy_graph_size - 2):  # self.size - 2 is already correct
            next_cur = rec_next.gather(1, cur)
            pre_next_wrt_old = argsort.gather(1, next_cur)
            reverse_link_condition = ((cur != pre_next_wrt_old) & ~((next_cur == right_nodes).any(-1, True)))
            next_next_cur = rec_next.gather(1, next_cur)
            rec_next.scatter_(1, next_cur, torch.where(reverse_link_condition, pre_next_wrt_old, next_next_cur))
            # if i >= self.size - 2: assert (reverse_link_condition == False).all()
            cur = next_cur

        return rec_next

    def rm_n_insert(self, rec, action, insert_before=True):
        """
        Perform remove and insert operations on the linked list solutions.

        Args:
            rec: Tensor, with shape (B, N), representing solutions for B instances, each solution is a linked list.
            action: Tensor, with shape (B, rm_num * 2), representing remove and insert actions for each instance.
                     Each row contains [remove_1, insert_1, remove_2, insert_2, remove_3, insert_3, ...].
            insert_before: Boolean, if True insert before the specified node, if False insert after the specified node.

        Returns:
            Tensor, with shape (B, N), representing the updated linked list solutions after remove and insert operations.
        """
        rm_num = action.size(1) // 2
        sol = rec2sol(rec)
        batch_size, num_nodes = sol.size()
        updated_sol = sol.clone()

        # Expand action to match dimensions
        remove_indices = action[:, ::2]  # Shape (B, 3)
        insert_indices = action[:, 1::2]  # Shape (B, 3)

        for i in range(rm_num):
            # Step 1: Find the position of the node to be removed
            remove_idx = remove_indices[:, i]  # Shape (B,)
            remove_mask = (updated_sol == remove_idx.unsqueeze(1))  # Shape (B, N), Boolean mask for nodes to be removed
            remove_pos = remove_mask.nonzero(as_tuple=True)[1]  # Shape (B,), indices of nodes to be removed

            # Step 2: Remove the node from the solution
            keep_mask = ~remove_mask  # Invert the mask to keep other nodes
            sol_without_removed = torch.masked_select(updated_sol, keep_mask).view(batch_size,
                                                                                   num_nodes - 1)  # Shape (B, N-1)

            # Step 3: Find the position to insert
            insert_idx = insert_indices[:, i]  # Shape (B,)
            insert_mask = (sol_without_removed == insert_idx.unsqueeze(1))  # Shape (B, N-1)
            insert_pos = insert_mask.nonzero(as_tuple=True)[1]  # Shape (B,), indices of nodes to insert before/after

            # Step 4: Insert the removed node before or after the specified position
            new_sol = []
            for b in range(batch_size):
                pos = insert_pos[b].item()
                if insert_before:
                    new_sol.append(torch.cat((sol_without_removed[b, :pos], remove_idx[b:b + 1], sol_without_removed[b, pos:])))
                else:
                    new_sol.append(torch.cat((sol_without_removed[b, :pos + 1], remove_idx[b:b + 1], sol_without_removed[b, pos + 1:])))

            updated_sol = torch.stack(new_sol)

        return sol2rec(updated_sol.unsqueeze(1)).squeeze(1)

    def f(self, p):  # The entropy measure in Eq.(5)
        return torch.clamp(1 - 0.5 * torch.log2(2.5 * np.pi * np.e * p * (1 - p) + 1e-5), 0, 1)
