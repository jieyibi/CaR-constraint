from dataclasses import dataclass
import torch
import os, pickle
from data_generator import generate_tsptw_data
from collections import namedtuple
from utils import *
__all__ = ['TSPTWEnv']

# EPSILON = {
#     20: 0.33,
#     50: 0.625,
#     100: 1.0,
#     200: 1.429
# }

EPSILON = {
    20: 0.74,
    50: 1.85,
    100: 3.7,
    200: 7.4
}

@dataclass
class Reset_State:
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_service_time: torch.Tensor = None
    # shape: (batch, problem)
    node_tw_start: torch.Tensor = None
    # shape: (batch, problem)
    node_tw_end: torch.Tensor = None
    # shape: (batch, problem)


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
    # shape: (batch, pomo, problem)
    finished: torch.Tensor = None
    infeasible: torch.Tensor = None
    # shape: (batch, pomo)
    current_time: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    current_coord: torch.Tensor = None
    # shape: (batch, pomo, 2)


class TSPTWEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.problem = "TSPTW"
        self.env_params = env_params
        self.tw_type = env_params['tw_type']
        self.tw_duration = env_params['tw_duration']
        self.problem_size = env_params['problem_size']
        self.epsilon = EPSILON[self.problem_size]
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
        self.node_xy = None
        # shape: (batch, problem, 2)
        self.node_service_time = None
        # shape: (batch, problem)
        self.node_tw_start = None
        # shape: (batch, problem)
        self.node_tw_end = None
        # shape: (batch, problem)
        self.speed = 1.0

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        self.timestamps = None
        self.infeasibility_list = None
        self.timeout_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.visited_ninf_flag = None
        self.simulated_ninf_flag = None
        self.global_mask = None
        self.global_mask_ninf_flag = None
        self.out_of_tw_ninf_flag = None
        # shape: (batch, pomo, problem)
        self.ninf_mask = None
        # shape: (batch, pomo, problem)
        self.finished = None
        self.infeasible = None
        # shape: (batch, pomo)
        self.current_time = None
        # shape: (batch, pomo)
        self.length = None
        # shape: (batch, pomo)
        self.current_coord = None
        # shape: (batch, pomo, 2)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size, rollout_size, problems=None, aug_factor=1, normalize=True):
        self.pomo_size = rollout_size
        if problems is not None:
            node_xy, service_time, tw_start, tw_end = problems
        else:
            node_xy, service_time, tw_start, tw_end = self.get_random_problems(batch_size,
                                                                                    self.problem_size,
                                                                                    max_tw_gap=10,
                                                                                    max_tw_size=100)
        if normalize:
            # Normalize as in DPDP (Kool et. al)
            loc_factor = 100
            node_xy = node_xy / loc_factor  # Normalize
            # Normalize same as coordinates to keep unit the same, not that these values do not fall in range [0,1]!
            # Todo: should we additionally normalize somehow, e.g. by expected makespan/tour length?
            tw_start = tw_start / loc_factor
            tw_end = tw_end / loc_factor
            # Upper bound for depot = max(node ub + dist to depot), to make this tight
            tw_end[:, 0] = (torch.cdist(node_xy[:, None, 0], node_xy[:, 1:]).squeeze(1) + tw_end[:, 1:]).max(dim=-1)[0]
            # nodes_timew = nodes_timew / nodes_timew[0, 1]

        self.batch_size = node_xy.size(0)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                node_xy = self.augment_xy_data_by_8_fold(node_xy)
                service_time = service_time.repeat(8, 1)
                tw_start = tw_start.repeat(8, 1)
                tw_end = tw_end.repeat(8, 1)
            else:
                raise NotImplementedError

        self.node_xy = node_xy
        # shape: (batch, problem, 2)
        self.node_service_time = service_time
        # shape: (batch, problem)
        self.node_tw_start = tw_start
        # shape: (batch, problem)
        self.node_tw_end = tw_end
        # shape: (batch, problem)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)

        self.reset_state.node_xy = node_xy
        self.reset_state.node_service_time = service_time
        self.reset_state.node_tw_start = tw_start
        self.reset_state.node_tw_end = tw_end

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.START_NODE = torch.arange(start=1, end=self.pomo_size + 1)[None, :].expand(self.batch_size, -1).to(self.device)
        self.step_state.PROBLEM = self.problem

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        self.timestamps = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)
        self.timeout_list = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)
        self.infeasibility_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.bool).to(self.device) # True for causing infeasibility
        # shape: (batch, pomo, 0~)

        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.simulated_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.global_mask_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.global_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # shape: (batch, pomo, problem)
        self.out_of_tw_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # shape: (batch, pomo, problem)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # shape: (batch, pomo, problem)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        self.infeasible = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        # shape: (batch, pomo)
        self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.length = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.current_coord = self.node_xy[:, :1, :]  # depot
        # shape: (batch, pomo, 2)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
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

    def step(self, selected, visit_mask_only =True, out_reward = False, simulation=False,
             infsb_level=False, return_route=False, safety_layer=None,
             simulate_and_val=None, use_sl_mask=False, soft_constrained = False, backhaul_mask = None, penalty_normalize=False):
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
        current_coord = self.node_xy[torch.arange(self.batch_size)[:, None], selected]
        # shape: (batch, pomo, 2)
        new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)
        # shape: (batch, pomo)
        self.length = self.length + new_length
        self.current_coord = current_coord

        # Mask
        ####################################
        # visited mask
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem)

        # wait until time window starts
        self.current_time = (torch.max(self.current_time + new_length / self.speed,
                                       self.node_tw_start[torch.arange(self.batch_size)[:, None], selected])
                             + self.node_service_time[torch.arange(self.batch_size)[:, None], selected])
        # shape: (batch, pomo)
        self.timestamps = torch.cat((self.timestamps, self.current_time[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # time window constraint
        #   current_time: the end time of serving the current node
        #   max(current_time + travel_time, tw_start) or current_time + travel_time <= tw_end
        round_error_epsilon = 0.00001
        next_arrival_time = torch.max(self.current_time[:, :, None] + (self.current_coord[:, :, None, :] - self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) / self.speed,
                                 self.node_tw_start[:, None, :].expand(-1, self.pomo_size, -1))
        node_tw_end = self.node_tw_end[:, None, :].expand(-1, self.pomo_size, -1)

        # simulate the right infsb mask and see ATTENTION!
        if simulation and self.selected_count < self.problem_size -1:
            if isinstance(simulation, bool):
                unvisited = torch.masked_select(
                    torch.arange(self.problem_size).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.pomo_size, self.problem_size),
                    self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
                two_step_unvisited = unvisited.unsqueeze(2).repeat(1, 1, self.problem_size-self.selected_count, 1)
                diag_element = torch.diag_embed(torch.diagonal(two_step_unvisited, dim1=-2, dim2=-1))
                two_step_idx = torch.masked_select(two_step_unvisited, diag_element==0).reshape(self.batch_size, self.pomo_size, self.problem_size-self.selected_count, -1)

                # add arrival_time of the first-step nodes
                first_step_current_coord = self.node_xy.unsqueeze(1).repeat(1,self.pomo_size,1,1).gather(dim=2, index = unvisited.unsqueeze(3).expand(-1, -1, -1, 2))
                # first_step_new_length = (first_step_current_coord - current_coord.unsqueeze(2).repeat(1,1,self.problem_size-self.selected_count,1)).norm(p=2, dim=-1)
                # current_time = self.current_time.unsqueeze(-1).repeat(1, 1, self.problem_size - self.selected_count)
                # first_step_tw_end = torch.masked_select(node_tw_end, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
                # node_tw_start = self.node_tw_start[:, None, :].expand(-1, self.pomo_size, -1)
                # first_step_tw_start = torch.masked_select(node_tw_start, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
                # node_service_time = self.node_service_time[:, None, :].expand(-1, self.pomo_size, -1)
                # first_step_node_service_time= torch.masked_select(node_service_time, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
                # first_step_current_time = torch.max(current_time + first_step_new_length / self.speed, first_step_tw_start) + first_step_node_service_time
                first_step_arrival_time = torch.masked_select(next_arrival_time, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)

                # add arrival_time of the second-step nodes
                two_step_tw_end = torch.masked_select(node_tw_end, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
                two_step_tw_end = two_step_tw_end.unsqueeze(2).repeat(1, 1, self.problem_size-self.selected_count, 1)
                two_step_tw_end = torch.masked_select(two_step_tw_end, diag_element==0).reshape(self.batch_size, self.pomo_size, self.problem_size-self.selected_count, -1)

                node_tw_start = self.node_tw_start[:, None, :].expand(-1, self.pomo_size, -1)
                two_step_tw_start = torch.masked_select(node_tw_start, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
                two_step_tw_start = two_step_tw_start.unsqueeze(2).repeat(1, 1, self.problem_size-self.selected_count, 1)
                two_step_tw_start = torch.masked_select(two_step_tw_start, diag_element==0).reshape(self.batch_size, self.pomo_size, self.problem_size-self.selected_count, -1)

                node_service_time = self.node_service_time[:, None, :].expand(-1, self.pomo_size, -1)
                two_step_node_service_time= torch.masked_select(node_service_time, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
                two_step_node_service_time = two_step_node_service_time.unsqueeze(2).repeat(1, 1, self.problem_size-self.selected_count, 1)
                two_step_node_service_time = torch.masked_select(two_step_node_service_time, diag_element==0).reshape(self.batch_size, self.pomo_size, self.problem_size-self.selected_count, -1)

                two_step_current_coord = first_step_current_coord.unsqueeze(2).repeat(1, 1, self.problem_size-self.selected_count, 1, 1)
                two_step_current_coord = torch.masked_select(two_step_current_coord, diag_element.unsqueeze(-1).expand(-1, -1, -1, -1, 2) == 0).reshape(self.batch_size, self.pomo_size, self.problem_size - self.selected_count, -1, 2)
                second_step_new_length = (two_step_current_coord - first_step_current_coord.unsqueeze(3).repeat(1,1,1,self.problem_size-self.selected_count-1,1)).norm(p=2, dim=-1)
                first_step_arrival_time = first_step_arrival_time.unsqueeze(-1).repeat(1, 1, 1, self.problem_size - self.selected_count-1)
                second_step_arrival_time = torch.max(first_step_arrival_time + second_step_new_length / self.speed, two_step_tw_start) + two_step_node_service_time

                # time window constraint
                #   current_time: the end time of serving the current node
                #   max(current_time + travel_time, tw_start) or current_time + travel_time <= tw_end
                # feasibility judgement
                delta_t = self.env_params["random_delta_t"] * torch.rand(size=second_step_arrival_time.size())
                second_step_arrival_time += delta_t
                infeasible_mark = (second_step_arrival_time > two_step_tw_end + round_error_epsilon)
                selectable = (infeasible_mark == False).all(dim=-1)
                self.global_mask = infeasible_mark.sum(-1) / infeasible_mark.size(-1)
                self.global_mask_ninf_flag = torch.full((self.batch_size, self.pomo_size, self.problem_size), float('-inf'))
                self.global_mask_ninf_flag.masked_scatter_(self.visited_ninf_flag == 0, self.global_mask)

                if self.env_params["reverse"]:
                    self.simulated_ninf_flag = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
                    unselectable = (selectable == False)
                    unselectable_indices = unselectable.nonzero(as_tuple=False)
                    unvisited_indices = unvisited[unselectable_indices[:, 0], unselectable_indices[:, 1], unselectable_indices[:, 2]]
                    self.simulated_ninf_flag[unselectable_indices[:, 0], unselectable_indices[:, 1], unvisited_indices] = float('-inf')
                else: # default
                    self.simulated_ninf_flag = torch.full((self.batch_size, self.pomo_size, self.problem_size), float('-inf'))
                    selected_indices = selectable.nonzero(as_tuple=False)
                    unvisited_indices = unvisited[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]]
                    self.simulated_ninf_flag[selected_indices[:, 0], selected_indices[:, 1], unvisited_indices] = 0.

                # self.global_mask = selectable.sum(-1) / selectable.size(-1)
                # torch.masked_select(simulated_ninf_flag, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)[selectable==False] = float('-inf')
                # for b in range(self.batch_size):
                #     for p in range(self.pomo_size):
                #         for i in range(self.problem_size-self.selected_count):
                #             if selectable[b,p,i]:
                #                 simulated_ninf_flag[b,p,unvisited[b,p,i]] = 0.

            else: # use pretrained safety layer
                node_xy = self.node_xy
                if self.problem in ["TSPTW"]:
                    node_tw_start = self.node_tw_start
                    node_tw_end = self.node_tw_end
                    # shape: (batch, problem)
                    feature = torch.cat((node_xy, node_tw_start[:, :, None], node_tw_end[:, :, None]), dim=2)
                    # shape: (batch, problem, 4)
                elif self.problem in ['TSPDL']:
                    node_demand = self.node_demand
                    node_draft_limit = self.node_draft_limit
                    feature = torch.cat((node_xy, node_demand[:, :, None], node_draft_limit[:, :, None]), dim=2)
                else:
                    raise NotImplementedError

                with torch.no_grad():
                    simulation_safety_layer = simulation
                    simulation_safety_layer.eval()
                    self.simulated_ninf_flag = simulation_safety_layer(features=feature[:, None, :, :].repeat(1, self.pomo_size, 1, 1).reshape(self.batch_size * self.pomo_size, self.problem_size, -1),
                                                        unvisited=(self.visited_ninf_flag == float('-inf')).reshape(self.batch_size * self.pomo_size, -1),
                                                        current_node=self.current_node.reshape(self.batch_size * self.pomo_size, -1),
                                                        current=self.timestamps[:, :, -1].reshape(self.batch_size * self.pomo_size, -1),
                                                        no_sigmoid=False).reshape(self.batch_size, self.pomo_size,-1)
                    self.simulated_ninf_flag = torch.where(self.simulated_ninf_flag > 0.5, float('-inf'),self.simulated_ninf_flag)
                    self.simulated_ninf_flag = torch.where(self.simulated_ninf_flag != float('-inf'), 0., self.simulated_ninf_flag)
        if simulate_and_val is not None: # use safety layer as simulated_ninf_flag and generate simulated_ninf_flag ground truth too
            node_xy = self.node_xy
            if self.problem in ["TSPTW"]:
                node_tw_start = self.node_tw_start
                node_tw_end = self.node_tw_end
                # shape: (batch, problem)
                feature = torch.cat((node_xy, node_tw_start[:, :, None], node_tw_end[:, :, None]), dim=2)
                # shape: (batch, problem, 4)
            elif self.problem in ['TSPDL']:
                node_demand = self.node_demand
                node_draft_limit = self.node_draft_limit
                feature = torch.cat((node_xy, node_demand[:, :, None], node_draft_limit[:, :, None]), dim=2)
            else:
                raise NotImplementedError

            with torch.no_grad():
                simulation_safety_layer = simulate_and_val
                simulation_safety_layer.eval()
                self.simulated_ninf_flag_gt = self.simulated_ninf_flag.clone()
                self.simulated_ninf_flag = simulation_safety_layer(
                                    features=feature[:, None, :, :].repeat(1, self.pomo_size, 1, 1).reshape(self.batch_size * self.pomo_size, self.problem_size, -1),
                                    unvisited=(self.visited_ninf_flag == float('-inf')).reshape(self.batch_size * self.pomo_size, -1),
                                    current_node=self.current_node.reshape(self.batch_size * self.pomo_size, -1),
                                    current=self.timestamps[:, :, -1].reshape(self.batch_size * self.pomo_size, -1),
                                    no_sigmoid=False).reshape(self.batch_size, self.pomo_size,-1)
                self.simulated_ninf_flag = torch.where(self.simulated_ninf_flag > 0.5, float('-inf'), self.simulated_ninf_flag)
                self.simulated_ninf_flag = torch.where(self.simulated_ninf_flag != float('-inf'), 0., self.simulated_ninf_flag)

        if safety_layer and self.selected_count < self.problem_size -1:
            unvisited = torch.masked_select(torch.arange(self.problem_size).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.pomo_size,self.problem_size),
                self.visited_ninf_flag != float('-inf')).reshape(self.batch_size*self.pomo_size, -1)
            features = torch.cat([self.node_xy, self.node_tw_start[:,:,None], self.node_tw_end[:,:,None]], dim=-1)
            features = features[:, None, :, :].repeat(1,self.pomo_size,1,1).reshape(self.batch_size*self.pomo_size, self.problem_size, -1)
            current_node = self.current_node.reshape(self.batch_size*self.pomo_size, -1)
            current = self.current_time.reshape(self.batch_size*self.pomo_size, -1)

            predict_mask = safety_layer.predict_infsb_mask(features, unvisited, current_node, current)
            predict_mask = predict_mask.reshape(self.batch_size, self.pomo_size, self.problem_size)
            simulated_ninf_flag1 = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
            simulated_ninf_flag1 = torch.where(predict_mask>0.5, float('-inf'), simulated_ninf_flag1)

        out_of_tw = next_arrival_time > self.node_tw_end[:, None, :].expand(-1, self.pomo_size, -1) + round_error_epsilon
        # shape: (batch, pomo, problem)
        self.out_of_tw_ninf_flag[out_of_tw] = float('-inf')

        # timeout value of the selected node = current time - tw_end
        total_timeout = self.current_time - self.node_tw_end[torch.arange(self.batch_size)[:, None], selected]
        # negative value means current time < tw_end, turn it into 0
        total_timeout = torch.where(total_timeout<0, torch.zeros_like(total_timeout), total_timeout)
        # shape: (batch, pomo)
        self.timeout_list = torch.cat((self.timeout_list, total_timeout[:, :, None]), dim=2)

        self.ninf_mask = self.visited_ninf_flag.clone()
        if not visit_mask_only or not soft_constrained:
            self.ninf_mask[out_of_tw] = float('-inf')
            all_infsb = ((self.ninf_mask == float('-inf')).all(dim=-1)).unsqueeze(-1).expand(-1, -1, self.problem_size)
            self.ninf_mask = torch.where(all_infsb, self.visited_ninf_flag, self.ninf_mask)
        if simulation and self.selected_count < self.problem_size -1 and (not use_sl_mask): # use simulated mask once not using sl_mask
            self.ninf_mask = torch.where(self.simulated_ninf_flag==float('-inf'), float('-inf'), self.ninf_mask)
            all_infsb = ((self.ninf_mask == float('-inf')).all(dim=-1)).unsqueeze(-1).expand(-1, -1, self.problem_size)
            # all_infsb = ((self.simulated_ninf_flag==float('-inf')).all(dim=-1)).unsqueeze(-1).expand(-1,-1,self.problem_size)
            self.ninf_mask = torch.where(all_infsb, self.visited_ninf_flag, self.ninf_mask)
        if safety_layer and self.selected_count < self.problem_size -1:
            self.ninf_mask = torch.where(simulated_ninf_flag1==float('-inf'), float('-inf'), self.ninf_mask)
            all_infsb = ((self.ninf_mask==float('-inf')).all(dim=-1)).unsqueeze(-1).expand(-1,-1,self.problem_size)
            self.ninf_mask = torch.where(all_infsb, self.visited_ninf_flag, self.ninf_mask)

        # visited == 0 means not visited
        # out_of_tw_ninf_flag == -inf means already can not be visited bacause current_time + travel_time > tw_end
        newly_infeasible = (((self.visited_ninf_flag == 0).int() + (self.out_of_tw_ninf_flag == float('-inf')).int()) == 2).any(dim=2)
        if infsb_level:
            infsb_level_value = (((self.visited_ninf_flag == 0).int() + (self.out_of_tw_ninf_flag == float('-inf')).int()) == 2).sum(dim=2)

        self.infeasible = self.infeasible + newly_infeasible
        # once the infeasibility occurs, no matter which node is selected next, the route has already become infeasible
        self.infeasibility_list = torch.cat((self.infeasibility_list, self.infeasible[:, :, None]), dim=2)
        infeasible = 0.
        # infeasible_rate = self.infeasible.sum() / (self.batch_size * self.pomo_size)
        # print(">> Cause Infeasibility: Inlegal rate: {}".format(infeasible_rate))

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.infeasible = self.infeasible
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        # returning values
        done = self.finished.all()
        if done:
            if not out_reward:
                reward = -self._get_travel_distance()  # note the minus sign!
            else:
                # shape: (batch, pomo)
                dist_reward = -self._get_travel_distance() # note the minus sign
                total_timeout_reward = -self.timeout_list.sum(dim=-1)
                if penalty_normalize:
                    total_timeout_reward = total_timeout_reward / self.node_tw_end[:,:1]
                timeout_nodes_reward = -torch.where(self.timeout_list>0, torch.ones_like(self.timeout_list), self.timeout_list).sum(-1).int()
                reward = [dist_reward, total_timeout_reward, timeout_nodes_reward]
            # not visited but can not reach
            # infeasible_rate = self.infeasible.sum() / (self.batch_size*self.pomo_size)
            infeasible = self.infeasible
            # print(">> Cause Infeasibility: Inlegal rate: {}".format(infeasible_rate))
        else:
            reward = None

        if infsb_level:
            return self.step_state, reward, done, [infeasible, infsb_level_value]

        return self.step_state, reward, done, infeasible

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
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
        data = self.get_random_problems(num_samples, problem_size)
        dataset = [attr.cpu().tolist() for attr in data]
        filedir = os.path.split(path)[0]
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        with open(path, 'wb') as f:
            pickle.dump(list(zip(*dataset)), f, pickle.HIGHEST_PROTOCOL)
        print("Save TSPTW dataset to {}".format(path))

    def load_dataset(self, path, offset=0, num_samples=1000, disable_print=True):
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset+num_samples]
            if not disable_print:
                print(">> Load {} data ({}) from {}".format(len(data), type(data), path))
        node_xy, service_time, tw_start, tw_end = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data]
        node_xy, service_time, tw_start, tw_end = torch.Tensor(node_xy), torch.Tensor(service_time), torch.Tensor(tw_start), torch.Tensor(tw_end)
        # # Normalize as in DPDP (Kool et. al)
        # loc_factor = 100
        # node_xy = node_xy / loc_factor  # Normalize
        # # Normalize same as coordinates to keep unit the same, not that these values do not fall in range [0,1]!
        # # Todo: should we additionally normalize somehow, e.g. by expected makespan/tour length?
        # tw_start = tw_start / loc_factor
        # tw_end = tw_end / loc_factor
        # # Upper bound for depot = max(node ub + dist to depot), to make this tight
        # tw_end[:, 0] = (torch.cdist(node_xy[:, None, 0], node_xy[:, 1:]).squeeze(1) + tw_end[:, 1:]).max(dim=-1)[0]
        # # nodes_timew = nodes_timew / nodes_timew[0, 1]
        data = (node_xy, service_time, tw_start, tw_end)
        return data

    def get_random_problems(self, batch_size, problem_size, normalized=True, coord_factor=100, max_tw_gap=10, max_tw_size=100):

        tw_type = self.tw_type

        if tw_type in ["cappart", "da_silva"]:
            # Taken from DPDP (Kool et. al)
            # Taken from https://github.com/qcappart/hybrid-cp-rl-solver/blob/master/src/problem/tsptw/environment/tsptw.py
            # max_tw_size = 1000 if tw_type == "da_silva" else 100 attention
            max_tw_size = problem_size * 2 if tw_type == "da_silva" else 100
            """
            :param problem_size: number of cities
            :param grid_size (=1): x-pos/y-pos of cities will be in the range [0, grid_size]
            :param max_tw_gap: maximum time windows gap allowed between the cities constituing the feasible tour
            :param max_tw_size: time windows of cities will be in the range [0, max_tw_size]
            :return: a feasible TSPTW instance randomly generated using the parameters
            """
            node_xy = torch.rand(size=(batch_size, problem_size, 2)) * coord_factor  # (batch, problem, 2)
            travel_time = torch.cdist(node_xy, node_xy, p=2, compute_mode='donot_use_mm_for_euclid_dist') / self.speed # (batch, problem, problem)

            # random_solution = torch.arange(1, problem_size).repeat(batch_size, 1)
            # idx = [torch.randperm(random_solution.size(1)) for _ in range(batch_size)]
            # out = torch.zeros(size=(0, problem_size-1)).long()
            # for i in range(batch_size):
            #     out = torch.cat([out, random_solution[i][idx[i]].unsqueeze(0)], dim=0)
            # zeros = torch.zeros(size=(batch_size, 1)).long()
            # random_solution = torch.cat([zeros, out], dim=1)
            random_solution = torch.arange(1, problem_size).repeat(batch_size, 1)
            for i in range(batch_size):
                random_solution[i] = random_solution[i][torch.randperm(random_solution.size(1))]
            zeros = torch.zeros(size=(batch_size, 1)).long()
            random_solution = torch.cat([zeros, random_solution], dim=1)

            time_windows = torch.zeros((batch_size, problem_size, 2))
            time_windows[:, 0, :] = torch.tensor([0, 1000. * coord_factor]).repeat(batch_size, 1)

            total_dist = torch.zeros(batch_size)
            for i in range(1, problem_size):

                prev_city = random_solution[:, i - 1]
                cur_city = random_solution[:, i]

                cur_dist = travel_time[torch.arange(batch_size), prev_city, cur_city]

                tw_lb_min = time_windows[torch.arange(batch_size), prev_city, 0] + cur_dist
                total_dist += cur_dist

                # print(tw_type)
                if tw_type=="da_silva":
                    # Style by Da Silva and Urrutia, 2010, "A VNS Heuristic for TSPTW"
                    rand_tw_lb = torch.rand(batch_size) * (max_tw_size / 2) + (total_dist - max_tw_size / 2)
                    rand_tw_ub = torch.rand(batch_size) * (max_tw_size / 2) + total_dist
                elif tw_type == "cappart":
                    # Cappart et al. style 'propagates' the time windows resulting in little overlap / easier instances
                    rand_tw_lb = torch.rand(batch_size) * (max_tw_gap) + tw_lb_min
                    rand_tw_ub = torch.rand(batch_size) * (max_tw_size) + rand_tw_lb

                time_windows[torch.arange(batch_size), cur_city, :] = torch.cat([rand_tw_lb.unsqueeze(1), rand_tw_ub.unsqueeze(1)], dim=1)
            # import random
            # import numpy as np
            # from scipy.spatial.distance import cdist
            #
            # # Taken from https://github.com/qcappart/hybrid-cp-rl-solver/blob/master/src/problem/tsptw/environment/tsptw.py
            # def generate_random_instance(n_city, grid_size, max_tw_gap, max_tw_size,
            #                              is_integer_instance, seed, fast=True, da_silva_style=False):
            #     """
            #     :param n_city: number of cities
            #     :param grid_size: x-pos/y-pos of cities will be in the range [0, grid_size]
            #     :param max_tw_gap: maximum time windows gap allowed between the cities constituing the feasible tour
            #     :param max_tw_size: time windows of cities will be in the range [0, max_tw_size]
            #     :param is_integer_instance: True if we want the distances and time widows to have integer values
            #     :param seed: seed used for generating the instance. -1 means no seed (instance is random)
            #     :return: a feasible TSPTW instance randomly generated using the parameters
            #     """
            #
            #     rand = random.Random()
            #
            #     if seed != -1:
            #         rand.seed(seed)
            #
            #     x_coord = [rand.uniform(0, grid_size) for _ in range(n_city)]
            #     y_coord = [rand.uniform(0, grid_size) for _ in range(n_city)]
            #     coord = np.array([x_coord, y_coord]).transpose()
            #
            #     if fast:  # Improved code but could (theoretically) give different results with rounding?
            #         travel_time = cdist(coord, coord)
            #         if is_integer_instance:
            #             travel_time = travel_time.round().astype(np.int32)
            #     else:
            #         travel_time = []
            #         for i in range(n_city):
            #
            #             dist = [float(np.sqrt((x_coord[i] - x_coord[j]) ** 2 + (y_coord[i] - y_coord[j]) ** 2))
            #                     for j in range(n_city)]
            #
            #             if is_integer_instance:
            #                 dist = [round(x) for x in dist]
            #
            #             travel_time.append(dist)
            #
            #     random_solution = list(range(1, n_city))
            #     rand.shuffle(random_solution)
            #
            #     random_solution = [0] + random_solution
            #
            #     time_windows = np.zeros((n_city, 2))
            #     time_windows[0, :] = [0, 1000 * grid_size]
            #
            #     total_dist = 0
            #     for i in range(1, n_city):
            #
            #         prev_city = random_solution[i - 1]
            #         cur_city = random_solution[i]
            #
            #         cur_dist = travel_time[prev_city][cur_city]
            #
            #         tw_lb_min = time_windows[prev_city, 0] + cur_dist
            #         total_dist += cur_dist
            #
            #         if da_silva_style:
            #             # Style by Da Silva and Urrutia, 2010, "A VNS Heuristic for TSPTW"
            #             rand_tw_lb = rand.uniform(total_dist - max_tw_size / 2, total_dist)
            #             rand_tw_ub = rand.uniform(total_dist, total_dist + max_tw_size / 2)
            #         else:
            #             # Cappart et al. style 'propagates' the time windows resulting in little overlap / easier instances
            #             rand_tw_lb = rand.uniform(tw_lb_min, tw_lb_min + max_tw_gap)
            #             rand_tw_ub = rand.uniform(rand_tw_lb, rand_tw_lb + max_tw_size)
            #
            #         if is_integer_instance:
            #             rand_tw_lb = np.floor(rand_tw_lb)
            #             rand_tw_ub = np.ceil(rand_tw_ub)
            #
            #         time_windows[cur_city, :] = [rand_tw_lb, rand_tw_ub]
            #
            #     if is_integer_instance:
            #         time_windows = time_windows.astype(np.int32)
            #
            #     # Don't store travel time since it takes up much
            #     return coord[0], coord[1:], time_windows, grid_size
            #
            # depot, node, time_windows, _ = generate_random_instance(n_city=problem_size, grid_size=100, max_tw_gap=10, max_tw_size=max_tw_size,
            #                                 is_integer_instance=True, seed=1234, da_silva_style=tw_type == 'da_silva')
            # depot = torch.tensor(depot).unsqueeze(0)
            # node = torch.tensor(node)
            # node_xy = torch.cat([depot, node], dim=0).unsqueeze(0)
            # time_windows = torch.tensor(time_windows).unsqueeze(0)
        elif tw_type == "zhang":
            TSPTW_SET = namedtuple("TSPTW_SET",
                                   ["node_loc",  # Node locations 1
                                    "node_tw",  # node time windows 5
                                    "durations",  # service duration per node 6
                                    "service_window",  # maximum of time units 7
                                    "time_factor", "loc_factor"])
            # tw = generate_tsptw_data(size=batch_size, graph_size=problem_size, time_factor=1.42,
            #                          tw_type="naive/hard", tw_duration=self.tw_duration) # 1.42 = sqrt()
            tw = generate_tsptw_data(size=batch_size, graph_size=problem_size, time_factor=problem_size*55, tw_type="naive/hard", tw_duration=self.tw_duration)
            node_xy = torch.tensor(tw.node_loc).float()
            time_windows = torch.tensor(tw.node_tw)
        elif tw_type == "random":
            node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)
            service_window = int(1.42 * problem_size)
            tw_start = torch.rand(size=(batch_size, problem_size, 1)) * (service_window/2)
            episilon = ((torch.rand(size=(batch_size, problem_size, 1)) * 0.8) + 0.1) * (service_window/2) #[0.1,0.9]
            tw_end = tw_start + episilon
            # Normalize as in DPDP (Kool et. al)
            # Upper bound for depot = max(node ub + dist to depot), to make this tight
            tw_start[:, 0, 0] = 0.
            tw_end[:, 0, 0] = (torch.cdist(node_xy[:, None, 0], node_xy[:, 1:]).squeeze(1) + tw_end[:, 1:, 0]).max(dim=1)[0]
            time_windows = torch.cat([tw_start, tw_end], dim=-1)
        else:
            raise NotImplementedError

        service_time = torch.zeros(size=(batch_size,problem_size))
        # Don't store travel time since it takes up much
        return node_xy, service_time, time_windows[:,:,0], time_windows[:,:,1]

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
        raise NotImplementedError #TODO: implement

    def preprocessing(self, rec):
        batch_size, seq_length = rec.size()
        k = batch_size // self.node_xy.size(0)
        arange = torch.arange(batch_size)

        pre = torch.zeros(batch_size).long()
        visited_time = torch.zeros((batch_size, seq_length)).long()
        arrival_time = torch.zeros((batch_size, seq_length))
        last_arrival_time = torch.zeros((batch_size, seq_length))
        current_time = torch.zeros((batch_size, ))
        coor = self.node_xy.repeat_interleave(k, 0)
        tw_end = self.node_tw_end.repeat_interleave(k, 0)
        tw_start = self.node_tw_start.repeat_interleave(k, 0)
        # timestamps = torch.gather(self.timestamps, 1, select_idx.unsqueeze(-1).expand(-1, -1, self.problem_size)).reshape( (batch_size, seq_length))
        for i in range(seq_length):
            next_ = rec[arange, pre]
            visited_time[arange, next_] = (i + 1) % seq_length
            last_arrival_time[arange, next_] = current_time.clone()
            travel_time = (coor[arange, pre] - coor[arange, next_]).norm(p=2, dim=1)
            current_time = torch.max(current_time+travel_time, tw_start[arange, next_])
            arrival_time[arange, next_] = current_time.clone()
            # if i != seq_length - 1:  arrival_time1[arange, next_] = timestamps[arange, i + 1]
            # last_arrival_time1[arange, next_] = timestamps[arange, i].clone()
            pre = next_.clone()
       # shape: (batch*k, problem_size)
        last_arrival_time[:, 0] = 0.
        arrival_time[:, 0] = 0.
        # check by: self.timestamps.squeeze(1) == arrival_time.sort()[0]

        return (visited_time, arrival_time, last_arrival_time, tw_start, tw_end)

    def check_feasibility(self, select_idx):
        raise NotImplementedError  # TODO: implement
        # assert (self.visited_ninf_flag == float('-inf')).all(), "not visiting all nodes!"
        # assert torch.gather(~self.infeasible, 1, select_idx).all(), "not valid tour!"

    def get_costs(self, rec, get_context=False, check_full_feasibility=False, out_reward=False, penalty_factor=1.0, penalty_normalize=False):

        k = rec.size(0) // self.node_xy.size(0)
        # check full feasibility if needed
        if get_context:
            context = self.preprocessing(rec)
        if check_full_feasibility:
            self.check_feasibility()

        coor = self.node_xy.repeat_interleave(k, 0)
        coor_next = coor.gather(1, rec.long().unsqueeze(-1).expand(*rec.size(), 2))
        cost = (coor - coor_next).norm(p=2, dim=2).sum(1)

        # arrival_time - tw_end
        exceed_time_window = torch.clamp_min(context[1] - context[-1], 0.0)
        out_node_penalty = (exceed_time_window > 1e-5).sum(-1)
        out_penalty = exceed_time_window.sum(-1)
        if penalty_normalize:
            out_penalty = out_penalty / context[-1][:, 0]
        if out_reward:
            cost = cost + penalty_factor * (out_node_penalty + out_penalty)

        # get context
        if get_context:
            return cost, context, out_penalty, out_node_penalty
        else:
            return cost, out_penalty, out_node_penalty

    def get_dynamic_feature(self, context, with_infsb_feature, tw_normalize=False):
        visited_time, arrival_time, last_arrival_time, tw_start, tw_end = context
        if tw_normalize:
            tw_end_max = tw_end[:, :1].clone()
            last_arrival_time = last_arrival_time / tw_end_max
            arrival_time = arrival_time / tw_end_max
            tw_start = tw_start / tw_end_max
            tw_end = tw_end / tw_end_max

        batch_size, seq_length = arrival_time.size()
        k = batch_size // self.node_xy.size(0)
        is_depot = torch.tensor([1.]+[0.]*(seq_length-1))[None,:].repeat_interleave(batch_size,0)
        exceed_time_window = torch.clamp_min(arrival_time - tw_end, 0.0)
        infeasibility_indicator_after_visit = exceed_time_window > 0

        to_actor = torch.cat((
            arrival_time.unsqueeze(-1),
            exceed_time_window.unsqueeze(-1),
            is_depot.unsqueeze(-1),
            last_arrival_time.unsqueeze(-1),
            # tw_start.unsqueeze(-1),
            infeasibility_indicator_after_visit.unsqueeze(-1),
        ), -1)  # the node features

        feature = torch.cat([self.node_xy.repeat_interleave(k, 0), tw_start.unsqueeze(-1), tw_end.unsqueeze(-1)], dim=-1)
        supplement_feature = to_actor
        if not with_infsb_feature:
            supplement_feature = to_actor[:, :, :-1]
        feature = torch.cat((feature, supplement_feature), dim=-1)

        return visited_time, None, feature

    def improvement_step(self, rec, action, obj, feasible_history, t, weights=0, out_reward = False, penalty_factor=1., penalty_normalize=False, improvement_method = "kopt", insert_before=True):

        _, total_history = feasible_history.size()
        pre_bsf = obj[:, 1:].clone()  # batch_size, 3 (current, bsf, tsp_bsf)
        feasible_history = feasible_history.clone()  # bs, total_history

        # improvement
        if improvement_method == "kopt":
            next_state = self.k_opt(rec, action)
        elif improvement_method == "rm_n_insert":
            next_state = self.rm_n_insert(rec, action, insert_before=insert_before)
        else:
            raise NotImplementedError()
        next_obj, context, out_penalty, out_node_penalty = self.get_costs(next_state, get_context=True, out_reward=out_reward, penalty_factor=penalty_factor, penalty_normalize=penalty_normalize)

        # MDP step
        non_feasible_cost_total = torch.clamp_min(context[1] - context[-1], 0.0).sum(-1)
        feasible = non_feasible_cost_total <= 0.0
        soft_infeasible = (non_feasible_cost_total <= self.epsilon) & (non_feasible_cost_total > 0.)

        now_obj = pre_bsf.clone()
        if not out_reward:
            # only update feasible obj
            now_obj[feasible, 0] = next_obj[feasible].clone()
        else:
            # update all obj, obj = cost + penalty
            now_obj[:, 0] = next_obj.clone()
        # only update epsilon feasible obj
        now_obj[soft_infeasible, 1] = next_obj[soft_infeasible].clone()
        now_bsf = torch.min(pre_bsf, now_obj)
        rewards = (pre_bsf - now_bsf)  # bs,2 (feasible_reward, epsilon-feasible_reward)

        # feasible history step
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

        # update regulation
        reg = self.f(f_to_f_2) + self.f(if_to_if_2)

        reward = torch.cat((rewards[:, :1],  # reward
                            -1 * reg * weights * 0.05 * self.env_params["with_regular"],  # regulation, alpha = 0.05
                            rewards[:, 1:2] * 0.05 * self.env_params["with_bonus"],  # bonus, beta = 0.05
                            ), -1)

        out = (next_state,
               reward,
               torch.cat((next_obj[:, None], now_bsf), -1),
               feasible_history,
               context,
               context2,
               (if_to_if, if_to_f, f_to_if, f_to_f, if_to_if_2, if_to_f_2, f_to_if_2, f_to_f_2),
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


