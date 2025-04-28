from dataclasses import dataclass
import torch
import os, pickle
import numpy as np
from utils import *
__all__ = ['TSPDLEnv']
EPSILON = {
    20: 0.33,
    50: 0.625,
    100: 1.0,
    200: 1.429
}
EPSILON_hardcoded = 0.625

@dataclass
class Reset_State:
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None 
    node_draft_limit: torch.Tensor = None # draft limit
    # shape: (batch, problem)
    prob_emb: torch.Tensor = None
    # shape: (num_training_prob)


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
    # shape: (batch, pomo)
    infeasible: torch.Tensor = None
    # shape: (batch, pomo)
    load: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    current_coord: torch.Tensor = None
    # shape: (batch, pomo, 2)


class TSPDLEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.problem = "TSPDL"
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.dl_percent = env_params['dl_percent']
        self.loc_scaler = env_params['loc_scaler'] if 'loc_scaler' in env_params.keys() else None

        self.epsilon = EPSILON[self.problem_size]
        self.k_max = self.env_params['k_max'] if 'k_max' in env_params.keys() else None
        if 'pomo_start' in env_params.keys():
            self.pomo_size = env_params['pomo_size'] if env_params['pomo_start'] else env_params['train_z_sample_size']

        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in env_params.keys() else env_params['device']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.START_NODE = None
        # IDX.shape: (batch, pomo)
        self.node_xy = None
        self.lib_node_xy = None
        # if self.env_params['original_lib_xy'] is not None:
        #     self.lib_node_xy  = self.env_params['original_lib_xy']
        # shape: (batch, problem, 2)
        self.node_demand = None
        self.node_draft_limit = None
        # shape: (batch, problem)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        self.infeasibility_list = None
        self.out_of_draft_limit_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        self.simulated_ninf_flag = None
        self.global_mask = None
        self.global_mask_ninf_flag = None
        self.out_of_dl_ninf_flag = None
        # shape: (batch, pomo, problem)
        self.ninf_mask = None
        # shape: (batch, pomo, problem)
        self.finished = None
        self.infeasible = None
        # shape: (batch, pomo)
        self.length = None
        # shape: (batch, pomo)
        self.current_coord = None
        # shape: (batch, pomo, 2)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size, rollout_size, problems=None, aug_factor=1, normalize=False):
        self.pomo_size = rollout_size
        if problems is not None:
            node_xy, node_demand, node_draft_limit = problems
        else:
            node_xy, node_demand, node_draft_limit = self.get_random_problems(batch_size, self.problem_size, normalized=normalize)
        self.batch_size = node_xy.size(0)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                node_xy = self.augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                node_draft_limit = node_draft_limit.repeat(8, 1)
            else:
                raise NotImplementedError
        # print(node_xy.size())
        self.node_xy = node_xy
        if self.lib_node_xy is not None:
            self.lib_node_xy = self.lib_node_xy.repeat(8,1,1)
            # shape: (8*batch, N, 2)
        # shape: (batch, problem, 2)
        self.node_demand = node_demand
        self.node_draft_limit = node_draft_limit
        # shape: (batch, problem)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)

        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_draft_limit = node_draft_limit

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.START_NODE = torch.arange(start=1, end=self.pomo_size+1)[None, :].expand(self.batch_size, -1).to(self.device)
        self.step_state.PROBLEM = self.problem

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        self.out_of_draft_limit_list = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)
        self.infeasibility_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.bool).to(self.device) # True for causing infeasibility
        # shape: (batch, pomo, 0~)

        self.load = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.simulated_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.global_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.global_mask_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.out_of_dl_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # shape: (batch, pomo, problem)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # shape: (batch, pomo, problem)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        self.infeasible = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
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
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.infeasible = self.infeasible
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected, visit_mask_only =True, out_reward = False, generate_PI_mask=False, use_predicted_PI_mask=False, pip_step=1,
             soft_constrained = False, backhaul_mask = None, penalty_normalize=False):
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
        demand_list = self.node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load += selected_demand

        current_coord = self.node_xy[torch.arange(self.batch_size)[:, None], selected]
        # shape: (batch, pomo, 2)
        new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)
        # shape: (batch, pomo)
        self.length = self.length + new_length
        self.current_coord = current_coord

        # Mask
        ####################################
        # visit constraint
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem)

        # draft limit constraint
        round_error_epsilon = 0.00001
        dl_list = self.node_draft_limit[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem)

        # simulate the right infsb mask and see ATTENTION!
        if generate_PI_mask and self.selected_count < self.problem_size -1:
            self._calculate_PIP_mask(pip_step)

        # (current load + demand of next node > draft limit of next node) means infeasible
        out_of_dl = self.load[:, :, None] + demand_list > dl_list + round_error_epsilon
        # shape: (batch, pomo, problem)
        self.out_of_dl_ninf_flag[out_of_dl] = float('-inf')
        # shape: (batch, pomo, problem)
        # value that exceeds draft limit of the selected node = current load - node_draft_limit
        total_out_of_dl = self.load - self.node_draft_limit[torch.arange(self.batch_size)[:, None], selected]
        # negative value means current load < node_draft_limit, turn it into 0
        total_out_of_dl = torch.where(total_out_of_dl<0, torch.zeros_like(total_out_of_dl), total_out_of_dl)
        # shape: (batch, pomo)
        self.out_of_draft_limit_list = torch.cat((self.out_of_draft_limit_list, total_out_of_dl[:, :, None]), dim=2)

        self.ninf_mask = self.visited_ninf_flag.clone()
        if not visit_mask_only or not soft_constrained:
            self.ninf_mask[out_of_dl] = float('-inf')
        if generate_PI_mask and self.selected_count < self.problem_size -1 and (not use_predicted_PI_mask):
            self.ninf_mask = torch.where(self.simulated_ninf_flag==float('-inf'), float('-inf'), self.ninf_mask)
            all_infsb = ((self.ninf_mask==float('-inf')).all(dim=-1)).unsqueeze(-1).expand(-1,-1,self.problem_size)
            self.ninf_mask = torch.where(all_infsb, self.visited_ninf_flag, self.ninf_mask)

        # visited == 0 means not visited
        # out_of_dl_ninf_flag == -inf means already can not be visited bacause current_load + node_demand > node_draft_limit
        newly_infeasible = (((self.visited_ninf_flag == 0).int() + (self.out_of_dl_ninf_flag == float('-inf')).int()) == 2).any(dim=2)
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
        self.step_state.load = self.load
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        # returning values
        done = self.finished.all()
        if done:
            if not out_reward:
                reward = -self._get_travel_distance()  # note the minus sign!
            else:
                # shape: (batch, pomo)
                dist_reward = -self._get_travel_distance()  # note the minus sign
                total_out_of_dl_reward = - self.out_of_draft_limit_list.sum(dim=-1)
                out_of_dl_nodes_reward = - torch.where(self.out_of_draft_limit_list > round_error_epsilon, torch.ones_like(self.out_of_draft_limit_list), self.out_of_draft_limit_list).sum(-1).int()
                reward = [dist_reward, total_out_of_dl_reward, out_of_dl_nodes_reward]
                # not visited but can not reach
                # infeasible_rate = self.infeasible.sum() / (self.batch_size*self.pomo_size)
            infeasible = self.infeasible
            # print(">> Cause Infeasibility: Inlegal rate: {}".format(infeasible_rate))
        else:
            reward = None

        return self.step_state, reward, done, infeasible

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        if self.lib_node_xy is not None:
            all_xy = self.lib_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        else:
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
        data = self.get_random_problems(num_samples, problem_size, normalized=False)
        dataset = [attr.cpu().tolist() for attr in data]
        filedir = os.path.split(path)[0]
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        with open(path, 'wb') as f:
            pickle.dump(list(zip(*dataset)), f, pickle.HIGHEST_PROTOCOL)
        print("Save TSPDL dataset to {}".format(path))

    def load_dataset(self, path, offset=0, num_samples=1000, disable_print=True):
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset+num_samples]
            if not disable_print:
                print(">> Load {} data ({}) from {}".format(len(data), type(data), path))
        node_xy, node_demand, node_draft_limit = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data]
        node_xy, node_demand, node_draft_limit = torch.Tensor(node_xy), torch.Tensor(node_demand), torch.Tensor(node_draft_limit)
        # scale it to [0,1]
        demand_sum = node_demand.sum(-1).view(-1, 1)
        node_demand = node_demand / demand_sum
        node_draft_limit = node_draft_limit / demand_sum
        data = (node_xy, node_demand, node_draft_limit)
        return data

    def get_random_problems(self, batch_size, problem_size, normalized=True):
        dl_percent =self.dl_percent
        node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)
        node_demand = torch.cat([torch.zeros((batch_size, 1)), torch.ones((batch_size, problem_size - 1))], dim=1)
        # (batch, problem) 0,1,1,1,1,....
        demand_sum = node_demand.sum(dim=1).unsqueeze(1)
        # currently, demand_sum == problem_size-1; if not, the program needs to be revised (todo)
        node_draft_limit = torch.ones((batch_size, problem_size)) * demand_sum
        for i in range(batch_size):
            # randomly choose half of the nodes (except depot) to lower their draft limit (range: [1, demand_sum))
            lower_dl_idx = np.random.choice(range(1, problem_size), size=problem_size * dl_percent // 100, replace=False)
            feasible_dl = False
            while not feasible_dl:
                lower_dl = torch.randint(1, demand_sum[i].int().item(), size=(problem_size * dl_percent // 100,))
                cnt = torch.bincount(lower_dl)
                cnt_cumsum = torch.cumsum(cnt, dim=0)
                feasible_dl = (cnt_cumsum <= torch.arange(0, cnt.size(0))).all()
            node_draft_limit[i, lower_dl_idx] = lower_dl.float()
        if normalized:
            node_demand = node_demand / demand_sum
            node_draft_limit = node_draft_limit / demand_sum

        return node_xy, node_demand, node_draft_limit

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

    def _calculate_PIP_mask(self, pip_step):
        '''
        copy from https://github.com/jieyibi/PIP-constraint/blob/main/POMO%2BPIP/envs/TSPTWEnv.py [NeurIPS'24]
        '''
        round_error_epsilon = 0.00001
        demand_list = self.node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1) # shape: (batch, pomo, problem)
        dl_list = self.node_draft_limit[:, None, :].expand(self.batch_size, self.pomo_size, -1) # shape: (batch, pomo, problem)

        if pip_step == 0:
            out_of_dl = self.load[:, :, None] + demand_list > dl_list + round_error_epsilon
            self.simulated_ninf_flag = torch.zeros((self.batch_size, self.pomo_size, self.problem_size)) # shape: (batch, pomo, problem)
            self.simulated_ninf_flag[out_of_dl] = float('-inf')
        elif pip_step == 1:
            unvisited = torch.masked_select(
                torch.arange(self.problem_size).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.pomo_size, self.problem_size),
                self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
            two_step_unvisited = unvisited.unsqueeze(2).repeat(1, 1, self.problem_size - self.selected_count, 1)
            diag_element = torch.diag_embed(torch.diagonal(two_step_unvisited, dim1=-2, dim2=-1))
            two_step_idx = torch.masked_select(two_step_unvisited, diag_element == 0).reshape(self.batch_size, self.pomo_size, self.problem_size - self.selected_count,-1)

            two_step_dl = torch.masked_select(dl_list, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
            two_step_dl = two_step_dl.unsqueeze(2).repeat(1, 1, self.problem_size - self.selected_count, 1)
            two_step_dl = torch.masked_select(two_step_dl, diag_element == 0).reshape(self.batch_size, self.pomo_size, self.problem_size - self.selected_count, -1)

            current_load = self.load.unsqueeze(-1).repeat(1, 1, self.problem_size - self.selected_count)
            # add demand of the first-step nodes
            first_step_demand = torch.masked_select(demand_list, self.visited_ninf_flag != float('-inf')).reshape(
                self.batch_size, self.pomo_size, -1)
            current_load += first_step_demand
            current_load = current_load.unsqueeze(-1).repeat(1, 1, 1, two_step_dl.size(-1))
            # add demand of the second-step nodes
            second_step_demand = first_step_demand.unsqueeze(2).repeat(1, 1, self.problem_size - self.selected_count, 1)
            second_step_demand = torch.masked_select(second_step_demand, diag_element == 0).reshape(self.batch_size, self.pomo_size, self.problem_size - self.selected_count,-1)
            current_load += second_step_demand

            # feasibility judgement
            infeasible_mark = (current_load > two_step_dl + round_error_epsilon)
            selectable = (infeasible_mark == False).all(dim=-1)
            self.global_mask = infeasible_mark.sum(-1) / infeasible_mark.size(-1)
            self.global_mask_ninf_flag = torch.full((self.batch_size, self.pomo_size, self.problem_size), float('-inf'))
            self.global_mask_ninf_flag.masked_scatter_(self.visited_ninf_flag == 0, self.global_mask)

            self.simulated_ninf_flag = torch.full((self.batch_size, self.pomo_size, self.problem_size), float('-inf'))
            selected_indices = selectable.nonzero(as_tuple=False)
            unvisited_indices = unvisited[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]]
            self.simulated_ninf_flag[selected_indices[:, 0], selected_indices[:, 1], unvisited_indices] = 0.
        else:
            raise NotImplementedError


    def get_initial_solutions(self, strategy, k, max_dummy_size=0):
        batch_size, problem_size, _ = self.node_xy.size()
        if strategy == "random": # not guarantee feasibility (may exceed tw)
            # start from 0
            B_k = batch_size * k
            # # random solution permutation
            customer = torch.rand(B_k, problem_size-1).argsort(dim=1) + 1
            solutions = torch.cat([torch.zeros((B_k, 1), dtype=torch.long), customer], dim=1)
            # judge feasibility
            context = self.preprocessing(sol2rec(solutions.unsqueeze(1)).squeeze(1))
            non_feasible_cost_total = torch.clamp_min(context[1] - context[-1], 0.0).sum(-1)
            self.infeasible = (non_feasible_cost_total > 0.0).view(batch_size, k)
            solutions = solutions.view(batch_size, k, -1)
        else:
            raise NotImplementedError()

        self.selected_node_list = solutions
        # shape: (batch, k, solution)

        return self._get_travel_distance()

    def preprocessing(self, rec):
        batch_size, seq_length = rec.size()
        k = batch_size // self.node_xy.size(0)
        arange = torch.arange(batch_size)

        pre = torch.zeros(batch_size).long()
        visited_time = torch.zeros((batch_size, seq_length)).long()
        current_load = torch.zeros((batch_size, ))
        after_load = torch.zeros((batch_size, seq_length))
        last_load = torch.zeros((batch_size, seq_length))
        node_demand = self.node_demand.repeat_interleave(k, 0)
        node_draft_limit = self.node_draft_limit.repeat_interleave(k, 0)
        for i in range(seq_length):
            next_ = rec[arange, pre]
            visited_time[arange, next_] = (i + 1) % seq_length
            last_load[arange, next_] = current_load.clone()
            current_load = current_load + node_demand[arange, next_]
            after_load[arange, next_] = current_load.clone()
            pre = next_.clone()
       # shape: (batch*k, problem_size)
        last_load[:, 0] = 0.
        after_load[:, 0] = 0.
        # check by: self.timestamps.squeeze(1) == arrival_time.sort()[0]

        return (visited_time, after_load, last_load, node_demand, node_draft_limit)

    def check_feasibility(self, select_idx=None):
        raise NotImplementedError  # TODO: implement
        # assert (self.visited_ninf_flag == float('-inf')).all(), "not visiting all nodes!"
        # assert torch.gather(~self.infeasible, 1, select_idx).all(), "not valid tour!"

    def get_costs(self, rec, get_context=False, check_full_feasibility=False, out_reward=False, penalty_factor=1.0, penalty_normalize=False, seperate_obj_penalty=False, non_linear=None, wo_node_penalty=False, wo_tour_penalty =False):

        k = rec.size(0) // self.node_xy.size(0)
        # check full feasibility if needed
        if get_context:
            context = self.preprocessing(rec)
        if check_full_feasibility:
            self.check_feasibility()

        coor = self.node_xy.repeat_interleave(k, 0)
        coor_next = coor.gather(1, rec.long().unsqueeze(-1).expand(*rec.size(), 2))
        cost = (coor - coor_next).norm(p=2, dim=2).sum(1)

        # visited_time, after_load, last_load, node_demand, node_draft_limit
        # after_load - dl
        exceed_dl = torch.clamp_min(context[1] - context[-1], 0.0)
        out_node_penalty = (exceed_dl > 1e-5).sum(-1)
        out_penalty = exceed_dl.sum(-1)
        if penalty_normalize:
            out_penalty = out_penalty / context[-1][:, 0]
        if out_reward:
            if wo_node_penalty:
                cost = cost + penalty_factor * (out_penalty)
            elif wo_tour_penalty:
                cost = cost + penalty_factor * (out_node_penalty)
            else:
                cost = cost + penalty_factor * (out_node_penalty + out_penalty)

        # get context
        if get_context:
            return cost, context, out_penalty.unsqueeze(0), out_node_penalty.unsqueeze(0)
        else:
            return cost, out_penalty.unsqueeze(0), out_node_penalty.unsqueeze(0)


    def get_dynamic_feature(self, context, with_infsb_feature, tw_normalize=False):
        visited_time, after_load, last_load, node_demand, node_draft_limit = context

        batch_size, seq_length = after_load.size()
        k = batch_size // self.node_xy.size(0)
        is_depot = torch.tensor([1.]+[0.]*(seq_length-1))[None,:].repeat_interleave(batch_size,0)
        exceed_dl = torch.clamp_min(after_load - node_draft_limit, 0.0)
        infeasibility_indicator_after_visit = exceed_dl > 0

        to_actor = torch.cat((
            after_load.unsqueeze(-1),
            exceed_dl.unsqueeze(-1),
            is_depot.unsqueeze(-1),
            last_load.unsqueeze(-1),
            # tw_start.unsqueeze(-1),
            infeasibility_indicator_after_visit.unsqueeze(-1),
        ), -1)  # the node features

        feature = torch.cat([self.node_xy.repeat_interleave(k, 0), node_demand.unsqueeze(-1), node_draft_limit.unsqueeze(-1)], dim=-1)
        supplement_feature = to_actor
        if not with_infsb_feature:
            supplement_feature = to_actor[:, :, :-1]
        feature = torch.cat((feature, supplement_feature), dim=-1)

        return visited_time, None, feature

    def improvement_step(self, rec, action, obj, feasible_history, t, weights=0, out_reward = False, penalty_factor=1., penalty_normalize=False, improvement_method = "kopt", insert_before=True, epsilon=EPSILON_hardcoded, seperate_obj_penalty=False, non_linear=None, n2s_decoder=False):

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
        soft_infeasible = (non_feasible_cost_total <= epsilon) & (non_feasible_cost_total > 0.)

        now_obj = pre_bsf.clone()
        if not out_reward:
            # only update feasible obj
            now_obj[feasible, 0] = next_obj[feasible].clone()
        else:
            # update all obj, obj = cost + penalty
            if non_linear is None:
                now_obj[:, 0] = next_obj.clone()
            elif non_linear in ["fixed_epsilon", "decayed_epsilon"]: # only have reward when penalty <= epsilon
                now_obj[soft_infeasible, 0] = next_obj[soft_infeasible].clone()
            else:
                raise NotImplementedError
        # only update epsilon feasible obj
        now_obj[soft_infeasible, 1] = next_obj[soft_infeasible].clone()
        now_bsf = torch.min(pre_bsf, now_obj)
        rewards = (pre_bsf - now_bsf)  # bs,2 (feasible_reward, epsilon-feasible_reward)

        # feasible history step
        if n2s_decoder: # calculate the removal record
            # todo: not carefully check yet but probably correct
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

        out = (next_state,
               reward,
               torch.cat((next_obj[:, None], now_bsf), -1),
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








