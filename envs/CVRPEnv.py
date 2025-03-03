from dataclasses import dataclass
import torch
import os, pickle
import numpy as np
from utils import *

__all__ = ['CVRPEnv']

EPSILON = {
    20: 0.33,
    50: 0.625,
    100: 1.0,
    200: 1.429
}
EPSILON_hardcoded = 0.625

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
    # shape: (batch, pomo)
    load: torch.Tensor = None
    # shape: (batch, pomo)
    # shape: (batch, pomo)
    current_time: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    open: torch.Tensor = None
    # shape: (batch, pomo)
    current_coord: torch.Tensor = None
    # shape: (batch, pomo, 2)

    # improvement
    visited_time: torch.Tensor = None
    # shape: (batch *pomo, solution)

class CVRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.problem = "CVRP"
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.epsilon = EPSILON[self.problem_size]
        self.k_max = self.env_params['k_max']
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
        # shape: (batch, 1)

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
            data = problems
        else:
            data = self.get_random_problems(batch_size, self.problem_size, normalized=True)

        depot_xy, node_xy, node_demand = data

        depot_xy = depot_xy.to(self.device)
        node_xy = node_xy.to(self.device)
        node_demand = node_demand.to(self.device)

        self.batch_size = depot_xy.size(0)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = self.augment_xy_data_by_8_fold(depot_xy)
                node_xy = self.augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1)).to(self.device)
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_service_time = torch.zeros(self.batch_size, self.problem_size).to(self.device)
        self.reset_state.node_tw_start = torch.zeros(self.batch_size, self.problem_size).to(self.device)
        self.reset_state.node_tw_end = torch.zeros(self.batch_size, self.problem_size).to(self.device)
        self.reset_state.prob_emb = torch.FloatTensor([1, 0, 0, 0, 0]).unsqueeze(0).to(self.device)  # bit vector for [C, O, B, L, TW]

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.open = torch.zeros(self.batch_size, self.pomo_size).to(self.device)
        self.step_state.START_NODE = torch.arange(start=1, end=self.pomo_size+1)[None, :].expand(self.batch_size, -1).to(self.device)
        self.step_state.PROBLEM = self.problem

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        self.load_list = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)
        self.out_of_capacity_list = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1)).to(self.device)
        self.demand_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1)).to(self.device)
        self.simulated_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1)).to(self.device)
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1)).to(self.device)
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
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

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected, out_reward = False, soft_constrained = False, backhaul_mask = None, penalty_normalize=True):
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
        self.load_list = torch.cat((self.load_list, self.load[:, :, None]), dim=2)

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

        # capacity constraint
        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        if not soft_constrained:
            self.ninf_mask[demand_too_large] = float('-inf')
        self.demand_ninf_flag[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        # calculate the penalty
        # out of capacity: out_of_capacity value = -remaining capacity (i.e., remain negative capacity means already exceeding, positive means real remaining)
        out_of_capacity = torch.clamp(- self.load - round_error_epsilon, min=0)
        if not soft_constrained:
            if (out_of_capacity!=0).any():
                print("out of capacity")
        self.out_of_capacity_list = torch.cat((self.out_of_capacity_list, out_of_capacity[:, :, None]), dim=2)


        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        infeasible = 0.0
        # returning values
        done = self.finished.all()
        if done:
            self.dummy_size = self.selected_node_list.size(-1) - self.problem_size
            reward = -self._get_travel_distance()  # note the minus sign!
            total_out_of_capacity_reward = -self.out_of_capacity_list.sum(dim=-1)
            out_of_capacity_nodes_reward = -torch.where(self.out_of_capacity_list > 0, torch.ones_like(self.out_of_capacity_list), self.out_of_capacity_list).sum(-1).int()
            infeasible = (out_of_capacity_nodes_reward != 0.)
            self.infeasible = infeasible
            # shape: (batch, pomo)
            if out_reward:
                reward = [reward, total_out_of_capacity_reward, out_of_capacity_nodes_reward]
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
        print("Save CVRP-LV dataset to {}".format(path))

    def load_dataset(self, path, offset=0, num_samples=1000, disable_print=True):
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset+num_samples]
            if not disable_print:
                print(">> Load {} data ({}) from {}".format(len(data), type(data), path))
        depot_xy, node_xy, node_demand, capacity = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data]
        depot_xy, node_xy, node_demand, capacity = torch.Tensor(depot_xy), torch.Tensor(node_xy), torch.Tensor(node_demand), torch.Tensor(capacity)
        node_demand = node_demand / capacity.view(-1, 1)
        data = (depot_xy, node_xy, node_demand)
        return data

    def get_random_problems(self, batch_size, problem_size, normalized=True):
        depot_xy = torch.rand(size=(batch_size, 1, 2))  # (batch, 1, 2)
        node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)

        if problem_size == 5:
            demand_scaler = 10
        elif problem_size == 20:
            demand_scaler = 30
        elif problem_size == 50:
            demand_scaler = 40
        elif problem_size == 100:
            demand_scaler = 50
        elif problem_size == 200:
            demand_scaler = 70
        else:
            raise NotImplementedError

        if normalized:
            node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)  # (batch, problem)
            return depot_xy, node_xy, node_demand
        else:
            node_demand = torch.Tensor(np.random.randint(1, 10, size=(batch_size, problem_size)))  # (unnormalized) shape: (batch, problem)
            capacity = torch.Tensor(np.full(batch_size, demand_scaler))
            return depot_xy, node_xy, node_demand, capacity


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
        batch_size, problem_size = self.depot_node_demand.size()
        problem_size -= 1 # remove depot
        max_path_length = max_dummy_size + problem_size
        depot_node_xy_expanded = self.depot_node_xy.unsqueeze(1).repeat(1, k, 1, 1)  # B*k*(N+1)*2
        depot_node_demand_expanded = self.depot_node_demand.unsqueeze(1).repeat(1, k, 1)  # B*k*(N+1)
        self.dummy_size = max_dummy_size
        if strategy == "random": # not guarantee feasibility (may exceed capacity)
            B_k = batch_size * k
            # # random solution permutation
            initial_zeros = torch.zeros((B_k, max_path_length - problem_size - 1), dtype=torch.long)
            initial_customers = torch.arange(1, problem_size + 1).repeat(B_k, 1)
            solutions = torch.cat([initial_customers, initial_zeros], dim=1)
            # shuffle solution
            middle_indices = torch.rand(B_k, max_path_length - 2).argsort(dim=1)
            middle_part = solutions[:, :-1].gather(1, middle_indices)
            solutions = torch.cat([torch.zeros((B_k, 1), dtype=torch.long), middle_part, torch.zeros((B_k, 1), dtype=torch.long)], dim=1)
            # judge feasibility
            solutions_rec = get_solution_with_dummy_depot(solutions.unsqueeze(1), problem_size)
            context = self.preprocessing(sol2rec(solutions_rec).squeeze(1), dummify(depot_node_demand_expanded.view(B_k,-1), max_dummy_size-1))
            non_feasible_cost_total = torch.clamp_min(context[-1] - 1.00001, 0.0).sum(-1)
            self.infeasible = (non_feasible_cost_total > 0.0).view(batch_size, k)
            solutions = solutions.view(batch_size, k, -1)
            # customer_nodes = torch.arange(1, problem_size + 1)
            # random_paths = torch.stack([customer_nodes[torch.randperm(problem_size)] for _ in range(B_k)])
            # # initialize the solution
            # current_solutions = torch.zeros((B_k, max_path_length), dtype=torch.long)
            # current_solutions[:, 0] = 0  # 起点为depot
            # current_lengths = torch.ones(B_k, dtype=torch.long)
            # current_demands = torch.zeros(B_k)
            # cum_demand = torch.zeros((batch_size, k, max_path_length), dtype=torch.float)
            # # generate solutions
            # for n in range(problem_size):
            #     # select next nodes
            #     next_nodes = random_paths[:, n]
            #     # update route
            #     current_solutions[torch.arange(B_k), current_lengths] = next_nodes
            #     current_demands += depot_node_demand_expanded.view(-1, problem_size + 1)[torch.arange(B_k), next_nodes]
            #     cum_demand.view(-1, max_path_length)[torch.arange(B_k), current_lengths] = current_demands
            #     current_lengths += 1
            #     # return depot if exceeds the capacity
            #     over_capacity = current_demands > 1.0
            #     if over_capacity.any():
            #         current_solutions[over_capacity, current_lengths[over_capacity]] = 0
            #         current_demands[over_capacity] = 0
            #         current_lengths[over_capacity] += 1
            #
            # solutions = current_solutions.view(batch_size, k, max_path_length)

        elif strategy == "random_feasible":
            depot_node_demand_expanded = dummify(depot_node_demand_expanded, max_dummy_size-1)
            B_k = batch_size * k
            candidates = torch.ones(B_k, max_path_length).bool()
            candidates[:, :max_dummy_size] = False

            rec = torch.zeros(B_k, max_path_length).long()
            selected_node = torch.zeros(B_k, 1).long()
            cum_demand = torch.zeros(B_k, 2)

            for i in range(max_path_length - 1):
                dists = torch.arange(problem_size).view(-1, problem_size).expand(B_k, problem_size).clone()
                dists.scatter_(1, selected_node, 1e5)
                dists[~candidates] = 1e5
                dists[cum_demand[:, -1:] + depot_node_demand_expanded > 1.] = 1e5
                dists.scatter_(1, cum_demand[:, :-1].long() + 1, 1e4)

                next_selected_node = dists.min(-1)[1].view(-1, 1)
                selected_demand = depot_node_demand_expanded.gather(1, next_selected_node)
                cum_demand[:, -1:] = torch.where(selected_demand > 0, selected_demand + cum_demand[:, -1:], 0 * cum_demand[:, -1:])
                cum_demand[:, :-1] = torch.where(selected_demand > 0, cum_demand[:, :-1], cum_demand[:, :-1] + 1)

                rec.scatter_(1, selected_node, next_selected_node)
                candidates.scatter_(1, next_selected_node, 0)
                selected_node = next_selected_node

            solutions = rec2sol(rec).view(batch_size, k, -1)
            self.infeasible = torch.zeros((batch_size, k)).bool() # False

        elif strategy == "greedy_feasible":
            assert k == 1, "It can only generate one solution when using greedy strategy!"
            B_k = batch_size * k
            candidates = torch.ones(B_k, max_path_length).bool()
            candidates[:, :max_dummy_size] = False

            rec = torch.zeros(B_k, max_path_length).long()
            selected_node = torch.zeros(B_k, 1).long()
            cum_demand = torch.zeros(B_k, 2)

            coor = dummify(depot_node_xy_expanded, max_dummy_size-1)
            demand = dummify(depot_node_demand_expanded, max_dummy_size-1)

            for i in range(max_path_length - 1):
                coor_now = coor.gather(1, selected_node.unsqueeze(-1).expand(B_k, max_path_length, 2))
                dists = (coor_now - coor).norm(p=2, dim=2)

                dists.scatter_(1, selected_node, 1e5)
                dists[~candidates] = 1e5
                dists[cum_demand[:, -1:] + demand > 1.] = 1e5
                dists.scatter_(1, cum_demand[:, :-1].long() + 1, 1e4)

                next_selected_node = dists.min(-1)[1].view(-1, 1)
                selected_demand = demand.gather(1, next_selected_node)
                cum_demand[:, -1:] = torch.where(selected_demand > 0, selected_demand + cum_demand[:, -1:], 0 * cum_demand[:, -1:])
                cum_demand[:, :-1] = torch.where(selected_demand > 0, cum_demand[:, :-1], cum_demand[:, :-1] + 1)

                rec.scatter_(1, selected_node, next_selected_node)
                candidates.scatter_(1, next_selected_node, 0)
                selected_node = next_selected_node

            solutions = rec2sol(rec).view(batch_size, k, -1)
            self.infeasible = torch.zeros((batch_size, k)).bool() # False

        else:
            raise NotImplementedError()

        self.selected_node_list = solutions
        # shape: (batch, k, solution)

        return self._get_travel_distance()

    def improvement_step(self, rec, action, obj, feasible_history, t, weights=0, out_reward = False, penalty_factor=1., improvement_method = "kopt", insert_before=True, epsilon=EPSILON_hardcoded, seperate_obj_penalty=False, non_linear=None, n2s_decoder=False, penalty_normalize=False):

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
        next_obj, context, out_penalty, out_node_penalty = self.get_costs(next_state, get_context=True, out_reward=out_reward, penalty_factor=penalty_factor)
        # if out_reward:
        #     out_node_penalty = (context[2] > 1.00001).sum(dim=-1)
        #     out_penalty = ((context[3]-1.00001) * (context[3] > 1.00001)).sum(dim=-1)
        #     # shape: (batch * pomo,)
        #     next_obj = next_obj + penalty_factor * (out_node_penalty + out_penalty)

        # MDP step
        non_feasible_cost_total = out_penalty.sum(0)
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

    def preprocessing(self, solutions):

        demand = self.dummy_demand

        batch_size, seq_length = solutions.size()
        assert seq_length < 1000
        arange = torch.arange(batch_size)

        pre = torch.zeros(batch_size).long()
        route = torch.zeros(batch_size).long()
        route_plan_visited_time = torch.zeros((batch_size, seq_length)).long()
        cum_demand = torch.zeros((batch_size, seq_length))
        partial_sum_wrt_route_plan = torch.zeros((batch_size, self.dummy_size))

        for i in range(seq_length):
            next_ = solutions[arange, pre]
            next_is_dummy_node = next_ < self.dummy_size
            route[next_is_dummy_node] += 1
            route_plan_visited_time[arange, next_] = (route % self.dummy_size) * int(1e3) + (i + 1) % seq_length
            new_cum_demand = partial_sum_wrt_route_plan[arange, route % self.dummy_size] + demand[arange, next_]
            partial_sum_wrt_route_plan[arange, route % self.dummy_size] = new_cum_demand.clone()
            cum_demand[arange, next_] = new_cum_demand * (~next_is_dummy_node)

            pre = next_.clone()

        route_plan_0x = (route_plan_visited_time // int(1e3))

        out = (
                route_plan_0x,  # route plan 0xxxxx, belongs to which route
               (route_plan_visited_time % int(1e3)),  # visited time
               cum_demand.clone(),  # cum_demand (inclusive)
               partial_sum_wrt_route_plan.clone()# partial_sum_wrt_route_plan
               )

        return out

    def get_order(self, rec, return_solution=False):

        bs, p_size = rec.size()
        visited_time = torch.zeros((bs, p_size), device=rec.device)
        pre = torch.zeros((bs), device=rec.device).long()
        for i in range(p_size - 1):
            visited_time[torch.arange(bs), rec[torch.arange(bs), pre]] = i + 1
            pre = rec[torch.arange(bs), pre]
        if return_solution:
            return visited_time.argsort()  # return decoded solution in sequence
        else:
            return visited_time.long()  # also return visited order

    def check_feasibility(self, rec, partial_sum_wrt_route_plan, basic=False):
        p_size = rec.size(1)
        assert (
                (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec) ==
                rec.sort(1)[0]
        ).all(), ((
                          (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec) ==
                          rec.sort(1)[0]
                  ).sum(-1), "not visiting all nodes")

        real_solution = self.get_order(rec, True)

        assert (
                (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec) ==
                real_solution.sort(1)[0]
        ).all(), ((
                          (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec) ==
                          real_solution.sort(1)[0]
                  ).sum(-1), "not valid tour")

        if not basic:
            assert (partial_sum_wrt_route_plan <= 1 + 1e-5).all(), (
            "not satisfying capacity constraint", partial_sum_wrt_route_plan, partial_sum_wrt_route_plan.max())

    def get_costs(self, rec, get_context=False, check_full_feasibility=False, out_reward=False, penalty_factor=1.0, penalty_normalize=False, seperate_obj_penalty=False, non_linear=None):

        # preprocess: make it with dummy depot
        pomo_size = rec.size(0) // self.batch_size
        self.dummy_size = rec.size(1) - self.problem_size

        if self.dummy_xy is None:
            coor = dummify(self.depot_node_xy, self.dummy_size-1)
            self.dummy_xy = coor.repeat_interleave(pomo_size, 0)
        if self.dummy_demand is None:
            demand = dummify(self.depot_node_demand, self.dummy_size-1)
            self.dummy_demand = demand.repeat_interleave(pomo_size, 0)

        # # check TSP feasibility if needed
        # if self.with_assert:
        #     self.check_feasibility(rec, None, basic=True)

        # check full feasibility if needed
        if check_full_feasibility or get_context:
            context = self.preprocessing(rec)
            if check_full_feasibility:
                self.check_feasibility(rec, context[-1], basic=False)

        coor_next = self.dummy_xy.gather(1, rec.long().unsqueeze(-1).expand(*rec.size(), 2))
        cost = (self.dummy_xy - coor_next).norm(p=2, dim=2).sum(1)

        out_node_penalty = (context[2] > 1.00001).sum(-1)
        out_penalty = ((context[3] - 1.00001) * (context[3] > 1.00001)).sum(dim=1)
        # shape: (batch * pomo,)

        if out_reward:
            cost = cost + penalty_factor * (out_node_penalty + out_penalty)

        # get CVRP context
        if get_context:
            return cost, context, out_penalty.unsqueeze(0), out_node_penalty.unsqueeze(0)
        else:
            return cost, out_penalty.unsqueeze(0), out_node_penalty.unsqueeze(0)

    def get_dynamic_feature(self, context, with_infsb_feature, tw_normalize=False):

        route_plan_0x, visited_time, cum_demand, partial_sum_wrt_route_plan = context
        demand = self.dummy_demand.unsqueeze(-1)
        cum_demand = cum_demand.unsqueeze(-1)
        route_total_demand_per_node = partial_sum_wrt_route_plan.gather(-1, route_plan_0x).unsqueeze(-1)

        infeasibility_indicator_after_visit = torch.clamp_min(cum_demand - 1.00001, 0.0) > 0
        infeasibility_indicator_before_visit = torch.clamp_min((cum_demand - demand) - 1.00001, 0.0) > 0

        to_actor = torch.cat((
            cum_demand, # demand,
            route_total_demand_per_node - cum_demand,
            (demand == 0).float(),
            infeasibility_indicator_before_visit,
            infeasibility_indicator_after_visit,
        ), -1)  # the node features

        feature = torch.cat([self.dummy_xy, self.dummy_demand.unsqueeze(-1)], dim=-1)
        supplement_feature = to_actor
        if not with_infsb_feature:
            supplement_feature = to_actor[:, :, :-2]
        feature = torch.cat((feature, supplement_feature), dim=-1)
        depot_feature = torch.cat((feature[:, :self.dummy_size, :3], feature[:, :self.dummy_size, 4:]),
                                  dim=2)  # rm demand dimension
        node_feature = feature[:, self.dummy_size:]

        return visited_time, depot_feature, node_feature
