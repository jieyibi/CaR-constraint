from dataclasses import dataclass
from turtle import distance
import torch
import os, pickle
import numpy as np
from utils import *

__all__ = ['PCTSPEnv']
EPSILON_hardcoded = 0.05  # maximal 5% of the precedence constraint can be violated


@dataclass
class Reset_State:
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    precedence_matrix: torch.Tensor = None  # precedence[b, i, j] = -1 if i precedes j and i->j is infeasible
    # shape: (batch, problem, problem)
    distance_matrix: torch.Tensor = None  # distance = Euclidean distance + precedence_matrix (Edge weight in LKH)
    # shape: (batch, problem, problem)
    color_matrix: torch.Tensor = None  # color[b, i, c] = 1 if node i belongs to color c (color = salesman)
    # shape: (batch, problem, color)


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
    length: torch.Tensor = None
    # shape: (batch, pomo)
    current_coord: torch.Tensor = None
    # shape: (batch, pomo, 2)


class PCTSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.problem = "PCTSP"
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.color_size = env_params['color_size']  # salesmen number
        self.pomo_size = env_params['pomo_size']
        self.loc_scaler = env_params['loc_scaler'] if 'loc_scaler' in env_params.keys() else None
        # hardness control parameters
        self.precedence_ratio = env_params['precedence_ratio']  # ratio of precedence constraints
        self.color_ratio = env_params['color_ratio']  # average ratio of salesmen that can visit each node

        self.epsilon = EPSILON_hardcoded
        self.k_max = self.env_params['k_max'] if 'k_max' in env_params.keys() else None
        if 'pomo_start' in env_params.keys():
            self.pomo_size = env_params['pomo_size'] if env_params['pomo_start'] else env_params['train_z_sample_size']

        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in env_params.keys() else \
            env_params['device']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.START_NODE = None
        # IDX.shape: (batch, pomo)
        self.node_xy = None
        # shape: (batch, problem, 2)
        self.precedence_matrix = None
        # shape: (batch, problem, problem)
        self.color_matrix = None
        # shape: (batch, problem, color)
        self.distance_matrix = None
        # shape: (batch, problem, problem)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        self.infeasibility_list = None
        self.wrong_precedence_list = None  # shape: (batch, pomo, 0~)
        self.wrong_color_list = None
        self.current_salesman = None  # shape: (batch, pomo) - current salesman ID for each pomo
        self.salesman_node_list = None  # shape: (batch, pomo, color_size) - list of nodes for each salesman

        # Dynamic-2
        ####################################
        self.visited_ninf_flag = None
        self.wrong_precedence_ninf_flag = None
        self.wrong_color_ninf_flag = None
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
            node_xy, precedence_matrix, color_matrix, distance_matrix = problems
        else:
            node_xy, precedence_matrix, color_matrix, distance_matrix = self.get_random_problems(batch_size,
                                                                                                 self.problem_size,
                                                                                                 normalized=normalize)
        self.batch_size = node_xy.size(0)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                node_xy = self.augment_xy_data_by_8_fold(node_xy)
                precedence_matrix = precedence_matrix.repeat(8, 1, 1)
                color_matrix = color_matrix.repeat(8, 1, 1)
                distance_matrix = distance_matrix.repeat(8, 1, 1)
            else:
                raise NotImplementedError
        # print(node_xy.size())
        self.node_xy = node_xy
        # shape: (batch, problem, 2)
        self.precedence_matrix = precedence_matrix
        # shape: (batch, problem, problem)
        self.color_matrix = color_matrix
        # shape: (batch, problem, color)
        self.distance_matrix = distance_matrix
        # shape: (batch, problem, problem)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)

        self.reset_state.node_xy = node_xy
        self.reset_state.precedence_matrix = precedence_matrix
        self.reset_state.color_matrix = color_matrix
        self.reset_state.distance_matrix = distance_matrix

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.START_NODE = torch.arange(start=1, end=self.pomo_size + 1)[None, :].expand(self.batch_size,
                                                                                                   -1).to(self.device)
        self.step_state.PROBLEM = self.problem

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        self.wrong_precedence_list = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)
        self.wrong_color_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.bool).to(self.device)
        self.infeasibility_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.bool).to(
            self.device)  # True for causing infeasibility
        # shape: (batch, pomo, 0~)

        # Initialize current_salesman: each pomo starts with salesman 0 (first salesman)
        # Note: salesman IDs are 0-indexed (0 to color_size-1)
        self.current_salesman = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long).to(self.device)
        # shape: (batch, pomo)
        # Initialize salesman_node_list: track which nodes belong to which salesman
        self.salesman_node_list = [[] for _ in range(self.batch_size * self.pomo_size)]
        # Will be converted to tensor when needed

        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.wrong_precedence_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(
            self.device)
        self.wrong_color_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(
            self.device)
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
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.infeasible = self.infeasible
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        reward = None
        done = False
        return self.step_state, reward, done

    def _compute_precedence_mask(self):
        """
        Compute mask for precedence constraints.
        If node j has been visited and precedence_matrix[i, j] = True (i must precede j),
        then node i cannot be visited anymore (mask it).

        Returns:
            precedence_mask: (batch, pomo, problem) - True means node is masked due to precedence violation
        """
        # visited_ninf_flag: (batch, pomo, problem)
        # visited[i] = -inf means node i has been visited
        visited = (self.visited_ninf_flag == float('-inf'))  # (batch, pomo, problem)

        # precedence_matrix: (batch, problem, problem)
        # precedence_matrix[b, i, j] = True means i must precede j

        # For each batch and pomo, check if any visited node j has precedence constraint i->j
        # where i has not been visited yet
        # We need to mask node i if:
        #   - i has not been visited
        #   - There exists j such that j has been visited AND precedence_matrix[i, j] = True

        # Expand visited to (batch, pomo, problem, 1) for broadcasting
        visited_expanded = visited.unsqueeze(-1)  # (batch, pomo, problem, 1)

        # Expand precedence_matrix to (batch, 1, problem, problem)
        precedence_expanded = self.precedence_matrix.unsqueeze(1)  # (batch, 1, problem, problem)

        # Check for each node i: does there exist j such that precedence[i, j] = True and visited[j] = True?
        # precedence_expanded: (batch, 1, problem, problem)
        # visited_expanded: (batch, pomo, problem, 1)
        # We want: for each (b, p, i), check if any j exists such that precedence[b, i, j] = True and visited[b, p, j] = True
        # This is: (precedence_expanded & visited_expanded).any(dim=-1)
        # But we need to align dimensions: precedence is (batch, 1, problem, problem), visited is (batch, pomo, problem, 1)
        # We need: (batch, pomo, problem, problem) where [b, p, i, j] = precedence[b, i, j] & visited[b, p, j]

        # Align dimensions: expand precedence to (batch, pomo, problem, problem)
        precedence_aligned = precedence_expanded.expand(-1, self.pomo_size, -1, -1)  # (batch, pomo, problem, problem)
        # visited_expanded: (batch, pomo, problem, 1) -> expand to (batch, pomo, 1, problem) then transpose
        visited_for_precedence = visited.unsqueeze(2)  # (batch, pomo, 1, problem)
        visited_for_precedence = visited_for_precedence.expand(-1, -1, self.problem_size,
                                                               -1)  # (batch, pomo, problem, problem)

        # Check: precedence[b, p, i, j] = True AND visited[b, p, j] = True
        precedence_violation = precedence_aligned & visited_for_precedence  # (batch, pomo, problem, problem)

        # For each node i, if any j violates precedence, mask i
        precedence_mask = precedence_violation.any(dim=-1)  # (batch, pomo, problem)

        # Only mask nodes that haven't been visited yet
        precedence_mask = precedence_mask & (~visited)  # (batch, pomo, problem)

        return precedence_mask

    def _compute_color_mask(self):
        """
        Compute mask for color constraints.
        Mask nodes that cannot be visited by current_salesman.

        Returns:
            color_mask: (batch, pomo, problem) - True means node is masked due to color constraint
        """
        # current_salesman: (batch, pomo)
        # color_matrix: (batch, problem, color_size)

        # Expand current_salesman to (batch, pomo, 1)
        current_salesman_expanded = self.current_salesman.unsqueeze(-1)  # (batch, pomo, 1)

        # Get color constraints for all nodes: (batch, problem, color_size)
        # For each (batch, pomo), get allowed colors for all nodes
        # We need to gather: color_matrix[batch, :, current_salesman[batch, pomo]]
        # This is tricky because we need to index color_matrix with current_salesman

        # Method: For each batch and pomo, check if node i can be visited by current_salesman[b, p]
        # color_matrix: (batch, problem, color_size)
        # current_salesman: (batch, pomo)

        # Create indices: for each (b, p), we want color_matrix[b, :, current_salesman[b, p]]
        batch_idx = torch.arange(self.batch_size, device=self.device)[:, None, None].expand(-1, self.pomo_size,
                                                                                            self.problem_size)
        node_idx = torch.arange(self.problem_size, device=self.device)[None, None, :].expand(self.batch_size,
                                                                                             self.pomo_size, -1)
        salesman_idx = current_salesman_expanded.expand(-1, -1, self.problem_size)  # (batch, pomo, problem)

        # Gather: color_matrix[batch_idx, node_idx, salesman_idx]
        # This gives us: (batch, pomo, problem) - True if node can be visited by current_salesman
        can_visit = self.color_matrix[batch_idx, node_idx, salesman_idx]  # (batch, pomo, problem)

        # Mask nodes that cannot be visited
        color_mask = ~can_visit  # (batch, pomo, problem)

        # Don't mask already visited nodes (they're already masked by visited_ninf_flag)
        visited = (self.visited_ninf_flag == float('-inf'))  # (batch, pomo, problem)
        color_mask = color_mask & (~visited)  # (batch, pomo, problem)

        return color_mask

    def step(self, selected, visit_mask_only=True, out_reward=False, generate_PI_mask=False,
             use_predicted_PI_mask=False, pip_step=1,
             soft_constrained=False, backhaul_mask=None, penalty_normalize=False):
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
        # visit constraint
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem)

        # PCTSP-specific constraints: precedence and color
        # Compute precedence mask: mask nodes that violate precedence constraints
        precedence_mask = self._compute_precedence_mask()  # (batch, pomo, problem)
        self.wrong_precedence_ninf_flag[precedence_mask] = float('-inf')

        # Compute color mask: mask nodes that cannot be visited by current_salesman
        color_mask = self._compute_color_mask()  # (batch, pomo, problem)
        self.wrong_color_ninf_flag[color_mask] = float('-inf')

        # Combine all masks: visited, precedence, and color constraints
        self.ninf_mask = self.visited_ninf_flag.clone()
        # Apply precedence mask
        self.ninf_mask[self.wrong_precedence_ninf_flag == float('-inf')] = float('-inf')
        # Apply color mask
        self.ninf_mask[self.wrong_color_ninf_flag == float('-inf')] = float('-inf')

        # Optional: draft limit constraint (if exists)
        if hasattr(self, 'node_draft_limit') and self.node_draft_limit is not None:
            round_error_epsilon = 0.00001
            dl_list = self.node_draft_limit[:, None, :].expand(self.batch_size, self.pomo_size, -1)
            # shape: (batch, pomo, problem)

            # simulate the right infsb mask and see ATTENTION!
            if generate_PI_mask and self.selected_count < self.problem_size - 1:
                self._calculate_PIP_mask(pip_step)

            # (current load + demand of next node > draft limit of next node) means infeasible
            if hasattr(self, 'load') and hasattr(self, 'node_demand'):
                demand_list = self.node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
                out_of_dl = self.load[:, :, None] + demand_list > dl_list + round_error_epsilon
                # shape: (batch, pomo, problem)
                if hasattr(self, 'out_of_dl_ninf_flag'):
                    self.out_of_dl_ninf_flag[out_of_dl] = float('-inf')
                # shape: (batch, pomo, problem)
                # value that exceeds draft limit of the selected node = current load - node_draft_limit
                total_out_of_dl = self.load - self.node_draft_limit[torch.arange(self.batch_size)[:, None], selected]
                # negative value means current load < node_draft_limit, turn it into 0
                total_out_of_dl = torch.where(total_out_of_dl < 0, torch.zeros_like(total_out_of_dl), total_out_of_dl)
                # shape: (batch, pomo)
                if hasattr(self, 'out_of_draft_limit_list'):
                    self.out_of_draft_limit_list = torch.cat(
                        (self.out_of_draft_limit_list, total_out_of_dl[:, :, None]), dim=2)

                if not visit_mask_only or not soft_constrained:
                    self.ninf_mask[out_of_dl] = float('-inf')

        if generate_PI_mask and self.selected_count < self.problem_size - 1 and (not use_predicted_PI_mask):
            if hasattr(self, 'simulated_ninf_flag'):
                self.ninf_mask = torch.where(self.simulated_ninf_flag == float('-inf'), float('-inf'), self.ninf_mask)
                all_infsb = ((self.ninf_mask == float('-inf')).all(dim=-1)).unsqueeze(-1).expand(-1, -1,
                                                                                                 self.problem_size)
                self.ninf_mask = torch.where(all_infsb, self.visited_ninf_flag, self.ninf_mask)

        # Check for infeasibility:
        # 1. Precedence violation: if a node that must precede an already-visited node is still unvisited
        # 2. Color violation: if a node cannot be visited by any salesman
        # 3. Draft limit violation (if applicable)
        precedence_violation = (self.wrong_precedence_ninf_flag == float('-inf')).any(dim=2)  # (batch, pomo)
        color_violation = (self.wrong_color_ninf_flag == float('-inf')).any(dim=2)  # (batch, pomo)

        # Check if all remaining unvisited nodes are masked (infeasible)
        unvisited = (self.visited_ninf_flag == 0)  # (batch, pomo, problem)
        all_unvisited_masked = ((unvisited & (self.ninf_mask == float('-inf'))).sum(dim=2) == unvisited.sum(dim=2)) & (
                    unvisited.sum(dim=2) > 0)
        # (batch, pomo) - True if all unvisited nodes are masked and there are unvisited nodes

        newly_infeasible = precedence_violation | color_violation | all_unvisited_masked

        # Also check draft limit if applicable
        if hasattr(self, 'out_of_dl_ninf_flag'):
            draft_limit_violation = (
                    ((self.visited_ninf_flag == 0).int() + (self.out_of_dl_ninf_flag == float('-inf')).int()) == 2).any(
                dim=2)
            newly_infeasible = newly_infeasible | draft_limit_violation

        self.infeasible = self.infeasible | newly_infeasible
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
                out_of_dl_nodes_reward = - torch.where(self.out_of_draft_limit_list > round_error_epsilon,
                                                       torch.ones_like(self.out_of_draft_limit_list),
                                                       self.out_of_draft_limit_list).sum(-1).int()
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
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)
        # For non-cyclic path, exclude the last segment (return to start)
        segment_lengths = segment_lengths[:, :, :-1]
        # shape: (batch, pomo, selected_list_length - 1)

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
        print("Save PCTSP dataset to {}".format(path))

    def load_dataset(self, path, offset=0, num_samples=1000, disable_print=True):
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset + num_samples]
            if not disable_print:
                print(">> Load {} data ({}) from {}".format(len(data), type(data), path))
        node_xy, precedence_matrix, color_matrix, distance_matrix = [i[0] for i in data], [i[1] for i in data], [i[2]
                                                                                                                 for i
                                                                                                                 in
                                                                                                                 data], [
            i[3] for i in data]
        node_xy, precedence_matrix, color_matrix, distance_matrix = torch.Tensor(node_xy), torch.Tensor(
            precedence_matrix), torch.Tensor(color_matrix), torch.Tensor(distance_matrix)
        data = (node_xy, precedence_matrix, color_matrix, distance_matrix)
        return data

    def get_random_problems(self, batch_size, problem_size, normalized=True):

        precedence_ratio = self.precedence_ratio
        color_ratio = self.color_ratio
        color_size = self.color_size

        # 1. Generate node coordinates (depot at index 0, customers at 1:problem_size)
        node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)

        # 2. Compute Euclidean distance matrix
        # node_xy: (batch, problem, 2)
        # Expand to compute pairwise distances
        node_xy_i = node_xy.unsqueeze(2)  # (batch, problem, 1, 2)
        node_xy_j = node_xy.unsqueeze(1)  # (batch, 1, problem, 2)
        distance_matrix = torch.sqrt(((node_xy_i - node_xy_j) ** 2).sum(dim=-1) + 1e-8)  # (batch, problem, problem)

        # 3. Generate precedence constraints (guaranteed feasible by construction)
        # if distance[i][j] == -1, it means:
        #   - i cannot directly go to j (infeasible edge)
        #   - i must precede j (precedence constraint)
        precedence_matrix = torch.zeros(size=(batch_size, problem_size, problem_size), dtype=torch.bool)

        # Strategy: Generate random topological orders for all batches, then add precedence constraints
        # This guarantees acyclicity and feasibility

        # Generate random permutations (topological orders) for all batches
        # Shape: (batch_size, problem_size)
        # Use argsort on random values to generate random permutations
        random_vals = torch.rand(batch_size, problem_size)
        topo_orders = torch.argsort(random_vals, dim=1)

        # Sample precedence constraints based on topological order
        # Only allow precedence from earlier nodes to later nodes in topo_order
        # Since we exclude adjacent pairs (pos_j - pos_i > 1), the maximum number of valid pairs is:
        # For n nodes: (n-2) + (n-3) + ... + 1 = (n-2)(n-1)/2
        max_valid_pairs = (problem_size - 2) * (problem_size - 1) // 2 if problem_size > 2 else 0
        num_constraints = int(precedence_ratio * max_valid_pairs)
        if num_constraints > 0:
            # Create position mapping for all batches
            # pos[b, i] = position of node i in topo_order for batch b
            # Shape: (batch_size, problem_size)
            pos = torch.zeros(batch_size, problem_size, dtype=torch.long)
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, problem_size)
            pos[batch_indices, topo_orders] = torch.arange(problem_size).unsqueeze(0).expand(batch_size, -1)

            # Generate all pairs (i, j) using meshgrid
            # Shape: (problem_size, problem_size)
            i_indices, j_indices = torch.meshgrid(
                torch.arange(problem_size, dtype=torch.long),
                torch.arange(problem_size, dtype=torch.long),
                indexing='ij'
            )

            # Expand for all batches
            # Shape: (batch_size, problem_size, problem_size)
            pos_i = pos[:, i_indices]  # pos[b, i] for all i
            pos_j = pos[:, j_indices]  # pos[b, j] for all j

            # Filter: only keep pairs where i comes before j in topo_order
            # This ensures: if distance[i][j] == -1, then:
            #   1. i must precede j (precedence constraint)
            #   2. i cannot directly go to j (infeasible edge, Distance_SOP returns M)
            # Additionally, we exclude adjacent pairs in topo_order (pos_j - pos_i > 1)
            # to avoid selecting pairs that are directly adjacent in the topological order
            # i.e., pos[b, i] < pos[b, j] and i != j and (pos_j - pos_i) > 1
            # Shape: (batch_size, problem_size, problem_size)
            valid_mask = (pos_i < pos_j) & (i_indices.unsqueeze(0) != j_indices.unsqueeze(0)) & ((pos_j - pos_i) > 1)

            # Generate random selection mask for all batches
            # Create a random probability matrix and select top-k for each batch
            # Shape: (batch_size, problem_size, problem_size)
            random_probs = torch.rand(batch_size, problem_size, problem_size, device=valid_mask.device)
            # Set invalid pairs to -inf so they won't be selected
            random_probs = torch.where(valid_mask, random_probs,
                                       torch.tensor(float('-inf'), device=random_probs.device))

            # Select top num_constraints for each batch
            # Flatten last two dimensions for easier topk selection
            random_probs_flat = random_probs.view(batch_size, -1)  # (batch_size, problem_size^2)
            _, topk_indices = torch.topk(random_probs_flat, k=num_constraints, dim=1)

            # Convert flat indices back to (i, j) pairs
            batch_idx = torch.arange(batch_size, device=topk_indices.device).unsqueeze(1).expand(-1,
                                                                                                 topk_indices.size(1))
            flat_indices = topk_indices.flatten()
            batch_flat = batch_idx.flatten()

            # Get (i, j) coordinates from flat indices
            # flat_index = i * problem_size + j
            selected_i_flat = (flat_indices // problem_size).long()
            selected_j_flat = (flat_indices % problem_size).long()

            # Filter out invalid selections (where valid_mask is False)
            valid_selections = valid_mask[batch_flat, selected_i_flat, selected_j_flat]
            selected_i = selected_i_flat[valid_selections]
            selected_j = selected_j_flat[valid_selections]
            selected_batch = batch_flat[valid_selections]

            # Set precedence constraints (fully vectorized)
            if len(selected_i) > 0:
                precedence_matrix[selected_batch, selected_i, selected_j] = True
                # Set distance to -1 (LKH convention)
                distance_matrix[selected_batch, selected_i, selected_j] = -1.0

            # Compute transitive closure for all batches (vectorized)
            prec = precedence_matrix.float()
            for k in range(problem_size):
                # Compute transitive closure: prec | (prec[:, :, k] & prec[:, k, :])
                # Use maximum instead of | for float tensors
                prec = torch.maximum(prec, (prec[:, :, k:k + 1] * prec[:, k:k + 1, :]))
            precedence_matrix = (prec > 0).bool()

        # 4. Generate color constraints (GCTSP_SET_SECTION format) - fully vectorized
        # color_matrix[b, i, c] = 1 means node i can be visited by salesman/color c
        color_matrix = torch.zeros(size=(batch_size, problem_size, color_size), dtype=torch.bool)

        # Each node must have at least one color
        # Sample colors based on color_ratio
        num_colors = max(1, int(color_ratio * color_size))

        if num_colors < color_size:
            # Generate random values for all (batch, node, color) combinations
            # Shape: (batch_size, problem_size, color_size)
            random_color_vals = torch.rand(batch_size, problem_size, color_size)
            # Use argsort to get random permutations for each node
            color_permutations = torch.argsort(random_color_vals, dim=2)
            # Select top num_colors for each node
            selected_color_indices = color_permutations[:, :, :num_colors]

            # Set selected colors to True using advanced indexing
            batch_idx = torch.arange(batch_size, device=selected_color_indices.device).unsqueeze(1).unsqueeze(2).expand(
                -1, problem_size, num_colors)
            node_idx = torch.arange(problem_size, device=selected_color_indices.device).unsqueeze(0).unsqueeze(
                2).expand(batch_size, -1, num_colors)
            color_matrix[batch_idx, node_idx, selected_color_indices] = True
        else:
            # All colors allowed for all nodes
            color_matrix[:, :, :] = True

        # # 5. Verify feasibility for each batch (sanity check)
        # # Since we generate constraints based on topological order, they should always be feasible
        # # But we verify to catch any bugs
        # for b in range(batch_size):
        #     feasible = self._verify_feasibility(
        #         precedence_matrix[b],
        #         color_matrix[b],
        #         problem_size,
        #         color_size
        #     )
        #     if not feasible:
        #         # This should never happen with the new generation method
        #         # But if it does, we raise an error to catch bugs
        #         raise RuntimeError(f"Generated infeasible instance at batch {b}. "
        #                          f"This indicates a bug in constraint generation.")

        return node_xy, precedence_matrix, color_matrix, distance_matrix

    def _verify_feasibility(self, precedence_matrix, color_matrix, problem_size, color_size):
        """
        Verify that a PCTSP instance is feasible:
        1. Precedence constraints form a DAG (no cycles)
        2. Each node has at least one color
        3. Topological sort exists (guaranteed if DAG)

        Args:
            precedence_matrix: (problem_size, problem_size) bool tensor
            color_matrix: (problem_size, color_size) bool tensor
            problem_size: int
            color_size: int

        Returns:
            bool: True if feasible, False otherwise
        """
        # Check 1: Verify DAG (no cycles in precedence constraints)
        # Compute transitive closure
        prec = precedence_matrix.float().clone()
        for k in range(problem_size):
            prec = prec | (prec[:, k:k + 1] & prec[k:k + 1, :])

        # Check for cycles (diagonal should be all zeros)
        if (prec.diagonal() > 0).any():
            return False

        # Check 2: Each node has at least one color
        if not (color_matrix.sum(dim=1) >= 1).all():
            return False

        # Check 3: Topological sort exists (Kahn's algorithm)
        # Build in-degree for each node
        in_degree = precedence_matrix.sum(dim=0)  # (problem_size,)

        # Find nodes with no incoming edges
        queue = (in_degree == 0).nonzero(as_tuple=False).squeeze(-1).tolist()
        if not queue:
            # If no node has zero in-degree and there are edges, it's a cycle
            if precedence_matrix.sum() > 0:
                return False

        # Perform topological sort
        visited_count = 0
        while queue:
            node = queue.pop(0)
            visited_count += 1

            # Remove this node and update in-degrees
            for j in range(problem_size):
                if precedence_matrix[node, j]:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)

        # All nodes should be visited if DAG
        if visited_count < problem_size:
            return False

        return True

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

    def get_initial_solutions(self, strategy, k, max_dummy_size=0):
        batch_size, problem_size, _ = self.node_xy.size()
        if strategy == "random":  # not guarantee feasibility (may exceed tw)
            # start from 0
            B_k = batch_size * k
            # # random solution permutation
            customer = torch.rand(B_k, problem_size - 1).argsort(dim=1) + 1
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
        current_load = torch.zeros((batch_size,))
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

    def get_costs(self, rec, get_context=False, check_full_feasibility=False, out_reward=False, penalty_factor=1.0,
                  penalty_normalize=False, seperate_obj_penalty=False, non_linear=None, wo_node_penalty=False,
                  wo_tour_penalty=False):

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
        is_depot = torch.tensor([1.] + [0.] * (seq_length - 1))[None, :].repeat_interleave(batch_size, 0)
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

        feature = torch.cat(
            [self.node_xy.repeat_interleave(k, 0), node_demand.unsqueeze(-1), node_draft_limit.unsqueeze(-1)], dim=-1)
        supplement_feature = to_actor
        if not with_infsb_feature:
            supplement_feature = to_actor[:, :, :-1]
        feature = torch.cat((feature, supplement_feature), dim=-1)

        return visited_time, None, feature

    def improvement_step(self, rec, action, obj, feasible_history, t, weights=0, out_reward=False, penalty_factor=1.,
                         penalty_normalize=False, improvement_method="kopt", insert_before=True,
                         epsilon=EPSILON_hardcoded, seperate_obj_penalty=False, non_linear=None, n2s_decoder=False):

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
        next_obj, context, out_penalty, out_node_penalty = self.get_costs(next_state, get_context=True,
                                                                          out_reward=out_reward,
                                                                          penalty_factor=penalty_factor,
                                                                          penalty_normalize=penalty_normalize)

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
            elif non_linear in ["fixed_epsilon", "decayed_epsilon"]:  # only have reward when penalty <= epsilon
                now_obj[soft_infeasible, 0] = next_obj[soft_infeasible].clone()
            else:
                raise NotImplementedError
        # only update epsilon feasible obj
        now_obj[soft_infeasible, 1] = next_obj[soft_infeasible].clone()
        now_bsf = torch.min(pre_bsf, now_obj)
        rewards = (pre_bsf - now_bsf)  # bs,2 (feasible_reward, epsilon-feasible_reward)

        # feasible history step
        if n2s_decoder:  # calculate the removal record
            # todo: not carefully check yet but probably correct
            info, reg = None, torch.zeros((action.size(0), 1))
            assert not self.env_params["with_regular"], "n2s decoder does not support regularization reward."
            feasible_history[:, 1:] = feasible_history[:, :total_history - 1].clone()
            action_removal = torch.zeros_like(feasible_history[:, 0])
            action_removal[torch.arange(action.size(0)).unsqueeze(1), action[:, :1]] = 1.
            feasible_history[:, 0] = action_removal.clone()
            context2 = torch.cat(
                (
                    feasible_history,  # last three removal
                    feasible_history.mean(1, keepdims=True) if t > (total_history - 2) else feasible_history[:,
                                                                                            :(t + 1)].mean(1,
                                                                                                           keepdims=True),
                    # note: slightly different from N2S due to shorter improvement steps; before/after?
                ), 1)  # (batch_size, 4, solution_size)
        else:
            feasible_history[:, 1:] = feasible_history[:, :total_history - 1].clone()
            feasible_history[:, 0] = feasible.clone()

            # compute the ES features
            feasible_history_pre = feasible_history[:, 1:]
            feasible_history_post = feasible_history[:, :total_history - 1]
            f_to_if = ((feasible_history_pre == True) & (feasible_history_post == False)).sum(1, True) / (
                    total_history - 1)
            f_to_f = ((feasible_history_pre == True) & (feasible_history_post == True)).sum(1, True) / (
                    total_history - 1)
            if_to_f = ((feasible_history_pre == False) & (feasible_history_post == True)).sum(1, True) / (
                    total_history - 1)
            if_to_if = ((feasible_history_pre == False) & (feasible_history_post == False)).sum(1, True) / (
                    total_history - 1)
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
                    new_sol.append(
                        torch.cat((sol_without_removed[b, :pos], remove_idx[b:b + 1], sol_without_removed[b, pos:])))
                else:
                    new_sol.append(torch.cat(
                        (sol_without_removed[b, :pos + 1], remove_idx[b:b + 1], sol_without_removed[b, pos + 1:])))

            updated_sol = torch.stack(new_sol)

        return sol2rec(updated_sol.unsqueeze(1)).squeeze(1)

    def f(self, p):  # The entropy measure in Eq.(5)
        return torch.clamp(1 - 0.5 * torch.log2(2.5 * np.pi * np.e * p * (1 - p) + 1e-5), 0, 1)








