from dataclasses import dataclass
import torch
import os, pickle
import numpy as np
from utils import *

__all__ = ['SOPEnv']
EPSILON_hardcoded = 0.1  # maximal 10% of the precedence constraint can be violated


@dataclass
class Reset_State:
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    precedence_matrix: torch.Tensor = None  # precedence[b, i, j] = -1 if i precedes j and i->j is infeasible
    # shape: (batch, problem, problem)


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


class SOPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.problem = "SOP"
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.loc_scaler = env_params['loc_scaler'] if 'loc_scaler' in env_params.keys() else None
        # Set SOP parameters based on variant
        sop_variant = env_params.get('sop_variant', 1)  # default to variant 1
        if sop_variant == 1:
            self.precedence_ratio = 0.2
            self.geometric_conflict_ratio = 0.3
            self.precedence_balance_ratio = 0.0
        elif sop_variant == 2:
            self.precedence_ratio = 0.2
            self.geometric_conflict_ratio = 0.8
            self.precedence_balance_ratio = 0.0
        else:
            raise ValueError(f"Unknown SOP variant: {sop_variant}")

        self.epsilon = EPSILON_hardcoded
        self.k_max = self.env_params['k_max'] if 'k_max' in env_params.keys() else None
        if 'pomo_start' in env_params.keys():
            self.pomo_size = env_params['pomo_size']

        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in env_params.keys() else \
        env_params['device']

        # Const
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

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        self.infeasibility_list = None
        self.wrong_precedence_list = None  # shape: (batch, pomo, 0~)
        self.unvisited_precedence_node = None # shape: (batch, pomo, problem)

        # Dynamic-2
        ####################################
        self.visited_ninf_flag = None
        self.precedence_ninf_flag = None
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
            node_xy, precedence_matrix = problems
        else:
            node_xy, precedence_matrix = self.get_random_problems(batch_size, self.problem_size, normalized=normalize)
        self.batch_size = node_xy.size(0)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                node_xy = self.augment_xy_data_by_8_fold(node_xy)
                precedence_matrix = precedence_matrix.repeat(8, 1, 1)
            else:
                raise NotImplementedError
        # print(node_xy.size())
        self.node_xy = node_xy
        # shape: (batch, problem, 2)
        self.precedence_matrix = precedence_matrix
        # shape: (batch, problem, problem)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)

        self.reset_state.node_xy = node_xy
        self.reset_state.precedence_matrix = precedence_matrix

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.START_NODE = torch.arange(start=1, end=self.pomo_size + 1)[None, :].expand(self.batch_size, -1).to(self.device)

        self.step_state.PROBLEM = self.problem

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        self.wrong_precedence_list = torch.zeros((self.batch_size, self.pomo_size, 1)).to(self.device)
        self.unvisited_precedence_node = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size), dtype=torch.int).to(self.device)
        self.infeasibility_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.bool).to(self.device)  # True for causing infeasibility
        # shape: (batch, pomo, 0~)

        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.precedence_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # shape: (batch, pomo, problem)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # shape: (batch, pomo, problem)
        # For SOP: mask the last node (end node) until all other nodes are visited
        last_node_idx = self.problem_size - 1
        self.ninf_mask[:, :, last_node_idx] = float('-inf')
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

    def step(self, selected, visit_mask_only=True, out_reward=False, generate_PI_mask=False,
             use_predicted_PI_mask=False, pip_step=1, soft_constrained=False, backhaul_mask=None, penalty_normalize=False):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)
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

        # precedence constraint
        round_error_epsilon = 0.00001
        self.precedence_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # 1. directly infeasible edges
        precedence_expanded = self.precedence_matrix.unsqueeze(1).expand(-1, self.pomo_size, -1, -1)  # (batch, pomo, problem, problem)
        predecessors_of_selected = precedence_expanded[self.BATCH_IDX, self.POMO_IDX, selected, :]
        # shape: (batch, pomo, problem)
        self.precedence_ninf_flag[predecessors_of_selected == -1] = float('-inf')
        # shape: (batch, pomo, problem) -> if precedence_matrix[current][next] == -1, it means: current cannot directly go to next (infeasible edge)
        # 2. Mask out nodes whose predecessors are not yet visited
        not_visited = (self.visited_ninf_flag != float('-inf'))  # (batch, pomo, problem)
        # For each i, check if any predecessor j (except start node) (where precedence_matrix[i,j]==-1) is not visited
        unvisited_predecessors = ((precedence_expanded == -1) & not_visited.unsqueeze(-2))[:,:,:,1:].any(dim=-1)  # (batch, pomo, problem)
        self.precedence_ninf_flag = torch.where(unvisited_predecessors, float('-inf'), self.precedence_ninf_flag)

        # Count precedence violations: number of unvisited predecessors of selected nodes
        if self.selected_count > 1:
            num_unvisited_predecessors = ((precedence_expanded == -1) & not_visited.unsqueeze(-2))[self.BATCH_IDX, self.POMO_IDX,selected,1:].sum(-1).float()  # (batch, pomo)
            self.unvisited_precedence_node[self.BATCH_IDX, self.POMO_IDX, selected] = (num_unvisited_predecessors > round_error_epsilon).int()
            self.wrong_precedence_list = torch.cat((self.wrong_precedence_list, num_unvisited_predecessors[:, :, None]), dim=2) # (batch, pomo, 0~)

        self.ninf_mask = self.visited_ninf_flag.clone()
        if not visit_mask_only or not soft_constrained:
            self.ninf_mask = torch.min(self.ninf_mask, self.precedence_ninf_flag)

        # For SOP: mask the last node until all other nodes (0 to problem_size-2) are visited
        last_node_idx = self.problem_size - 1
        # Check if all nodes except the last one are visited
        # visited_ninf_flag == -inf means visited, so we check if all except last are -inf
        all_others_visited = (self.visited_ninf_flag[:, :, :last_node_idx] == float('-inf')).all(dim=-1)  # (batch, pomo)
        # Only unmask last node if all other nodes are visited
        self.ninf_mask[:, :, last_node_idx] = torch.where(all_others_visited, self.visited_ninf_flag[:, :, last_node_idx], float('-inf'))

        # Check if the route is infeasible: precedence violation
        newly_infeasible = (num_unvisited_predecessors > round_error_epsilon) if self.selected_count > 1 else False
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
                total_wrong_precedence_reward = - self.wrong_precedence_list.sum(dim=-1)
                wrong_precedence_nodes_reward = - self.unvisited_precedence_node.sum(-1).int()
                reward = [dist_reward, total_wrong_precedence_reward, wrong_precedence_nodes_reward]
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
        all_xy = self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)
        # For non-cyclic path, exclude the last segment (no return to start)
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
        print("Save SOP dataset to {}".format(path))

    def load_dataset(self, path, offset=0, num_samples=1000, disable_print=True):
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset + num_samples]
            if not disable_print:
                print(">> Load {} data ({}) from {}".format(len(data), type(data), path))
        node_xy, precedence_matrix = [i[0] for i in data], [i[1] for i in data]
        node_xy, precedence_matrix = torch.Tensor(node_xy), torch.Tensor(precedence_matrix)
        data = (node_xy, precedence_matrix)
        return data

    def get_random_problems(self, batch_size, problem_size, normalized=True):
        precedence_ratio = self.precedence_ratio
        geometric_conflict_ratio = self.geometric_conflict_ratio
        precedence_balance_ratio = self.precedence_balance_ratio

        # SOP rules:
        # - Node 0 (index 0) is the start node, does not participate in precedence constraints
        # - Node problem_size-1 (last node) is the end node, all outgoing edges are -1 (forbidden)
        # - Other nodes (1 to problem_size-2) can participate in precedence constraints

        # 1. Generate node coordinates
        node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)

        # 2. Generate precedence constraints (guaranteed feasible by construction)
        # if precedence_matrix[i][j] == -1, it means:
        #   - i cannot directly go to j (infeasible edge)
        #   - j must precede i (precedence constraint)!!!
        precedence_matrix = torch.zeros(size=(batch_size, problem_size, problem_size), dtype=torch.float32).to(node_xy.device)

        # 3. Generate random topological orders for all batches, then add precedence constraints
        # This guarantees acyclicity and feasibility
        # Shape: (batch_size, problem_size)
        # For middle nodes (1 to problem_size-2), generate random order
        if problem_size > 2:
            random_vals = torch.rand(batch_size, problem_size - 2)
            middle_order = torch.argsort(random_vals, dim=1) + 1  # +1 because we skip node 0
            topo_orders = torch.cat([
                torch.zeros(batch_size, 1, dtype=torch.long),  # Node 0 is always first
                middle_order,
                torch.full((batch_size, 1), problem_size - 1, dtype=torch.long)  # Last node is always last
            ], dim=1)
        else:
            # problem_size <= 2: only start and end nodes
            topo_orders = torch.tensor([[0, problem_size - 1]], dtype=torch.long).expand(batch_size, -1).to(node_xy.device)

        # 4. Sample precedence constraints based on topological order
        # Only allow precedence from earlier nodes to later nodes in topo_order
        # Exclude:
        #   - Node 0 (start) cannot have precedence constraints (it's always first)
        #   - Last node cannot have precedence constraints (it's always last)
        #   - Adjacent pairs in topo_order (pos_j - pos_i > 1)
        # Maximum valid pairs: (n-3) + ... + 1 = (n-3+1)(n-3)/2 for middle nodes
        if problem_size > 2:
            max_valid_pairs = (problem_size - 2) * (problem_size - 3) // 2
            num_constraints = int(precedence_ratio * max_valid_pairs)

            if num_constraints > 0:
                # Create position mapping for all batches
                # pos[b, i] = position of node i in topo_order for batch b
                pos = torch.zeros(batch_size, problem_size, dtype=torch.long)
                batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, problem_size)
                pos[batch_indices, topo_orders] = torch.arange(problem_size).unsqueeze(0).expand(batch_size, -1)
                i_indices, j_indices = torch.meshgrid(
                    torch.arange(problem_size, dtype=torch.long), 
                    torch.arange(problem_size, dtype=torch.long), 
                    indexing='ij'
                )
                pos_i = pos[:, i_indices]  # pos[b, i] for all i
                pos_j = pos[:, j_indices]  # pos[b, j] for all j

                # Filter: valid pairs for precedence constraints
                valid_mask = (
                        (pos_j < pos_i) & # if j < i (j precedes i), i->j infeasible
                        (i_indices.unsqueeze(0) != j_indices.unsqueeze(0)) & # i != j （no self-loop）
                        (i_indices.unsqueeze(0) != 0) &  # i is not start node
                        (j_indices.unsqueeze(0) != 0) &  # j is not start node
                        (i_indices.unsqueeze(0) != problem_size - 1) &  # i is not end node
                        (j_indices.unsqueeze(0) != problem_size - 1)  # j is not end node
                )

                # Apply geometric conflict ratio: increase difficulty by making precedence constraints conflict with geometric proximity
                if geometric_conflict_ratio > 0:
                    # For each valid pair (i, j), compute geometric distance
                    # Higher geometric_conflict_ratio means we prefer pairs that are geometrically far
                    # This creates conflict between precedence and geometry
                    distance_matrix = torch.sqrt(torch.sum((node_xy.unsqueeze(2) - node_xy.unsqueeze(1)) ** 2, dim=-1))
                    geometric_distances = distance_matrix[:, i_indices, j_indices]  # (batch, problem, problem)
                    # Normalize distances to [0, 1] range for each batch
                    max_dist = geometric_distances.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
                    min_dist = geometric_distances.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
                    normalized_dist = (geometric_distances - min_dist) / (max_dist - min_dist + 1e-8)
                    # Higher distance = higher score (we want to select far pairs)
                    conflict_scores = normalized_dist * geometric_conflict_ratio
                else:
                    conflict_scores = torch.zeros_like(valid_mask.float())

                random_probs = torch.rand(batch_size, problem_size, problem_size, device=valid_mask.device)
                
                if precedence_balance_ratio == 0:
                    selection_scores = (random_probs * (1.0 - geometric_conflict_ratio) + conflict_scores * geometric_conflict_ratio)
                    selection_scores = torch.where(valid_mask, selection_scores, torch.tensor(float('-inf'), device=selection_scores.device))
                    
                    flat_scores = selection_scores.view(batch_size, -1)
                    _, topk_indices = torch.topk(flat_scores, k=num_constraints, dim=1)
                    
                    batch_idx = torch.arange(batch_size, device=topk_indices.device).unsqueeze(1).expand(-1, topk_indices.size(1))
                    flat_indices = topk_indices.flatten()
                    batch_flat = batch_idx.flatten()
                    
                    selected_i_flat = (flat_indices // problem_size).long()
                    selected_j_flat = (flat_indices % problem_size).long()
                    
                    valid_selections = valid_mask[batch_flat, selected_i_flat, selected_j_flat]
                    selected_i = selected_i_flat[valid_selections]
                    selected_j = selected_j_flat[valid_selections]
                    selected_batch = batch_flat[valid_selections]
                    
                    if len(selected_i) > 0:
                        precedence_matrix[selected_batch, selected_i, selected_j] = -1.0
                else:
                    node_i_selected_count = torch.zeros(batch_size, problem_size, device=valid_mask.device)
                    node_j_selected_count = torch.zeros(batch_size, problem_size, device=valid_mask.device)
                    remaining_valid_mask = valid_mask.clone()
                    batch_selected_count = torch.zeros(batch_size, dtype=torch.long, device=valid_mask.device)
                    
                    for _ in range(num_constraints):
                        max_i_count = node_i_selected_count.max(dim=-1, keepdim=True)[0]
                        min_i_count = node_i_selected_count.min(dim=-1, keepdim=True)[0]
                        max_j_count = node_j_selected_count.max(dim=-1, keepdim=True)[0]
                        min_j_count = node_j_selected_count.min(dim=-1, keepdim=True)[0]
                        i_range = max_i_count - min_i_count
                        j_range = max_j_count - min_j_count
                        
                        if precedence_balance_ratio >= 1.0:
                            balance_i_prob = 1.0 - (node_i_selected_count - min_i_count) / (i_range + 1e-8)
                            balance_j_prob = 1.0 - (node_j_selected_count - min_j_count) / (j_range + 1e-8)
                        else:
                            balance_i_prob = (node_i_selected_count - min_i_count) / (i_range + 1e-8) 
                            balance_j_prob = (node_j_selected_count - min_j_count) / (j_range + 1e-8) 
                        
                        normalized_i_balance = torch.where(i_range > 1e-8, balance_i_prob, torch.ones_like(node_i_selected_count))
                        normalized_j_balance = torch.where(j_range > 1e-8, balance_j_prob, torch.ones_like(node_j_selected_count))
                        balance_scores = (2 * normalized_i_balance.unsqueeze(-1) + normalized_j_balance.unsqueeze(-2)) / 3.0
                        
                        if geometric_conflict_ratio > 0:
                            combined_scores = (balance_scores * (1.0 - geometric_conflict_ratio) + conflict_scores * geometric_conflict_ratio) + random_probs * 0.01
                        else:
                            combined_scores = balance_scores + random_probs * 0.01
                        
                        combined_scores = torch.where(remaining_valid_mask, combined_scores, torch.tensor(float('-inf'), device=combined_scores.device))
                        
                        flat_scores = combined_scores.view(batch_size, -1)
                        _, top_indices = torch.topk(flat_scores, k=1, dim=1)
                        
                        batch_mask = batch_selected_count < num_constraints
                        if batch_mask.any():
                            flat_indices = top_indices[batch_mask, 0]
                            selected_i = (flat_indices // problem_size).long()
                            selected_j = (flat_indices % problem_size).long()
                            batch_idx = torch.arange(batch_size, device=valid_mask.device)[batch_mask]
                            
                            valid_selections = remaining_valid_mask[batch_idx, selected_i, selected_j]
                            if valid_selections.any():
                                valid_batch = batch_idx[valid_selections]
                                valid_i = selected_i[valid_selections]
                                valid_j = selected_j[valid_selections]
                                
                                precedence_matrix[valid_batch, valid_i, valid_j] = -1.0
                                node_i_selected_count[valid_batch, valid_i] += 1
                                node_j_selected_count[valid_batch, valid_j] += 1
                                batch_selected_count[valid_batch] += 1
                                remaining_valid_mask[valid_batch, valid_i, valid_j] = False

        # mask the start nodes' incoming edges and the end nodes' outgoing edges
        precedence_matrix[:, 1:, 0] = -1.0
        precedence_matrix[:, problem_size - 1, :-1] = -1.0

        return node_xy, precedence_matrix

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

        pre = torch.zeros(batch_size, dtype=torch.long, device=rec.device)
        visited_time = torch.zeros((batch_size, seq_length), dtype=torch.long, device=rec.device)
        current_violation = torch.zeros((batch_size,), device=rec.device)  # Cumulative violation count
        after_violation = torch.zeros((batch_size, seq_length), device=rec.device)  # Violation after visiting node
        violation = torch.zeros((batch_size, seq_length), device=rec.device)  # Violation at the time of visit
        precedence_matrix = self.precedence_matrix.repeat_interleave(k, 0)  # (batch*k, problem, problem)
        visited_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=rec.device)
        
        for i in range(seq_length):
            next_ = rec[arange, pre]
            visited_time[arange, next_] = (i + 1) % seq_length
            # For each batch, check which predecessors of next_ are not yet visited
            # precedence_matrix[b, next_] == -1.0 means those nodes must precede next_
            predecessors_mask = (precedence_matrix[arange, next_] == -1.0)  # (batch, problem)
            unvisited_predecessors = predecessors_mask & (~visited_mask[arange, :])  # (batch, problem)
            new_violations = unvisited_predecessors[:, 1:].sum(dim=-1).float()  # (batch,)
            violation[arange, next_] = new_violations.clone()
            current_violation = current_violation + new_violations
            after_violation[arange, next_] = current_violation.clone()
            visited_mask[arange, next_] = True
            pre = next_.clone()
        
        # Start node (node 0) has no violations
        violation[:, 0] = 0.
        after_violation[:, 0] = 0.

        return (visited_time, after_violation, violation, precedence_matrix)

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
        cost = (coor - coor_next).norm(p=2, dim=2)[:, :-1].sum(1) # is path not loop

        # context: (visited_time, after_violation, last_violation, precedence_matrix)
        after_violation, violation = context[1], context[2]  # (batch, seq_length)
        out_node_penalty = (violation > 0).sum(-1)  # Number of nodes with violations
        out_penalty = after_violation.max(dim=-1)[0]  # (batch,)  # Total violation count
        if penalty_normalize:
            # Normalize by problem size
            out_penalty = out_penalty / rec.size(1)
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

    def get_dynamic_feature(self, context, with_infsb_feature, tw_normalize=False, feature=None):
        visited_time, after_violation, violation, precedence_matrix = context

        batch_size, seq_length = after_violation.size()
        k = batch_size // self.node_xy.size(0)
        device = after_violation.device
        is_start_node= torch.zeros(batch_size, seq_length, device=device)
        is_start_node[:, 0] = 1.0
        is_end_node= torch.zeros(batch_size, seq_length, device=device)
        is_end_node[:, -1] = 1.0
        infeasibility_indicator_after_visit = violation > 0  # (batch, seq_length)
        
        # Dynamic features for each node at the time of visit
        # violation already contains unvisited predecessors count (computed in preprocessing)
        to_actor = torch.cat((
            after_violation.unsqueeze(-1),                    # Total violation after visiting node
            violation.unsqueeze(-1),                         # Violation when visiting node (unvisited predecessors count)
            is_start_node.unsqueeze(-1),                     # Whether node is the start node
            is_end_node.unsqueeze(-1),                      # Whether node is the end node
            infeasibility_indicator_after_visit.unsqueeze(-1).float(),  # Infeasibility indicator
        ), -1)  # (batch, seq_length, 5)
        supplement_feature = to_actor
        if not with_infsb_feature:
            supplement_feature = to_actor[:, :, :-1]
        node_feature = torch.cat((feature.repeat_interleave(k, 0), supplement_feature), dim=-1)

        return visited_time, None, node_feature

    def improvement_step(self, rec, action, obj, feasible_history, t, weights=0, out_reward=False, penalty_factor=1.,
                         penalty_normalize=False, improvement_method="kopt", insert_before=True,
                         epsilon=EPSILON_hardcoded, seperate_obj_penalty=False, non_linear=None, n2s_decoder=False):

        total_history = feasible_history.size(1)
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
        non_feasible_cost_total = out_penalty[0]
        feasible = non_feasible_cost_total <= 0.0
        soft_infeasible = (non_feasible_cost_total <= self.problem_size * EPSILON_hardcoded) & (non_feasible_cost_total > 0.)

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
                    feasible_history.mean(1, keepdims=True) if t > (total_history - 2) 
                    else feasible_history[:,:(t + 1)].mean(1,keepdims=True),
                # note: slightly different from N2S due to shorter improvement steps; before/after?
                ), 1)  # (batch_size, 4, solution_size)
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
        start_node, end_node = 0, self.problem_size - 1

        # action bs * (K_index, K_from, K_to)
        selected_index = action[:, :self.k_max]
        left = action[:, self.k_max:2 * self.k_max]
        right = action[:, 2 * self.k_max:]

        # Filter out edges involving start or end nodes
        # Cannot modify edges from/to start node (0) or end node (problem_size-1)
        valid_mask = (selected_index != start_node) & (selected_index != end_node) & \
                     (left != start_node) & (left != end_node) & \
                     (right != start_node) & (right != end_node)

        # prepare
        rec_next = rec.clone()
        right_nodes = rec.gather(1, selected_index)
        argsort = rec.argsort()

        # Only apply valid actions: for invalid edges, keep original target
        right_filtered = torch.where(valid_mask.unsqueeze(-1).expand_as(right), right, rec.gather(1, left))
        
        # new rec
        rec_next.scatter_(1, left, right_filtered)
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
        start_node, end_node = 0, self.problem_size - 1

        # Expand action to match dimensions
        remove_indices = action[:, ::2]  # Shape (B, rm_num)
        insert_indices = action[:, 1::2]  # Shape (B, rm_num)

        for i in range(rm_num):
            # Step 1: Find the position of the node to be removed
            remove_idx = remove_indices[:, i]  # Shape (B,)
            remove_mask = (updated_sol == remove_idx.unsqueeze(1))  # Shape (B, N), Boolean mask for nodes to be removed
            remove_pos = remove_mask.nonzero(as_tuple=True)[1]  # Shape (B,), indices of nodes to be removed

            # Skip if trying to remove start or end node, or from position 0 or last position
            valid_remove = (remove_idx != start_node) & (remove_idx != end_node)
            for b in range(batch_size):
                if not valid_remove[b] or remove_pos[b].item() == 0 or remove_pos[b].item() == num_nodes - 1:
                    continue

                # Step 2: Remove the node from the solution
                keep_mask = ~remove_mask[b]
                sol_without_removed = torch.masked_select(updated_sol[b], keep_mask).view(num_nodes - 1)

                # Step 3: Find the position to insert
                insert_idx = insert_indices[b, i]
                insert_mask = (sol_without_removed == insert_idx)
                insert_pos = insert_mask.nonzero(as_tuple=True)[0]
                
                if len(insert_pos) == 0 or insert_pos[0].item() == 0 or insert_pos[0].item() == num_nodes - 2:
                    continue
                
                pos = insert_pos[0].item()
                if insert_before:
                    new_sol = torch.cat((sol_without_removed[:pos], remove_idx[b:b+1], sol_without_removed[pos:]))
                else:
                    new_sol = torch.cat((sol_without_removed[:pos+1], remove_idx[b:b+1], sol_without_removed[pos+1:]))
                updated_sol[b] = new_sol

        return sol2rec(updated_sol.unsqueeze(1)).squeeze(1)

    def f(self, p):  # The entropy measure in Eq.(5)
        return torch.clamp(1 - 0.5 * torch.log2(2.5 * np.pi * np.e * p * (1 - p) + 1e-5), 0, 1)