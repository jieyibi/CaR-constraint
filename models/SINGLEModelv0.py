import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import dummify

__all__ = ['SINGLEModel']


class SINGLEModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = self.model_params['eval_type']
        self.problem = self.model_params['problem']
        embedding_dim = self.model_params['embedding_dim']

        self.encoder = SINGLE_Encoder(**model_params)
        self.decoder = SINGLE_Decoder(**model_params)
        if self.model_params["improve_steps"] > 0.:
            self.supplement_encoder = EmbeddingNet(**model_params)
            self.multi_view_aggr = nn.Linear(embedding_dim * 3, embedding_dim)
            self.kopt_decoder = kopt_Decoder(**model_params)

        if self.model_params["dual_decoder"]:
            self.feasible_decoder = SINGLE_Decoder(**model_params)

        self.encoded_nodes = None

        self.pattern = None

        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else \
            model_params['device']
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        if not self.problem.startswith('TSP'):
            depot_xy = reset_state.depot_xy
            # shape: (batch, 1, 2)
            node_demand = reset_state.node_demand
        else:
            depot_xy = None

        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)

        if self.problem in ["CVRP", "OVRP", "VRPB", "VRPL", "VRPBL", "OVRPB", "OVRPL", "OVRPBL"]:
            feature = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
            # shape: (batch, problem, 3)
        elif self.problem in ["TSPTW"]:
            node_tw_start = reset_state.node_tw_start
            node_tw_end = reset_state.node_tw_end
            # shape: (batch, problem)
            tw_start = node_tw_start[:, :, None]
            tw_end = node_tw_end[:, :, None]
            # _, problem_size = node_tw_end.size()
            if self.model_params["tw_normalize"]:
                tw_end_max = node_tw_end[:, :1, None]
                tw_start = tw_start / tw_end_max
                tw_end = tw_end / tw_end_max
            # tw_start =  (((node_tw_start[:, :, None] - node_tw_start[:, :, None].min(dim=1)[0].unsqueeze(1).repeat(1, problem_size, 1))
            #             / (node_tw_start[:, :, None].max(dim=1)[0].unsqueeze(1).repeat(1, problem_size, 1) - node_tw_start[:, :, None].min(dim=1)[0].unsqueeze(1).repeat(1, problem_size, 1) + 1e-6)))
            # tw_end = ((node_tw_end[:, :, None] - node_tw_end[:, :, None].min(dim=1)[0].unsqueeze(1).repeat(1, problem_size, 1))
            #           / (node_tw_end[:, :, None].max(dim=1)[0].unsqueeze(1).repeat(1, problem_size, 1) - node_tw_end[:, :, None].min(dim=1)[0].unsqueeze(1).repeat(1, problem_size, 1) + 1e-6))
            # if node_xy.max() > 1.:
            #     node_xy = node_xy / 100.
            feature = torch.cat((node_xy, tw_start, tw_end), dim=2)
            # shape: (batch, problem, 4)
        elif self.problem in ['TSPDL']:
            node_demand = reset_state.node_demand
            node_draft_limit = reset_state.node_draft_limit
            feature = torch.cat((node_xy, node_demand[:, :, None], node_draft_limit[:, :, None]), dim=2)
        elif self.problem in ["VRPTW", "OVRPTW", "VRPBTW", "VRPLTW", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
            node_tw_start = reset_state.node_tw_start
            node_tw_end = reset_state.node_tw_end
            # shape: (batch, problem)
            feature = torch.cat((node_xy, node_demand[:, :, None], node_tw_start[:, :, None], node_tw_end[:, :, None]),
                                dim=2)
            # shape: (batch, problem, 5)
        else:
            raise NotImplementedError

        # pdb.set_trace()
        self.encoded_nodes = self.encoder(depot_xy, feature)
        # shape: (batch, problem(+1), embedding)
        # pdb.set_trace()
        self.decoder.set_kv(self.encoded_nodes)
        if self.model_params["dual_decoder"]:
            self.feasible_decoder.set_kv(self.encoded_nodes)

        return self.encoded_nodes, feature

    def pre_forward1(self, features):
        # features.shape = (batch, problem, features)
        depot_xy = None
        # shape: (batch, problem, 2)
        if self.problem in ["TSPTW"] and self.model_params["tw_normalize"]:
            tw_end_max = features[:, :1, 3:].clone()
            features[:, :, 2:3] = features[:, :, 2:3] / tw_end_max
            features[:, :, 3:] = features[:, :, 3:] / tw_end_max

        self.encoded_nodes = self.encoder(depot_xy, features)
        # shape: (batch, problem(+1), embedding)

        return self.encoded_nodes, features

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type

    def forward(self, state, solver="construction", selected=None, pomo=False, candidate_feature=None,
                feasible_start=None, return_probs=False, require_entropy=False, fixed_action=None):

        if solver == "construction":
            batch_size = state.BATCH_IDX.size(0)
            pomo_size = state.BATCH_IDX.size(1)

            if state.selected_count == 0:  # First Move, depot
                selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long).to(self.device)
                prob = torch.ones(size=(batch_size, pomo_size))
                # probs = torch.ones(size=(batch_size, pomo_size, self.encoded_nodes.size(1)))
                # shape: (batch, pomo, problem_size+1)

                # # Use Averaged encoded nodes for decoder input_1
                # encoded_nodes_mean = self.encoded_nodes.mean(dim=1, keepdim=True)
                # # shape: (batch, 1, embedding)
                # self.decoder.set_q1(encoded_nodes_mean)

                # # Use encoded_depot for decoder input_2
                # encoded_first_node = self.encoded_nodes[:, [0], :]
                # # shape: (batch, 1, embedding)
                # self.decoder.set_q2(encoded_first_node)
            elif (
                    feasible_start is not None or pomo) and state.selected_count == 1 and pomo_size > 1:  # Second Move, POMO
                # selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size).to(self.device)
                if feasible_start is not None:
                    # 找到每个batch中值为0的索引
                    batch_indices, value_indices = (feasible_start[:, 0, :] == 0).nonzero(as_tuple=True)
                    # 准备输出tensor
                    selected = torch.zeros((batch_size, pomo_size), dtype=torch.long)
                    # 对每个batch进行随机抽样
                    for i in range(batch_size):
                        # 获取当前batch中值为0的索引
                        current_batch_indices = value_indices[batch_indices == i]
                        if len(current_batch_indices) > 0:  # 如果当前batch中存在值为0的情况
                            # 有放回地随机抽取sample_size个索引
                            sampled_indices = current_batch_indices[
                                torch.randint(len(current_batch_indices), (pomo_size,))]
                            selected[i] = sampled_indices
                        else:
                            assert 0, "Warning! Infeasible instance appears!"
                    state.START_NODE = selected
                else:
                    selected = state.START_NODE
                prob = torch.ones(size=(batch_size, pomo_size))
            else:
                encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
                # shape: (batch, pomo, embedding)
                attr = self._get_extra_features(state, candidate_feature)

                ninf_mask = state.ninf_mask

                if self.model_params["dual_decoder"]:
                    probs1 = self.decoder(encoded_last_node, attr, ninf_mask=ninf_mask)
                    probs2 = self.feasible_decoder(encoded_last_node, attr, ninf_mask=ninf_mask)
                    probs = self.model_params["fsb_decoder_weight"] * probs2 + (
                            1 - self.model_params["fsb_decoder_weight"]) * probs1
                else:
                    probs = self.decoder(encoded_last_node, attr, ninf_mask=ninf_mask, return_probs=return_probs)
                    if return_probs:
                        probs, probs_return = probs
                # shape: (batch, pomo, problem+1)
                if selected is None:
                    while True:
                        if self.training or self.eval_type == 'softmax':
                            try:
                                selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(
                                    dim=1).reshape(batch_size, pomo_size)
                            except Exception as exception:
                                torch.save(probs, "prob.pt")
                                print(">> Catch Exception: {}, on the instances of {}".format(exception, state.PROBLEM))
                                exit(0)
                        else:
                            selected = probs.argmax(dim=2)
                        prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                        # shape: (batch, pomo)
                        if (prob != 0).all():
                            break
                else:
                    selected = selected
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

            if self.model_params["dual_decoder"]:
                try:
                    prob1 = probs1[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    prob2 = probs2[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                except:
                    prob1 = torch.ones(size=(batch_size, pomo_size))
                    prob2 = torch.ones(size=(batch_size, pomo_size))
                return selected, [prob1, prob2]
            if return_probs:
                try:
                    return selected, prob, probs_return
                except:
                    return selected, prob, None
            return selected, prob, None

        elif solver == "improvement":
            env, rec, context, context2, action = state
            batch_size, solution_size = rec.size()

            node_embedding = dummify(self.encoded_nodes, env.dummy_size - 1)
            node_embedding = node_embedding.repeat_interleave(batch_size // node_embedding.size(0), 0)

            # supplementary node features based on current solution
            if self.problem == 'CVRP':
                visited_time, supplement_feature = env.get_dynamic_feature(context)
                if not self.model_params["with_infsb_feature"]:
                    supplement_feature = supplement_feature[:, :, :-2]
                node_supplement_embedding = self.supplement_encoder(supplement_feature)
                del context
            # elif self.problem == 'TSP':
            #     visited_time = env.get_order(rec, return_solution=False)
            else:
                raise NotImplementedError()

            # positional features based on current solution
            self.pattern = self.cyclic_position_encoding_pattern(solution_size, self.model_params['embedding_dim'])
            h_pos = self.position_encoding(self.pattern, self.model_params['embedding_dim'], visited_time)
            # shape:(batch_size, problem_size+dummy_size, embedding_dim)

            # merge three embeddings
            h_em_final = torch.cat([node_embedding, node_supplement_embedding, h_pos], -1)
            h_em_final = self.multi_view_aggr(h_em_final)
            # shape:(batch_size, problem_size+dummy_size, embedding_dim)

            action, log_ll, entropy = self.kopt_decoder(h_em_final, rec, context2, visited_time, action,
                                                        fixed_action=fixed_action, require_entropy=require_entropy)

            # assert (visited_time == visited_time_clone).all()
            if require_entropy:
                return action, log_ll, entropy
            else:
                return action, log_ll

    def _get_extra_features(self, state, candidate_feature=None):

        if self.problem in ["CVRP"]:
            attr = state.load[:, :, None]
        elif self.problem in ["VRPB", 'TSPDL']:
            attr = state.load[:, :, None]  # shape: (batch, pomo, 1)
        elif self.problem in ["TSPTW"]:
            attr = state.current_time[:, :, None]  # shape: (batch, pomo, 1)
            if self.model_params["tw_normalize"]:
                attr = attr / candidate_feature[:, 0][:, None, None]
        elif self.problem in ["OVRP", "OVRPB"]:
            attr = torch.cat((state.load[:, :, None], state.open[:, :, None]), dim=2)  # shape: (batch, pomo, 2)
        elif self.problem in ["VRPTW", "VRPBTW"]:
            if self.model_params["extra_feature"]:
                attr = torch.cat(
                    (state.load[:, :, None], state.current_time[:, :, None], state.vehicle_remaining[:, :, None]),
                    dim=2)
            else:
                attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None]),
                                 dim=2)  # shape: (batch, pomo, 2)
        elif self.problem in ["VRPL", "VRPBL"]:
            attr = torch.cat((state.load[:, :, None], state.length[:, :, None]), dim=2)  # shape: (batch, pomo, 2)
        elif self.problem in ["VRPLTW", "VRPBLTW"]:
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.length[:, :, None]),
                             dim=2)  # shape: (batch, pomo, 3)
        elif self.problem in ["OVRPL", "OVRPBL"]:
            attr = torch.cat((state.load[:, :, None], state.length[:, :, None], state.open[:, :, None]),
                             dim=2)  # shape: (batch, pomo, 3)
        elif self.problem in ["OVRPTW", "OVRPBTW"]:
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.open[:, :, None]),
                             dim=2)  # shape: (batch, pomo, 3)
        elif self.problem in ["OVRPLTW", "OVRPBLTW"]:
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.length[:, :, None],
                              state.open[:, :, None]), dim=2)  # shape: (batch, pomo, 4)
        else:
            raise NotImplementedError

        return attr

    def basesin(self, x, T, fai=0):
        return np.sin(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)

    def basecos(self, x, T, fai=0):
        return np.cos(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)

    def cyclic_position_encoding_pattern(self, n_position, emb_dim, mean_pooling=True):

        Td_set = np.linspace(np.power(n_position, 1 / (emb_dim // 2)), n_position, emb_dim // 2, dtype='int')
        x = np.zeros((n_position, emb_dim))

        for i in range(emb_dim):
            Td = Td_set[i // 3 * 3 + 1] if (i // 3 * 3 + 1) < (emb_dim // 2) else Td_set[-1]
            fai = 0 if i <= (emb_dim // 2) else 2 * np.pi * ((-i + (emb_dim // 2)) / (emb_dim // 2))
            longer_pattern = np.arange(0, np.ceil((n_position) / Td) * Td, 0.01)
            if i % 2 == 1:
                x[:, i] = self.basecos(longer_pattern, Td, fai)[
                    np.linspace(0, len(longer_pattern), n_position, dtype='int', endpoint=False)]
            else:
                x[:, i] = self.basesin(longer_pattern, Td, fai)[
                    np.linspace(0, len(longer_pattern), n_position, dtype='int', endpoint=False)]

        pattern = torch.from_numpy(x).type(torch.FloatTensor)
        pattern_sum = torch.zeros_like(pattern).cpu()

        # averaging the adjacient embeddings if needed (optional, almost the same performance)
        arange = torch.arange(n_position).cpu()
        pooling = [0] if not mean_pooling else [-2, -1, 0, 1, 2]
        time = 0
        for i in pooling:
            time += 1
            index = (arange + i + n_position) % n_position
            pattern_sum += pattern.gather(0, index.view(-1, 1).expand_as(pattern))
        pattern = 1. / time * pattern_sum - pattern.mean(0)

        return pattern

    def position_encoding(self, base, embedding_dim, order_vector):
        batch_size, seq_length = order_vector.size()

        # expand for every batch
        position_enc = base.expand(batch_size, *base.size()).clone().to(order_vector.device)

        # get index according to the solutions
        index = order_vector.unsqueeze(-1).expand(batch_size, seq_length, embedding_dim)

        # return
        return torch.gather(position_enc, 1, index)


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class SINGLE_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        if not self.problem.startswith("TSP"):
            self.embedding_depot = nn.Linear(2, embedding_dim)
        if self.problem in ["CVRP", "OVRP", "VRPB", "VRPL", "VRPBL", "OVRPB", "OVRPL", "OVRPBL"]:
            self.embedding_node = nn.Linear(3, embedding_dim)
        elif self.problem in ["TSPTW", "TSPDL"]:
            self.embedding_node = nn.Linear(4, embedding_dim)
        elif self.problem in ["VRPTW", "OVRPTW", "VRPBTW", "VRPLTW", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
            self.embedding_node = nn.Linear(5, embedding_dim)
        else:
            raise NotImplementedError
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand_tw):
        # depot_xy.shape: (batch, 1, 2) if self.problem is CVRP variants
        # node_xy_demand_tw.shape: (batch, problem, 3/4/5) - based on self.problem
        if depot_xy is not None:
            embedded_depot = self.embedding_depot(depot_xy)
            # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand_tw)
        # shape: (batch, problem, embedding)

        if depot_xy is not None:
            out = torch.cat((embedded_depot, embedded_node), dim=1)
            # shape: (batch, problem+1, embedding)
        else:
            out = embedded_node
            # shape: (batch, problem, embedding)

        for layer in self.layers:
            out = layer(out)

        return out
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        """
        Two implementations:
            norm_last: the original implementation of AM/POMO: MHA -> Add & Norm -> FFN/MOE -> Add & Norm
            norm_first: the convention in NLP: Norm -> MHA -> Add -> Norm -> FFN/MOE -> Add
        """
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        if self.model_params['norm_loc'] == "norm_last":
            out_concat = multi_head_attention(q, k, v)  # (batch, problem, HEAD_NUM*KEY_DIM)
            multi_head_out = self.multi_head_combine(out_concat)  # (batch, problem, EMBEDDING_DIM)
            out1 = self.addAndNormalization1(input1, multi_head_out)
            out2 = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)  # (batch, problem, EMBEDDING_DIM)
        else:
            out1 = self.addAndNormalization1(None, input1)
            multi_head_out = self.multi_head_combine(out1)
            input2 = input1 + multi_head_out
            out2 = self.addAndNormalization2(None, input2)
            out2 = self.feedForward(out2)
            out3 = input2 + out2

        return out3


class EmbeddingNet(nn.Module):

    def __init__(self, **model_params):
        super(EmbeddingNet, self).__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']
        embedding_dim = self.model_params['embedding_dim']
        supplement_feature_dim = self.model_params['supplement_feature_dim']

        self.embedder = nn.Sequential(
            nn.Linear(supplement_feature_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim // 2, embedding_dim))

        # self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, supplement_feature):
        return self.embedder(supplement_feature)


class MultiHeadPosCompat(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadPosCompat, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        # self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h):

        batch_size, graph_size, input_dim = h.size()
        posflat = h.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(posflat, self.W_query).view(shp)
        K = torch.matmul(posflat, self.W_key).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        return torch.matmul(Q, K.transpose(2, 3))


########################################
# DECODER
########################################

class SINGLE_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        if self.problem == "CVRP":
            self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        elif self.problem in ["VRPB", "TSPTW", "TSPDL"]:
            self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        elif self.problem in ["OVRP", "OVRPB", "VRPTW", "VRPBTW", "VRPL", "VRPBL"]:
            attr_num = 3 if self.model_params["extra_feature"] else 2
            self.Wq_last = nn.Linear(embedding_dim + attr_num, head_num * qkv_dim, bias=False)
        elif self.problem in ["VRPLTW", "VRPBLTW", "OVRPL", "OVRPBL", "OVRPTW", "OVRPBTW"]:
            self.Wq_last = nn.Linear(embedding_dim + 3, head_num * qkv_dim, bias=False)
        elif self.problem in ["OVRPLTW", "OVRPBLTW"]:
            self.Wq_last = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        else:
            raise NotImplementedError

        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head_attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head_attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, attr, ninf_mask, return_probs=False):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, 1~4)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, attr), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)  # TODO: CONFLICTS WITH ISL
        # shape: (batch, pomo, head_num*qkv_dim)
        mh_atten_out = self.multi_head_combine(out_concat)  # TODO
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        if return_probs:
            out_concat0 = multi_head_attention(q, self.k, self.v)
            mh_atten_out0 = self.multi_head_combine(out_concat0)
            score0 = torch.matmul(mh_atten_out0, self.single_head_key)
            score_scaled0 = score0 / sqrt_embedding_dim
            score_clipped0 = logit_clipping * torch.tanh(score_scaled0)
            probs_return = F.softmax(score_clipped0, dim=2)
            return probs, probs_return
        return probs


class kopt_Decoder(nn.Module):
    def __init__(self, **model_params):
        super(kopt_Decoder, self).__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']
        self.embedding_dim = self.model_params['embedding_dim']
        self.with_RNN = self.model_params['with_RNN']
        self.with_explore_stat_feature = self.model_params['with_explore_stat_feature']
        self.k_max = self.model_params['k_max']

        self.linear_K1 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.linear_K2 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.linear_K3 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.linear_K4 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.linear_Q1 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.linear_Q2 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.linear_Q3 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.linear_Q4 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        if self.with_explore_stat_feature:
            self.meta_linear = nn.Sequential(
                nn.Linear(9, 8),
                nn.ReLU(inplace=True),
                nn.Linear(8, self.embedding_dim * 2))
        else:
            self.linear_V1 = nn.Parameter(torch.Tensor(self.embedding_dim))
            self.linear_V2 = nn.Parameter(torch.Tensor(self.embedding_dim))

        if self.with_RNN:
            self.init_hidden_W = nn.Linear(self.embedding_dim, self.embedding_dim)
            self.init_query_learnable = nn.Parameter(torch.Tensor(self.embedding_dim))
            self.rnn1 = nn.GRUCell(self.embedding_dim, self.embedding_dim)
            self.rnn2 = nn.GRUCell(self.embedding_dim, self.embedding_dim)
        else:
            self.init_query_learnable = nn.Parameter(torch.Tensor(self.embedding_dim))

        # self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h, rec, context2, visited_time, last_action, fixed_action=None, require_entropy=False):
        # input: h_em_final, rec, context2, visited_time, action
        # h: comprehensive embeddings of each node, shape: (batch, dummy_graph_size, embedding_dim)
        # rec: solution in linked list, shape: (batch, dummy_graph_size)
        # context2: exploration statistics, shape: (batch, 9)
        # visited_time: visited time of each node, shape: (batch, dummy_graph_size)
        # last_action, shape: (batch, dummy_graph_size)

        bs, gs, _, ll, action, entropys = *h.size(), 0.0, None, []
        action_index = torch.zeros(bs, self.k_max, dtype=torch.long).to(rec.device)
        k_action_left = torch.zeros(bs, self.k_max + 1, dtype=torch.long).to(rec.device)
        k_action_right = torch.zeros(bs, self.k_max, dtype=torch.long).to(rec.device)
        next_of_last_action = torch.zeros_like(rec[:, :1], dtype=torch.long).to(rec.device) - 1
        mask = torch.zeros_like(rec, dtype=torch.bool).to(rec.device)
        stopped = torch.ones(bs, dtype=torch.bool).to(rec.device)
        h_mean = h.mean(1)
        init_query = self.init_query_learnable.repeat(bs, 1)
        input_q1 = input_q2 = init_query.clone()

        if self.with_RNN:
            init_hidden = self.init_hidden_W(h_mean)
            q1 = q2 = init_hidden.clone()

        if self.with_explore_stat_feature:
            decoder_condition = context2
            linear_V = self.meta_linear(decoder_condition)
            linear_V1 = linear_V[:, :self.embedding_dim]
            linear_V2 = linear_V[:, self.embedding_dim:]
        else:
            linear_V1 = self.linear_V1.view(1, -1).expand(bs, -1)
            linear_V2 = self.linear_V2.view(1, -1).expand(bs, -1)

        for i in range(self.k_max):

            # GRUs
            if self.with_RNN:
                q1 = self.rnn1(input_q1, q1)
                q2 = self.rnn2(input_q2, q2)
            else:
                q1 = input_q1
                q2 = input_q2

            # Dual-Stream Attention
            result = (linear_V1.unsqueeze(1) * torch.tanh(self.linear_K1(h) +
                                                          self.linear_Q1(q1).unsqueeze(1) +
                                                          self.linear_K3(h) * self.linear_Q3(q1).unsqueeze(1)
                                                          )).sum(-1)  # \mu stream
            result += (linear_V2.unsqueeze(1) * torch.tanh(self.linear_K2(h) +
                                                           self.linear_Q2(q2).unsqueeze(1) +
                                                           self.linear_K4(h) * self.linear_Q4(q2).unsqueeze(1)
                                                           )).sum(-1)  # \lambda stream

            # Calc probs
            logits = torch.tanh(result) * self.model_params['logit_clipping']
            # assert (~mask).any(-1).all(), (i, (~mask).any(-1))
            logits[mask.clone()] = -1e30
            if i == 0 and isinstance(last_action, torch.Tensor):
                logits.scatter_(1, last_action[:, :1], -1e30)
            probs = F.softmax(logits, dim=-1)

            # Sample action for a_i
            if fixed_action is None:
                action = probs.multinomial(1)
                value_max, action_max = probs.max(-1, True)  ### fix bug of pytorch
                action = torch.where(1 - value_max.view(-1, 1) < 1e-5, action_max.view(-1, 1),
                                     action)  ### fix bug of pytorch
            else:
                action = fixed_action[:, i:i + 1]

            if i > 0:
                action = torch.where(stopped.unsqueeze(-1), action_index[:, :1], action)

            # Record log_likelihood and Entropy
            if self.training:
                loss_now = F.log_softmax(logits, dim=-1).gather(-1, action).squeeze()
                if i > 0:
                    ll = ll + torch.where(stopped, loss_now * 0, loss_now)
                else:
                    ll = ll + loss_now
                if require_entropy:
                    dist = Categorical(probs, validate_args=False)
                    entropys.append(dist.entropy())

                    # Store and Process actions
            next_of_new_action = rec.gather(1, action)
            action_index[:, i] = action.squeeze().clone()
            k_action_left[stopped, i] = action[stopped].squeeze().clone()
            k_action_right[~stopped, i - 1] = action[~stopped].squeeze().clone()
            k_action_left[:, i + 1] = next_of_new_action.squeeze().clone()

            # Prepare next RNN input
            input_q1 = h.gather(1, action.view(bs, 1, 1).expand(bs, 1, self.embedding_dim)).squeeze(1)
            input_q2 = torch.where(stopped.view(bs, 1).expand(bs, self.embedding_dim), input_q1.clone(),
                                   h.gather(1, (next_of_last_action % gs).view(bs, 1, 1).expand(bs, 1,
                                                                                                self.embedding_dim)).squeeze(
                                       1))

            # Process if k-opt close
            # assert (input_q1[stopped] == input_q2[stopped]).all()
            if i > 0:
                stopped = stopped | (action == next_of_last_action).squeeze()
            else:
                stopped = (action == next_of_last_action).squeeze()
            # assert (input_q1[stopped] == input_q2[stopped]).all()

            k_action_left[stopped, i] = k_action_left[stopped, i - 1]
            k_action_right[stopped, i] = k_action_right[stopped, i - 1]

            # Calc next basic masks
            if i == 0:
                visited_time_tag = (visited_time - visited_time.gather(1, action)) % gs
            mask &= False
            mask[(visited_time_tag <= visited_time_tag.gather(1, action))] = True
            if i == 0:
                mask[visited_time_tag > (gs - 2)] = True
            mask[stopped, action[stopped].squeeze()] = False  # allow next k-opt starts immediately
            # if True:#i == self.k_max - 2: # allow special case: close k-opt at the first selected node
            index_allow_first_node = (~stopped) & (next_of_new_action.squeeze() == action_index[:, 0])
            mask[index_allow_first_node, action_index[index_allow_first_node, 0]] = False

            # Move to next
            next_of_last_action = next_of_new_action
            next_of_last_action[stopped] = -1

        # Form final action
        k_action_right[~stopped, -1] = k_action_left[~stopped, -1].clone()
        k_action_left = k_action_left[:, :self.k_max]
        action_all = torch.cat((action_index, k_action_left, k_action_right), -1)

        return action_all, ll, torch.stack(entropys).mean(0) if require_entropy and self.training else None


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.add = True if 'norm_loc' in model_params.keys() and model_params['norm_loc'] == "norm_last" else False
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1=None, input2=None):
        # input.shape: (batch, problem, embedding)
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2 if self.add else input2
            transposed = added.transpose(1, 2)
            # shape: (batch, embedding, problem)
            normalized = self.norm(transposed)
            # shape: (batch, embedding, problem)
            back_trans = normalized.transpose(1, 2)
            # shape: (batch, problem, embedding)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2 if self.add else input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.LayerNorm):
            added = input1 + input2 if self.add else input2
            back_trans = self.norm(added)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2 if self.add else self.norm * input2
        else:
            back_trans = input1 + input2 if self.add else input2

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


class FC(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.W1 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input):
        # input.shape: (batch, problem, embedding)
        return F.relu(self.W1(input))
