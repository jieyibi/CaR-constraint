import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np
from utils import loss_edges, clip_grad_norms
import time
import pdb
__all__ = ['SafetyLayer']


class SafetyLayer(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']

        self.encoder = Encoder(**model_params)
        self.decoder = Decoder(**model_params)

        self.encoded_nodes = None
        self.tw_end_max = None

        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else model_params['device']
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, features):
        # features.shape = (batch, problem, features)
        depot_xy = None
        # shape: (batch, problem, 2)
        if self.problem in ["TSPTW"] and self.model_params["tw_normalize"]:
            if self.model_params["constraint_related_feature_only"]:
                tw_end_max = features[:,:1,1:].clone()
                features = features / tw_end_max.repeat(1,1,2)
            else:
                tw_end_max = features[:,:1,3:].clone()
                features[:,:,2:3] = features[:,:,2:3] / tw_end_max
                features[:,:,3:] = features[:,:,3:] / tw_end_max
            self.tw_end_max = tw_end_max

        self.encoded_nodes = self.encoder(depot_xy, features)
        # shape: (batch, problem(+1), embedding)

        return self.encoded_nodes, features


    def forward(self, partial_solution_embedding, unvisited_nodes_embedding_avg, current_time, no_sigmoid=True):
        # shape of input:
        # partial_solution_embedding.shape = (batch, 1, embedding)
        # unvisited_nodes_embedding_avg.shape = (batch, 1, embedding)
        # current_time.shape = (batch, 1, 1)
        # candidate_embedding.shape = (batch, 1, embedding)

        # current_time = current_time.expand_as(partial_solution_embedding)
        # query = torch.cat([partial_solution_embedding, current_time], dim=-1)
        if self.tw_end_max:
            current_time = current_time / self.tw_end_max
        query = torch.cat([partial_solution_embedding, unvisited_nodes_embedding_avg, current_time], dim=-1)

        probs = self.decoder(query, no_sigmoid)
        # shape: (batch, 1, 1)

        return probs

    def predict_infsb_mask(self, features, unvisited, current_node, current):
        # features.shape = (batch*pomo, candidate_num, features)
        # unvisited.shape = (batch*pomo, unvisited_nodes_len)
        # current_node.shape = (batch*pomo, 1)
        # current_node.shape = (batch*pomo, 1)

        # set in the evaluation mode
        self.eval()

        # half positive samples and half negative samples
        _, features_batch_norm = self.pre_forward(features)  # get node embedding
        candidate_num = features_batch_norm.size(1)

        # 计算特征的累积和，用于之后计算未访问节点的特征平均值
        gathering_idx = unvisited[:, :, None].expand(-1, -1, self.model_params["embedding_dim"])
        unvisited_nodes_embedding = self.encoded_nodes.gather(dim=1, index=gathering_idx).mean(dim=1, keepdims=True)
        unvisited_nodes_embedding = unvisited_nodes_embedding.repeat(1, candidate_num, 1).reshape(-1, 1, self.model_params["embedding_dim"])
        # (batch*pomo*candidate_num, 1, embedding_dim)
        gathering_idx = current_node[:, :, None].expand(-1, -1, self.model_params["embedding_dim"])
        current_node_embedding = self.encoded_nodes.gather(dim=1, index=gathering_idx)
        current_node_embedding = current_node_embedding.repeat(1, candidate_num, 1).reshape(-1, 1, self.model_params["embedding_dim"])
        # (batch*pomo*candidate_num, 1, embedding_dim)

        if self.model_params["constraint_related_feature_only"]:
            candidate_tw_end = features_batch_norm[:, :, 1:].reshape(-1,1,1)
        else:
            candidate_tw_end = features_batch_norm[:, :, 3:].reshape(-1, 1, 1)
        # (batch*pomo*candidate_num, 1, 1)
        candidate_embedding = self.encoded_nodes.reshape(-1,1,self.model_params["embedding_dim"])
        # (batch*pomo,*candidate_num, 1, embedding_dim)
        candidate_embedding = torch.cat([candidate_embedding, candidate_tw_end], dim=-1)
        # (batch*pomo*candidate_num, 1, embedding_dim+1)

        self.decoder.set_kv(candidate_embedding)

        predict_out = self(partial_solution_embedding=current_node_embedding,
                           unvisited_nodes_embedding_avg= unvisited_nodes_embedding,
                           current_time=current[:,None,:].repeat(1, candidate_num, 1).reshape(-1, 1, 1), no_sigmoid=False)
        predict_out = predict_out.reshape(-1, candidate_num, 1)

        return predict_out


class SafetyLayer_LEHD(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']
        embedding_dim = self.model_params['embedding_dim']

        # encoder: linear projection
        if not self.problem.startswith("TSP"):
            self.encoder = nn.Linear(2, embedding_dim)
        if self.problem in ["CVRP", "OVRP", "VRPB", "VRPL", "VRPBL", "OVRPB", "OVRPL", "OVRPBL"]:
            self.encoder = nn.Linear(3, embedding_dim)
        elif self.problem in ["TSPTW", "TSPDL"]:
            if self.model_params["constraint_related_feature_only"]:
                self.encoder = nn.Linear(2, embedding_dim)
            else:
                self.encoder = nn.Linear(4, embedding_dim)
        elif self.problem in ["VRPTW", "OVRPTW", "VRPBTW", "VRPLTW", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
            self.encoder = nn.Linear(5, embedding_dim)
        else:
            raise NotImplementedError

        # decoder: Transformer
        self.decoder = HeavyDecoder(**model_params)

        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else model_params['device']
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def forward(self, features, unvisited, current_node, current, no_sigmoid=False):
        # features.shape = (batch*pomo, candidate_num, features)
        # unvisited.shape = (batch*pomo, unvisited_nodes_len)
        # current_node.shape = (batch*pomo, 1)
        # current_node.shape = (batch*pomo, 1)

        # half positive samples and half negative samples
        # shape: (batch*pomo, problem, 2)
        if self.problem in ["TSPTW"] and self.model_params["tw_normalize"]:
            if self.model_params["constraint_related_feature_only"]:
                tw_end_max = features[:,:1,1:].clone()
                features = features / tw_end_max
            else:
                tw_end_max = features[:,:1,3:].clone()
                features[:,:,2:3] = features[:,:,2:3] / tw_end_max
                features[:,:,3:] = features[:,:,3:] / tw_end_max
            current = current / tw_end_max.reshape(-1, 1)

        encoded_nodes = self.encoder(features)# get node embedding
        # shape: (batch*pomo, candidate_num, embedding)
        candidate_num = features.size(1)

        # 计算特征的累积和，用于之后计算未访问节点的特征平均值
        unvisited = unvisited[:, :, None].expand(-1, -1, self.model_params["embedding_dim"])
        unvisited_nodes_embedding = encoded_nodes[~unvisited].reshape(features.size(0), -1, self.model_params["embedding_dim"]).mean(dim=1, keepdims=True)
        # (batch*pomo, 1, embedding_dim)
        gathering_idx = current_node[:, :, None].expand(-1, -1, self.model_params["embedding_dim"])
        current_node_embedding = encoded_nodes.gather(dim=1, index=gathering_idx)
        # (batch*pomo, 1, embedding_dim)

        candidate_embedding = encoded_nodes.reshape(-1,candidate_num,self.model_params["embedding_dim"])
        # (batch*pomo, candidate_num, embedding_dim)
        if self.model_params["add_candidate_feature"]:
            if self.model_params["constraint_related_feature_only"]:
                candidate_tw_end = features[:, :, 1:].reshape(-1, candidate_num, 1)
            else:
                candidate_tw_end = features[:, :, 3:].reshape(-1, candidate_num, 1)
            # (batch, candidate_num, 1)
            candidate_embedding = torch.cat([candidate_embedding, candidate_tw_end], dim=-1)
            # (batch, candidate_num, embedding_dim+1)

        predict_out = self.decoder(partial_solution_embedding=current_node_embedding,
                                   unvisited_nodes_embedding_avg= unvisited_nodes_embedding,
                                   current_time=current[:, :, None],
                                   candidate_embedding = candidate_embedding,
                                   no_sigmoid=no_sigmoid)
        # shape: (batch*pomo, n_candidate +1 (or+3), 1)

        start_candidate_idx = 1 if self.model_params['aggregate_current_route'] else 3
        predict_out = predict_out[:,start_candidate_idx:,:]
        # shape: (batch*pomo, n_candidate, 1)
        return predict_out




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

class Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['sl_encoder_layer_num']

        if not self.problem.startswith("TSP"):
            self.embedding_depot = nn.Linear(2, embedding_dim)
        if self.problem in ["CVRP", "OVRP", "VRPB", "VRPL", "VRPBL", "OVRPB", "OVRPL", "OVRPBL"]:
            self.embedding_node = nn.Linear(3, embedding_dim)
        elif self.problem in ["TSPTW", "TSPDL"]:
            if self.model_params["constraint_related_feature_only"]:
                self.embedding_node = nn.Linear(2, embedding_dim)
            else:
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
        # shape: (batch, problem, embedding)


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

########################################
# DECODER
########################################

class Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        decoder_layer_num = self.model_params['decoder_layer_num']

        self.Wq_last = nn.Linear(embedding_dim *2+1, head_num * qkv_dim, bias=False)
        self.W_candidate = nn.Linear(embedding_dim +1, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

        if decoder_layer_num > 1:
            self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(decoder_layer_num)])
        else:
            self.layers=None

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        if encoded_nodes.size(-1) != self.model_params["embedding_dim"]:
            encoded_nodes = self.W_candidate(encoded_nodes)
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def forward(self, query, no_sigmoid=True):
        # query.shape: (batch, embedding*3)
        # candidate.shape: (batch, embedding)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        ######################################################
        q_last = reshape_by_heads(self.Wq_last(query), head_num=head_num)

        out_concat = multi_head_attention(q_last, self.k, self.v)
        # shape: (batch, 1, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, 1, embedding)

        if self.layers:
            out = mh_atten_out
            for layer in self.layers:
                out = layer(out)
            mh_atten_out = out

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, 1, 1)

        probs = F.sigmoid(score) if not no_sigmoid else score
        # shape: (batch, 1, 1)

        return probs

class HeavyDecoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']
        embedding_dim = self.model_params['embedding_dim']
        decoder_layer_num = self.model_params['sl_decoder_layer_num']

        if self.model_params['aggregate_current_route']:
            self.W_current_route = nn.Linear(embedding_dim*2+1, embedding_dim, bias=False)
        if self.model_params["add_candidate_feature"]:
            self.W_candidate = nn.Linear(embedding_dim + 1, embedding_dim, bias=False)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(decoder_layer_num)])

        self.W_output = nn.Linear(embedding_dim, 1, bias=False)


    def forward(self, partial_solution_embedding, unvisited_nodes_embedding_avg, current_time, candidate_embedding, no_sigmoid=False):
        # shape of input:
        # partial_solution_embedding.shape = (batch, 1, embedding)
        # unvisited_nodes_embedding_avg.shape = (batch, 1, embedding)
        # current_time.shape = (batch, 1, 1)
        # candidate_embedding.shape = (batch, n_candidate, embedding)

        if self.model_params['aggregate_current_route']:
            current_route = torch.cat([partial_solution_embedding, unvisited_nodes_embedding_avg, current_time], dim=2)
            current_route = self.W_current_route(current_route)
            # shape = (batch, 1, embedding)
        else:
            current_time = current_time.expand_as(partial_solution_embedding)
            current_route = torch.cat([partial_solution_embedding, unvisited_nodes_embedding_avg, current_time], dim=1)
            # shape = (batch, 3, embedding)

        if self.model_params["add_candidate_feature"]:
            candidate_embedding = self.W_candidate(candidate_embedding)

        out = torch.concat([current_route, candidate_embedding], dim=1)
        # shape = (batch, n_candidate +1 (or+3), embedding)
        for layer in self.layers:
            out = layer(out)
        # shape: (batch, n_candidate +1 (or+3), embedding)

        out = self.W_output(out)
        # shape: (batch, n_candidate +1 (or+3), 1)

        if not no_sigmoid:
            out = nn.Sigmoid()(out)
            # shape: (batch, n_candidate +1 (or+3), 1)

        return out


class DecoderLayer(nn.Module):
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

