import torch
import torch.nn as nn


def explicit_broadcast(this, other):
    for _ in range(this.dim(), other.dim()):
        this = this.unsqueeze(-1)

    return this.expand_as(other)


class GATLayer(nn.Module):
    nodes_dim = 0
    heads_dim = 1

    src_nodes_dim = 0
    trg_nodes_dim = 1

    """
    num_in_features: FIN
    num_of_heads: NH
    num_out_features: FOUT
    """

    def __init__(self, num_in_features, num_of_heads, num_out_features, dropout_prob, activation, bias, concat,
                 add_skip_connection):
        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.bias = bias
        self.concat = concat
        self.add_skip_connection = add_skip_connection

        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.activation = activation

        if bias:
            if concat:
                self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
            else:
                self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.linear_proj.weight, a=0.2)
        nn.init.kaiming_uniform_(self.scoring_fn_source, a=0.2)
        nn.init.kaiming_uniform_(self.scoring_fn_target, a=0.2)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, data):
        # in_nodes_features shape = (N, FIN)
        # edges shape = (2, E), E为图中所包含的边数
        in_nodes_features, edges = data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]

        in_nodes_features = self.dropout(in_nodes_features)
        # (N, FIN) -> (N, NH * FOUT) -> (N, NH, FOUT)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)

        # (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH)
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # scores_source_lifted, scores_target_lifted shape = (E, NH)
        # nodes_features_proj_lifted shape = (E, NH, FOUT)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target,
                                                                                           nodes_features_proj, edges)
        # shape = (E, NH)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # attentions_per_edge shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edges[self.trg_nodes_dim], num_of_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)

        # (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT)
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # out_nodes_features shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edges, in_nodes_features,
                                                      num_of_nodes)

        out_nodes_features = self.skip_concat_bias(in_nodes_features, out_nodes_features)
        return out_nodes_features, edges

    def lift(self, scores_source, scores_target, nodes_features_proj, edges):
        src_nodes_index = edges[self.src_nodes_dim]
        trg_nodes_index = edges[self.trg_nodes_dim]

        scores_source_lifted = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target_lifted = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_proj_lifted = nodes_features_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()

        # exp_scores_per_edge shape = (E, NH)
        neigborhood_aware_denominator = self.neighborhood_aware_sum(exp_scores_per_edge, trg_index, num_of_nodes)
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # (E, NH) -> (E, NH, 1)
        return attentions_per_edge.unsqueeze(-1)

    def neighborhood_aware_sum(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # 通过广播原理将 trg_index shape 从 E -> (E, NH)
        trg_index_broadcasted = explicit_broadcast(trg_index, exp_scores_per_edge)

        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        # neighborhood_sums shape = (N, NH)
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edges, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes
        # out_nodes_features shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = explicit_broadcast(edges[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)

        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def skip_concat_bias(self, in_nodes_features, out_nodes_features):
        if self.add_skip_connection:
            # FIN == FOUT
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
                # in_nodes_features shape (N, FIN) -> (N, 1, FIN)
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # in_nodes_features shape (N, FIN) -> (N, NH, FOUT)
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # (N, NH, FOUT) -> (N, NH * FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.heads_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)
