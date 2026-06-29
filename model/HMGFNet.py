import torch.nn as nn
import torch
from .mlp import MultiLayerPerceptron, GraphMLP
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HMGFNet(nn.Module):

    def __init__(
            self,
            num_nodes,
            adj_mx,
            in_steps,
            out_steps,
            steps_per_day,
            input_dim,
            output_dim,
            input_embedding_dim,
            tod_embedding_dim,
            ts_embedding_dim,
            dow_embedding_dim,
            time_embedding_dim,
            adaptive_embedding_dim,
            node_dim,
            feed_forward_dim,
            out_feed_forward_dim,
            num_heads,
            num_layers,
            num_layers_m,
            mlp_num_layers,
            dropout,
            use_mixed_proj,
            k_value=15,

            mem_num=128,
            mem_dim=128,
            **kwargs

    ):
        super().__init__()
        self.num_nodes = num_nodes

        self.in_steps = in_steps

        print("INFO: Assuming symmetric graph, calculating a single diffusion matrix.")

        if isinstance(adj_mx, (list, tuple)):
            actual_adj = adj_mx[0]
            print("INFO: adj_mx was passed as a list/tuple, extracted the first element.")
        else:
            actual_adj = adj_mx

        adj_tensor = torch.tensor(actual_adj, dtype=torch.float32)

        P_f = calculate_diffusion_matrix(adj_tensor)

        print(f"Applying Top-K filtering with K={k_value}...")
        P_filtered = top_k_filtering(P_f, k=k_value)
        print("Top-K filtering complete.")

        self.adj_mx = P_filtered


        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.ts_embedding_dim = ts_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.node_dim = node_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + adaptive_embedding_dim
                + ts_embedding_dim
                + time_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.num_layers_m = num_layers_m
        if self.input_embedding_dim > 0:
            self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if time_embedding_dim > 0:
            self.time_embedding = nn.Embedding(7 * steps_per_day, self.time_embedding_dim)

        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        self.adj_mx_encoder_1 = nn.Sequential(
            GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim, dropout=dropout)
        )
        self.adj_mx_encoder_2 = nn.Sequential(
            GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim,dropout=dropout)
        )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.memory_enhancer = Memory(
            model_dim=self.model_dim,
            mem_num=mem_num,  # from config
            mem_dim=mem_dim,  # from config
            dropout=dropout
        )

        self.ar_attn = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, out_feed_forward_dim, num_heads,
                                   dropout)
                for _ in range(num_layers_m)
            ]
        )
        if self.ts_embedding_dim > 0:
            self.time_series_emb_layer = nn.Conv2d(
                in_channels=self.input_dim * self.in_steps, out_channels=self.ts_embedding_dim, kernel_size=(1, 1),
                bias=True)

        fusion_input_dim = self.adaptive_embedding_dim + 2 * self.node_dim
        self.fusion_model = nn.Sequential(
            *[MultiLayerPerceptron(input_dim=fusion_input_dim,
                                    hidden_dim=fusion_input_dim,
                                    dropout=0.2)
                for _ in range(mlp_num_layers)],
            nn.Linear(in_features=fusion_input_dim, out_features=self.adaptive_embedding_dim, bias=True)
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs):

        x = history_data
        batch_size, _, num_nodes, _ = x.shape

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        if self.time_embedding_dim > 0:
            tod = x[..., 1]
            dow = x[..., 2]
        x = x[..., : self.input_dim]
        if self.ts_embedding_dim > 0:
            input_data = x.transpose(1, 2).contiguous()
            input_data = input_data.view(
                batch_size, self.num_nodes, -1).transpose(1, 2).unsqueeze(-1)
            # B L*3 N 1
            time_series_emb = self.time_series_emb_layer(input_data)
            time_series_emb = time_series_emb.transpose(1, -1).expand(batch_size, self.in_steps, self.num_nodes,
                                                                      self.ts_embedding_dim)
        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]

        if self.ts_embedding_dim > 0:
            features.append(time_series_emb)

        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.time_embedding_dim > 0:
            time_emb = self.time_embedding(
                ((tod + dow * 7) * self.steps_per_day).long()
            )
            features.append(time_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)

        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        for attn_t in self.attn_layers_t:
            x = attn_t(x, dim=1)

        for attn_s in self.attn_layers_s:
            x = attn_s(x, dim=2)

        x = self.memory_enhancer(x)

        if self.node_dim > 0:

            diffusion_matrix = self.adj_mx.to(device)  # (N, N)

            long_diffusion_matrix = calculate_long_range_diffusion(diffusion_matrix, k=2)

            node_embedding_short = self.adj_mx_encoder_1(diffusion_matrix.unsqueeze(0)).expand(batch_size, self.in_steps,
                                                                                             -1, -1)

            node_embedding_long = self.adj_mx_encoder_2(long_diffusion_matrix.unsqueeze(0)).expand(batch_size,
                                                                                                 self.in_steps, -1, -1)
            adp_graph = x[..., -self.adaptive_embedding_dim:]
            x_basic = x[..., :self.model_dim - self.adaptive_embedding_dim]

            graph_concat = torch.cat([adp_graph, node_embedding_short, node_embedding_long],
                                     dim=-1)  # (B, T, N, adaptive+2*node_dim)
            graph = self.fusion_model(graph_concat)

            x = torch.cat([x_basic, graph], dim=-1)

        for attn in self.ar_attn:                
            x = attn(x, dim=2, augment=True)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out
