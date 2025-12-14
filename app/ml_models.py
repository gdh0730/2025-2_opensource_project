# app/ml_models.py
import torch
import torch.nn as nn


class RoadGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 X0: torch.Tensor, A_hat: torch.Tensor):
        super().__init__()
        self.register_buffer("A_hat", A_hat)
        self.register_buffer("X0", X0)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self):
        H1 = torch.relu(self.A_hat @ self.fc1(self.X0))
        Z = torch.relu(self.A_hat @ self.fc2(H1))
        return Z


class EncoderGCN(nn.Module):
    def __init__(
        self,
        dynamic_dim: int,
        gcn_in_dim: int,
        gcn_hidden_dim: int,
        gcn_out_dim: int,
        X0: torch.Tensor,
        A_hat: torch.Tensor,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.gcn = RoadGCN(gcn_in_dim, gcn_hidden_dim, gcn_out_dim, X0, A_hat)
        self.gcn_out_dim = gcn_out_dim
        self.lstm_input_dim = dynamic_dim + gcn_out_dim
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_p if lstm_layers > 1 else 0.0,
        )

    def forward(self, x_dyn, node_idx):
        Z = self.gcn()              # (N, gcn_out_dim)
        e = Z[node_idx]             # (B, gcn_out_dim)

        B, T, _ = x_dyn.shape
        e_expanded = e.unsqueeze(1).repeat(1, T, 1)

        lstm_input = torch.cat([x_dyn, e_expanded], dim=-1)
        _, (h_n, c_n) = self.lstm(lstm_input)
        return h_n, c_n, e


class DecoderGCN(nn.Module):
    def __init__(
        self,
        gcn_out_dim: int,
        static_dim: int,
        lstm_hidden_dim: int,
        horizon: int,
        time_dim: int,
        fc_hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.time_dim = time_dim
        self.input_dim = 1 + gcn_out_dim + static_dim + time_dim

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim + static_dim + gcn_out_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1),
        )

    def forward(
        self,
        h0, c0,
        e_node,
        x_static,
        future_time=None,
        y=None,
        last_value=None,
        teacher_forcing: bool = False,
    ):
        B = e_node.size(0)
        H = self.horizon
        outputs = []

        prev = last_value.view(B, 1)
        h, c = h0, c0

        if self.time_dim > 0 and future_time is None:
            raise ValueError("time_dim > 0 인데 future_time 이 없습니다.")

        for t in range(H):
            if self.time_dim > 0:
                ft = future_time[:, t, :]
                dec_in = torch.cat([prev, e_node, x_static, ft], dim=-1)
            else:
                dec_in = torch.cat([prev, e_node, x_static], dim=-1)

            dec_in = dec_in.unsqueeze(1)  # (B,1,input_dim)
            out, (h, c) = self.lstm(dec_in, (h, c))
            h_t = out[:, -1, :]

            fc_in = torch.cat([h_t, x_static, e_node], dim=-1)
            fc_in = self.dropout(fc_in)
            step = self.fc(fc_in).squeeze(-1)  # (B,)

            outputs.append(step.unsqueeze(1))

            if teacher_forcing and y is not None:
                prev = y[:, t].view(B, 1)
            else:
                prev = step.detach().view(B, 1)

        return torch.cat(outputs, dim=1)  # (B,H)


class Seq2SeqGCNLSTM(nn.Module):
    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        horizon: int,
        time_dim: int,
        gcn_in_dim: int,
        gcn_hidden_dim: int,
        gcn_out_dim: int,
        X0: torch.Tensor,
        A_hat: torch.Tensor,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        fc_hidden_dim: int = 128,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.encoder = EncoderGCN(
            dynamic_dim=dynamic_dim,
            gcn_in_dim=gcn_in_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            gcn_out_dim=gcn_out_dim,
            X0=X0,
            A_hat=A_hat,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            dropout_p=dropout_p,
        )
        self.decoder = DecoderGCN(
            gcn_out_dim=gcn_out_dim,
            static_dim=static_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            horizon=horizon,
            time_dim=time_dim,
            fc_hidden_dim=fc_hidden_dim,
            lstm_layers=lstm_layers,
            dropout_p=dropout_p,
        )

    def forward(
        self,
        x_dyn,
        node_idx,
        x_static,
        future_time=None,
        y=None,
        last_value=None,
        teacher_forcing: bool = False,
    ):
        h0, c0, e = self.encoder(x_dyn, node_idx)
        return self.decoder(
            h0, c0,
            e_node=e,
            x_static=x_static,
            future_time=future_time,
            y=y,
            last_value=last_value,
            teacher_forcing=teacher_forcing,
        )
