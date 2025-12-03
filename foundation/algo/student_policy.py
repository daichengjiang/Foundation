import torch
import torch.nn as nn

class RaptorStudent(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=16):
        super().__init__()
        # 论文结构: 
        # Input (26) -> Dense -> GRU (16) -> Dense -> Output (4)
        # 注意: 论文 S33-S36 描述: Input->Dense(16), GRU(16), Output->Dense(4)
        
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, 16),
            nn.Tanh() # 论文未明确激活函数，通常用 Tanh 或 ReLU
        )
        
        self.gru = nn.GRU(input_size=16, hidden_size=hidden_dim, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, num_actions),
            nn.Tanh() # 动作输出通常限制在 -1 到 1
        )
        
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden_state=None):
        # x shape: [Batch, Seq_Len, Obs_Dim] or [Batch, Obs_Dim]
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add sequence dim
            
        x_enc = self.encoder(x)
        
        # GRU Forward
        # out: [Batch, Seq, Hidden], hidden: [1, Batch, Hidden]
        out, new_hidden = self.gru(x_enc, hidden_state)
        
        # Take the last output for action
        action = self.decoder(out[:, -1, :])
        
        return action, new_hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)