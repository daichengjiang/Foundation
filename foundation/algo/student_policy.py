import torch
import torch.nn as nn

class RaptorStudent(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 论文结构: Dense -> GRU -> Dense
        # 这里的 input_dim 包含了 Observation + Previous Action
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, action_dim)
        
        # 激活函数通常用 Tanh 或 ELU，RAPTOR 论文未详述，通常 Tanh 用于 GRU
        self.activation = nn.Tanh() 

    def forward(self, x, hidden=None):
        """
        x: (Batch, Seq_Len, Dim) for training 
           OR (Batch, Dim) for inference
        """
        # 如果输入是 (Batch, Dim)，增加时间维度
        is_inference = (x.dim() == 2)
        if is_inference:
            x = x.unsqueeze(1)
            
        # Encoder
        x = self.activation(self.encoder(x))
        
        # GRU
        # output: (Batch, Seq, Hidden), hidden: (1, Batch, Hidden)
        output, hidden = self.gru(x, hidden)
        
        # Decoder
        actions = self.decoder(output)
        
        if is_inference:
            return actions.squeeze(1), hidden
        else:
            return actions, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)