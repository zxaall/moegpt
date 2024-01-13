# Copyright (c) 2023 Val Krigan, no rights reserved. 
# I.e. you can copy/paste SoftMoE in your project, modify, remove comments, including this one

import torch
import torch.nn as nn
from torch.nn import functional as F

class SoftMoE(nn.Module):
    """ soft mixture of experts 
        the idea:  have a trainable routing table in front of N*FF experts.
            table multiplied by token produces weights for each of N experts.
            output is a weighted sum (softmaxed) of N experts output
        sparse mixture of experts is trained soft MoE with fixed routing table,
            then only K (=2 usually) outputs a selected, with highests weights.
            as weights are known before experts call we can call only K selected experts
    """
    
    # making state shared by all objects, it's hard to control individual within mingpt architecture
    moe_nums = None  # (total, active) number of experts
    sparse = None    # None for soft moe mode, means router and all experts are trained
                     # active number of experts for sparse mode

    def set_sparse(on=True):
        print(f"SoftMoE setting sparse mode to '{on}', MoE params: {SoftMoE.moe_nums}") 
        if on:
            SoftMoE.sparse = SoftMoE.moe_nums[1]
        else:
            SoftMoE.sparse = None
    
    class Expert(nn.Module):
        """ simplified FF network, note: with no dropout"""
        def __init__(self, expert_size, n_embd, forward_width):
            super().__init__()
            self.n_embd = n_embd
            self.forward_in = expert_size  # all experts work in paralles, have the same input
            self.forward_width = forward_width 
            self.net = nn.Sequential(
                nn.Linear(self.forward_in, forward_width),
                nn.GELU(), 
                nn.Linear(forward_width, n_embd),
            )

        def forward(self, x):
            return self.net(x)
        
    """ params:
            moe_nums  - (total, active)  numbers of experts
            expert_size - whatever comes out of attention layer, usually ==m_embd
            m_embd      - emedding width
            forward_width  - internal intermediate data width, controls expert's size, usually 4*m_embd
    #"""
    def __init__(self, moe_nums, expert_size, n_embd, forward_width, dropout, sparse =False):
        super().__init__()
        self.n_embd = n_embd
        self.dropout_ratio = dropout
        self.forward_in = expert_size  # all experts have the same input
        self.forward_width = forward_width if forward_width!=None else 4*n_embd
        #!!self.forward_width = forward_width if forward_width!=None else n_embd//2
        #self.moe_nums = moe_nums
        SoftMoE.moe_nums = moe_nums   # globalizing it
        
        self.num_experts = moe_nums[0]
        self.router = nn.Linear(self.forward_in, self.num_experts)   # from token produces weights for each expert
        self.experts = nn.ModuleList([SoftMoE.Expert(expert_size, n_embd, self.forward_width) for _ in range(self.num_experts)])
        self.dropout = nn.Dropout(dropout)  # dropout at the end

    def forward(self, x):
        sparse = SoftMoE.sparse  # global state for all objects
        # there is no separate call for it, changing here
        if sparse:
            self.router.eval()
            self.router.requires_grad_(False)
        else:
            self.router.train()
            self.router.requires_grad_(True)
        
        weights = self.router(x)

        if sparse == None or sparse==self.num_experts:
            # can be done more efficiently if we represent all experts as one matrix
            weights = F.softmax(weights, dim=-1) 
            out = torch.stack([e(x) for e in self.experts], dim=-1)
            out = torch.sum(out * weights.unsqueeze(-2), dim=-1)
        else:
            # x: Tensor of shape [batch_size, n, d_model]
            # weights: Tensor of shape [batch_size, n, k],  k=self.num_experts
            # m: number of top experts to select (integer), m=self.sparse
            batch_size, n, _ = x.shape
            
            # the number of selected experts
            m = min(self.num_experts, sparse)

            # Select the top m weights and their indices
            top_m_values, top_m_indices = torch.topk(weights, m, dim=-1)
            
            # Apply softmax to the selected weights
            normalized_weights = F.softmax(top_m_values, dim=-1)
            

            # Initialize the result tensor
            out = torch.zeros_like(x)

            # Process each expert's inputs
            for j in range(m):
                for expert_idx in range(len(self.experts)):
                    # Create a mask for the current expert
                    mask = (top_m_indices[:, :, j] == expert_idx)

                    # Check if the current expert is used
                    if mask.any():
                        # Gather inputs and weights for the current expert
                        expert_inputs = x[mask]
                        expert_weights = normalized_weights[:, :, j][mask].unsqueeze(-1)

                        # Call the expert with gathered inputs
                        expert_output = self.experts[expert_idx](expert_inputs)

                        # Scale outputs by weights and add them to the result tensor
                        out[mask] += expert_output * expert_weights
        
        if self.dropout_ratio:
            out = self.dropout(out)
        return out

