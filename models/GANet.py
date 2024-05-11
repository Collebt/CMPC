import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import math
from models.model import MLP


def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -torch.log(-torch.log(u + eps) + eps)

def sinkhorn_sorting_operator(r, n_iters=8):
    n = r.shape[1]
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)
def gumbel_sinkhorn(r, n_iters=8, temperature=0.7):
    r = torch.log(r)
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    r = (r + gumbel) / temperature
    return sinkhorn_sorting_operator(r, n_iters)

def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out
    
class AttentionPermMatrix(nn.Module):
    def __init__(self, blocks, temperature, sinkhorn_iter):
        super().__init__()
        self.blocks = blocks
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter

    def forward(self, b_q, b_k):
        sq = b_q.mean(dim=1).transpose(1,2)          # B * C * H_block * block  -->  B * block * H_block
        sk = b_k.mean(dim=1).transpose(1,2)          # B * C * H_block * block  -->  B * block * H_block

        R = torch.einsum('bie,bje->bij', sq, sk).to(b_q) * (b_q.shape[1] ** -0.5)

        return gumbel_sinkhorn(F.relu(R), self.sinkhorn_iter, self.temperature)

class SparseCutAttention(nn.Module):
    def __init__(self, blocks, temperature, sinkhorn_iter, inchannel, cut_length):
        super().__init__()
        self.blocks = blocks
        self.temperature = temperature
        self.cut_length = cut_length
        self.perm_mat = AttentionPermMatrix(blocks, temperature, sinkhorn_iter)
        self.norm = nn.InstanceNorm2d(inchannel) 
        
    def forward(self, q, k, v, edge=None):
        b_q = torch.cat(q.chunk(self.blocks, dim=2)[:-1], dim=-1)        # B * C * H * 1  -->  B * C * H_block * block-1  
        b_k = torch.cat(k.chunk(self.blocks, dim=2)[:-1], dim=-1)        # B * C * H * 1  -->  B * C * H_block * block-1
        b_v = torch.cat(v.chunk(self.blocks, dim=2)[:-1], dim=-1)        # B * C * H * 1  -->  B * C * H_block * block-1
        
        perm = self.perm_mat(b_q, b_k).unsqueeze(1).repeat(1, q.shape[1], 1, 1)           # B * C * block-1 * block-1
        k_sort = torch.matmul(b_k, perm)                                                  # B * C * H_block * block-1
        v_sort = torch.matmul(b_v, perm)                                                  # B * C * H_block * block-1
        
        q = q.transpose(1, 3)                                                             # B * 1 * H * C
        k_sort = torch.cat(k_sort.chunk(self.blocks, dim=-1)[:self.cut_length], dim=-2).transpose(1, 3)   # B * 1 * H_cut * C
        v_sort = torch.cat(v_sort.chunk(self.blocks, dim=-1)[:self.cut_length], dim=-2).transpose(1, 3)   # B * 1 * H_cut * C
        
        #for i in range(cut_length):
        attn_logits = torch.matmul(q, k_sort.transpose(2,3)) / self.temperature            # B * 1 * H * H_cut
        attn = F.softmax(attn_logits, dim=-1)
        val = torch.matmul(attn, v_sort)                                                   # B * 1 * H * C
        val = self.norm(val)
        
        return val #B*1*N*C
        
        '''
        b_v = torch.cat(val.transpose(1,3).chunk(self.blocks, dim=2)[:-1], dim=-1)         # B * C * H * 1  -->  B * C * H_block * block-1
        
        org_perm_val = torch.matmul(perm, b_v.transpose(2,3))                              # B * C * block-1 * H_block
        b_v_last = v.chunk(self.blocks, dim=2)[-1].transpose(1,3)
        
        return torch.cat([torch.cat(org_perm_val.chunk(self.blocks-1, dim=1), dim=2), b_v_last], dim =2)
        '''


class PosAttention(nn.Module):
    def __init__(self, blocks, temperature, sinkhorn_iter, inchannel, cut_length):
        super().__init__()
        self.blocks = blocks
        self.temperature = temperature
        self.cut_length = cut_length
        self.perm_mat = AttentionPermMatrix(blocks, temperature, sinkhorn_iter)
        self.norm = nn.InstanceNorm2d(inchannel) 
        
    def forward(self, q, k, v, adj_map, adj_attr):
        """attention based on the position KNN 

        Args:
            q B*C*N*1
            k B*C*N*1
            v B*C*N*1
            adj_map B*N*N

        Returns:
            _type_: _description_
        """
        #for i in range(cut_length):
        attn_logits = torch.einsum('bdn, bdm-> bnm', q.squeeze(), k.squeeze()) / self.temperature            # B * H*H
        attn_logits = adj_attr * attn_logits # give edge weight to the attention
        
        attn = masked_softmax(attn_logits, adj_map.to(torch.bool), dim=-1) #softmax with edge adjacency #B*H*H
        val = torch.matmul(attn, v.squeeze().transpose(1, 2)).unsqueeze(1)                       # B * 1 * H * C
        val = self.norm(val)
        
        return val

class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, pre=True):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1), stride=(1, stride)#, padding=(0, 1)
            ),
            nn.BatchNorm2d(outchannel)
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1), stride=(1, stride)#, padding=(0, 1)
            ),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, (1, 1), stride=1#, padding=(0, 1)
            ),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel)
        )
    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)       
        out = out + x1
        return F.relu(out)

class Group_Attention(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Group_Attention, self).__init__()
        self.group = 32
        self.cut_length = 3
        self.group_norm = nn.GroupNorm(self.group, inchannel)
        self.v_activation = F.elu
        self.linear = nn.Conv2d(inchannel, inchannel, kernel_size=(1,1), stride=(1,1), padding=0, bias=False)
        self.ffn = ResNet_Block(inchannel, outchannel, stride=1, pre=True)
        self.atten = PosAttention(blocks=10, temperature= (inchannel ** .5), sinkhorn_iter=8, inchannel=inchannel, cut_length=self.cut_length)


    def forward(self, data):

        x = data[0]
        edge_map = data[1]
        edge_attr = data[2]

        
        feat = self.linear(x)         
        #attention
        q = feat  # B * C * N * 1
        k = feat 
        v = self.v_activation(feat)          
        attn = self.atten(q, k, v, edge_map, edge_attr) # B * 1 * N * C
        feat_attn = attn.transpose(1,3) + x          # B * C * N * 1
        
        x_out = self.ffn(feat_attn)

        output = [x_out, edge_map, edge_attr]
        
        return output
        

class ATEM(nn.Module):
    def __init__(self, args):
        super(ATEM, self).__init__()
        #self.conv = nn.Conv2d(1, net_channels, (1, 2), stride=(1, 2),bias=True)
        #self.gconv = nn.Conv2d(1, 1, (1, input_channel), stride=(1, input_channel),bias=True)
        
        net_channels=args.net_channels
        feat_in_dim = 2048 if args.graph_in == "out_feat" else args.low_dim
        if args.node_func == 'cat':
            input_channel = 2 * feat_in_dim
        else: 
            input_channel = feat_in_dim

        self.encode_pos = args.encode_pos
        
        self.pos_emb = MLP(args.pos_mlp)  
        
        
        self.norm = nn.InstanceNorm2d(net_channels)  
        #self.em = EM(net_channels, 64, 3)      # 1 * c *iteration
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, net_channels, (1, input_channel), stride=(1, input_channel),bias=True),
            nn.BatchNorm2d(net_channels),
            nn.ReLU(),
        )
                 
        self.conv2 = nn.Sequential(
            nn.Conv2d(net_channels, net_channels, (1, 1)),
            nn.InstanceNorm2d(net_channels),
            nn.BatchNorm2d(net_channels),
            nn.ReLU(),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(net_channels, 1, (1, 1)),
        ) 
        self.layer = self.make_layers(Group_Attention, net_channels)
        
    def make_layers(self, Group_Attention, net_channels):
        layers = []
        layers.append(Group_Attention(net_channels, net_channels))
        layers.append(Group_Attention(net_channels, net_channels))
        layers.append(Group_Attention(net_channels, net_channels))
        layers.append(Group_Attention(net_channels, net_channels))

        return nn.Sequential(*layers)
        
    
    def forward(self, x, edge_mask, edge_attr, pos):

        """
        x: the node feature B*N*C
        edge_mask: adjacency mask  B*N*N in {0,1}
        edge_attr: adjacency weight  B*N*N in 
        pos: position of the node B*N*2 
        """

        x_input = x.unsqueeze(1) #B*N*C -> B*1*N*C
        x_input =  self.norm(x_input) #first norm the input pairs
        # input matching pairs (batch, N, channel)
        out = self.conv1(x_input) #B*1*N*C -> B*C*N*1

        if self.encode_pos:
            pos_feat = self.pos_emb(pos.transpose(1, 2)) #make input[batch,L,2] ->input[batch,2, L] suit for mlp
            out = out + pos_feat.unsqueeze(-1) #[batch, num_node, out_channel]

        #resnet block and attention
        input = [out, edge_mask, edge_attr]
        
        data = self.layer(input)

        out = data[0]
        node_feat = out.squeeze().permute(0,2,1) #[batch, N, C]

        out = self.final_conv(out) 
        out = out.view(out.size(0), -1) #[batch, N]
        # out = out
        # w = torch.tanh(out)
        # w = F.relu(out)  #output weight (batch, N)
        w = out

        weighted_x = w.unsqueeze(-1) * x #the product is not supprt for the concatnate. 
        coeff_mat = torch.einsum('bmd, bnd-> bmn', weighted_x, weighted_x) #attention connect
        node_val = torch.softmax(coeff_mat, dim=-1).sum(-1)
        # output_predict = torch.softmax(coedd_mat, dim=-1) #get output predict

        node_val = w #test the output weight
        
        return node_val, node_feat#, residual, mean


