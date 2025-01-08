import torch
import torch.nn as nn
import torch.nn.functional as F 
from collections import OrderedDict



class ConsistencyModule(nn.Module):
    
    def __init__(self, embed_dim):
        super(ConsistencyModule, self).__init__()
        self.entity_cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4,batch_first=True)
        self.relation_cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4,batch_first=True)
        self.entity_self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4,batch_first=True)
        self.relation_self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4,batch_first=True)
       

    def forward(self, entity, relation, subgraph):
        
        #bs*num*hs
        entity_subgraph_attn, _ = self.entity_cross_attention(entity, subgraph, subgraph)
        relation_subgraph_attn, _ = self.relation_cross_attention(relation, subgraph, subgraph)

        entity_mlp_output, _ = self.entity_self_attention(entity, entity, entity)
        relation_mlp_output, _ = self.relation_self_attention(relation, relation, relation)
        return entity_subgraph_attn, relation_subgraph_attn, entity_mlp_output, relation_mlp_output
 