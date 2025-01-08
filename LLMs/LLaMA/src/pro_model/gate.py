import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim // 2)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear( input_dim // 2, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.mlp = MLP(embedding_dim, embedding_dim)

    def forward(self, question, retrieval):
        question = self.mlp(question)
        retrieval = self.mlp(retrieval)
        simi = torch.bmm(retrieval,question.permute(0,2,1)) # B x L_retrieval x L_question
        simi = torch.mean(simi,dim=-1,keepdim=True)  # B x L_retrieval x 1
        simi = torch.sigmoid(simi)
        return simi


class GatingModule(nn.Module):
    def __init__(self, token_dim):
        super(GatingModule, self).__init__()
        self.dense_question_mlp = MLP(token_dim, token_dim)
        self.siamese_entity = SiameseNetwork(token_dim)
        self.siamese_relation = SiameseNetwork(token_dim)
        self.siamese_subgraph = SiameseNetwork(token_dim)

    #bs*length*embed
    def forward(self, question, entity_token, relation_token, subgraph_token):
        dense_question = self.dense_question_mlp(question) 
        
        gate_entity = self.siamese_entity(dense_question, entity_token)
        gate_relation = self.siamese_relation(dense_question, relation_token)
        gate_subgraph = self.siamese_subgraph(dense_question, subgraph_token)
        
        weighted_entity = (gate_entity * entity_token) # B x L_retrieval x 1  and   B x L_retrieval x embed
        weighted_relation = (gate_relation * relation_token)
        weighted_subgraph = (gate_subgraph * subgraph_token)
        
        combined_embedding = torch.cat([ weighted_entity, weighted_relation, weighted_subgraph], dim=1)
        
        return combined_embedding