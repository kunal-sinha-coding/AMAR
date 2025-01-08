import torch
import torch.nn as nn
from collections import OrderedDict

class Mapnet(nn.Module):
    def __init__(self, word_embeddings, hidden_size):
        super(Mapnet, self).__init__()
        self.word_embeddings = word_embeddings  # 4* topk  * 256 * 1024
        self.entity_map_module = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(hidden_size, hidden_size // 16)),
            ("gelu", nn.GELU()),
            ("linear2", nn.Linear(hidden_size // 16, hidden_size))
        ]))
        self.relation_map_module = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(hidden_size, hidden_size // 16)),
            ("gelu", nn.GELU()),
            ("linear2", nn.Linear(hidden_size // 16, hidden_size))
        ]))
        self.subgraph_map_module = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(hidden_size, hidden_size // 16)),
            ("gelu", nn.GELU()),
            ("linear2", nn.Linear(hidden_size // 16, hidden_size))
        ]))

    


    def forward(self, entitys, relations, subgraphs):
        entity_embs = self.word_embeddings(entitys)
        relation_embs = self.word_embeddings(relations)
        subgraph_embs = self.word_embeddings(subgraphs)
      

        mean_entity_embs = torch.mean(entity_embs, dim=2)
        entity_vector = self.entity_map_module(mean_entity_embs) # 4* topk * 1024

        mean_relation_embs = torch.mean(relation_embs, dim=2)  
        relation_vector = self.relation_map_module(mean_relation_embs)

        mean_subgraph_embs = torch.mean(subgraph_embs, dim=2)  
        subgraph_vector = self.subgraph_map_module(mean_subgraph_embs)

        return entity_vector, relation_vector, subgraph_vector

