from typing import List
import json
import math
def process_ent_list(ent_list, data_args):
    if len(ent_list) >= data_args.topk:
        return ent_list[:data_args.topk]
    else:
        new_ent_list = ent_list * math.ceil(data_args.topk / len(ent_list))
        return new_ent_list[:data_args.topk]

def data_load_retrieval(data_args):
    
    if 'WebQSP' in data_args.dataset:
        dataset = 'WebQSP'
        splits = ['train','test']
    else:
        dataset = 'CWQ'
        splits = ['train','test','dev']
    id2rel = {}
    for split in splits:
        with open(f'data/retrieval_data/{dataset}_{split}_cand_rels_sorted.json','r') as f:
            id2rel.update(json.load(f))
    id2rel = {Id:rel[:data_args.topk] for Id, rel in id2rel.items()}

    id2ent_ = {}
    for split in splits:
        with open(f'data/retrieval_data/{dataset}_{split}_merged_cand_entities_elq_facc1.json','r') as f:
            id2ent_.update(json.load(f))
    id2ent = {}
    for Id, ent in id2ent_.items():
        ent = [en['label'] for en in process_ent_list(ent, data_args)]
        id2ent[Id] = ent

    subgraphs = []
    for split in splits:
        with open(f'data/retrieval_data/{dataset}_{split}_subgraph_BM25.json','r') as f:
            subgraphs.extend(json.load(f))
    id2subgraph = {}
    for subg in subgraphs:
        id2subgraph[subg['QuestionId']] = [ctx['text'] for ctx in subg['ctxs'][:data_args.topk]]

    return id2rel, id2ent, id2subgraph

def get_ids(tokenizer, sequence, max_length):
    sequence = tokenizer.encode(sequence)
    sequence = [tokenizer.bos_token_id] + sequence + [tokenizer.eos_token_id]
    if len(sequence) < max_length:
        sequence = sequence + [tokenizer.pad_token_id] * (max_length - len(sequence))
    return sequence[:max_length]

def get_extra_input_ids(tokenizer, 
                  entitys: List[str], 
                  relations: List[str], 
                  subgraphs: List[str],
                  max_length: int 
                  ):

    entity_ids = [get_ids(tokenizer, entity, max_length) for entity in entitys]
    relation_ids = [get_ids(tokenizer, relation, max_length) for relation in relations]
    subgraph_ids = [get_ids(tokenizer, subgraph, max_length) for subgraph in subgraphs]
    

    return entity_ids, relation_ids, subgraph_ids