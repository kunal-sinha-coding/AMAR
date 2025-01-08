import torch
from pro_model.pm import PromptTuningModelForCausalLM
from pro_model.totoken import data_load_retrieval, get_extra_input_ids
from typing import Any, Dict, Generator, List, Optional, Tuple
from threading import Thread
from transformers import GenerationConfig, TextIteratorStreamer

from llmtuner.extras.misc import dispatch_model, get_logits_processor
from llmtuner.extras.template import get_template_and_fix_tokenizer
from llmtuner.tuner.core import get_infer_args, load_model_and_tokenizer
import os
def get_latest_checkpoint_dir(checkpoint_base_dir):
    checkpoint_dirs = [d for d in os.listdir(checkpoint_base_dir) if os.path.isdir(os.path.join(checkpoint_base_dir, d))]
    if not checkpoint_dirs:
        return None
    checkpoint_dirs.sort()
    latest_checkpoint_dir = checkpoint_dirs[-1]
    return latest_checkpoint_dir


class ChatModel:

    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        model_args, data_args, finetuning_args, self.generating_args = get_infer_args(args)
        self.data_args = data_args
        self.model_args = model_args
        # last_checkpoint_dir = os.path.join(model_args.checkpoint_dir[0], get_latest_checkpoint_dir(model_args.checkpoint_dir[0]))
        last_checkpoint_dir = model_args.checkpoint_dir[0]
        model, self.tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
        model = PromptTuningModelForCausalLM(model, model_args.soft_prompt_length).to(dtype=torch.bfloat16)

        print("load model checkpoint : {}.".format(last_checkpoint_dir+ '/pytorch_model.bin'))
        state_dicts = torch.load(last_checkpoint_dir+ '/pytorch_model.bin',map_location='cpu')
        # state_dicts = torch.load('/data2/lixinhang/aaai/chatkbqa2/ChatKBQA/Reading/LLaMA-2-7b-hf/WebQSP_default/checkpoint2/checkpoint-105/pytorch_model.bin',map_location='cpu')
        if "model.base_model.model.lm_head.0.weight" in state_dicts:
            new_key = "model.base_model.model.lm_head.weight"
            state_dicts[new_key] =state_dicts.pop("model.base_model.model.lm_head.0.weight")

        model.load_state_dict(state_dicts)
    
       
        self.model = model.to(dtype=torch.bfloat16)
        self.model.eval()
        

        self.tokenizer.padding_side = "left"
        self.model = dispatch_model(self.model)
        self.template = get_template_and_fix_tokenizer(data_args.template, self.tokenizer)
        self.system_prompt = data_args.system_prompt

    def process_args(
        self,
        query: str,
        entity: str,
        relation: str,
        subgraph: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Tuple[Dict[str, Any], int]:
        system = system or self.system_prompt

        prompt, _ = self.template.encode_oneturn(
            tokenizer=self.tokenizer, query=query, resp="", history=history, system=system
        )
        entity_ids, relation_ids, subgraph_ids = get_extra_input_ids(self.tokenizer, 
                                                                         entity, relation, subgraph,
                                                                         self.data_args.extra_infor_len)
        gate_ids = [self.tokenizer.bos_token_id] + prompt
        if len(gate_ids) > self.data_args.gate_len-1:
            gate_ids = gate_ids[:self.data_args.gate_len-1]
        pad_len =  self.data_args.gate_len-len(gate_ids)-1
        pad_gate_ids =  gate_ids + [self.tokenizer.pad_token_id]*pad_len + [self.tokenizer.eos_token_id]

                

        input_ids = torch.tensor([prompt], device=self.model.model.device)
        gate_ids = torch.tensor([pad_gate_ids], device=self.model.model.device)
        
        entity_ids = torch.tensor([entity_ids], device=self.model.model.device)
        relation_ids = torch.tensor([relation_ids], device=self.model.model.device)
        subgraph_ids = torch.tensor([subgraph_ids], device=self.model.model.device)

        prompt_length = len(input_ids[0])

        do_sample = input_kwargs.pop("do_sample", None)
        temperature = input_kwargs.pop("temperature", None)
        top_p = input_kwargs.pop("top_p", None)
        top_k = input_kwargs.pop("top_k", None)
        repetition_penalty = input_kwargs.pop("repetition_penalty", None)
        max_length = input_kwargs.pop("max_length", None)
        max_new_tokens = input_kwargs.pop("max_new_tokens", None)

        generating_args = self.generating_args.to_dict()
        generating_args.update(dict(
            do_sample=False,
            num_beams = generating_args["num_beams"],
            num_beam_groups = generating_args["num_beams"],
            diversity_penalty = 1.0,
            num_return_sequences=generating_args["num_beams"],
            temperature=temperature or generating_args["temperature"],
            top_p=top_p or generating_args["top_p"],
            top_k=top_k or generating_args["top_k"],
            repetition_penalty=repetition_penalty or generating_args["repetition_penalty"],
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            pad_token_id=self.tokenizer.pad_token_id
        ))

        if max_length:
            generating_args.pop("max_new_tokens", None)
            generating_args["max_length"] = max_length

        if max_new_tokens:
            generating_args.pop("max_length", None)
            generating_args["max_new_tokens"] = max_new_tokens

        gen_kwargs = dict(
            inputs =input_ids,
            gate_ids = gate_ids,
            entitys = entity_ids,
            relations = relation_ids,
            subgraphs = subgraph_ids,
            generation_config=GenerationConfig(**generating_args),
            logits_processor=get_logits_processor()
        )
       

        return gen_kwargs, prompt_length

    @torch.inference_mode()
    def chat_beam(
        self,
        query: str,
        entity: str,
        relation: str,
        subgraph: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Tuple[str, Tuple[int, int]]:
        gen_kwargs, prompt_length = self.process_args(query, entity, relation, subgraph, history, system, **input_kwargs)
        generation_output = self.model.inference(**gen_kwargs,return_dict_in_generate=True, output_scores=True)
    
        outputs = [g[prompt_length:] for g in generation_output['sequences'].tolist()]
        outputs_scores = [s for s in generation_output['sequences_scores'].tolist()]
        response = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        
        response_dict = {}
        for resp, score in zip(response, outputs_scores):
            if resp not in response_dict or score > response_dict[resp]:
                response_dict[resp] = score

        # 将字典转换为元组列表并按得分排序
        sorted_responses = sorted(response_dict.items(), key=lambda x: x[1], reverse=True)
        # response_length = len(outputs)
        return sorted_responses
    
    @torch.inference_mode()
    def chat(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Tuple[str, Tuple[int, int]]:
        gen_kwargs, prompt_length = self.process_args(query, history, system, **input_kwargs)
        generation_output = self.model.generate(**gen_kwargs)
        outputs = generation_output.tolist()[0][prompt_length:]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        response_length = len(outputs)
        return response, (prompt_length, response_length)

    @torch.inference_mode()
    def stream_chat(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Generator[str, None, None]:
        gen_kwargs, _ = self.process_args(query, history, system, **input_kwargs)
        streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        yield from streamer
