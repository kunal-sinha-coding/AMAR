#!/bin/bash

# webqsp
CUDA_VISIBLE_DEVICES=0 bash run_ft.sh WebQSP LLaMA-2-7b-hf webqsp_100_7_32_16 train 100 7 0 32 16 15
sleep 1m
CUDA_VISIBLE_DEVICES=0 bash run_ft.sh WebQSP LLaMA-2-7b-hf webqsp_100_7_32_16 test 100 7 0 32 16 15
CUDA_VISIBLE_DEVICES=0 python -u eval_final.py --dataset WebQSP --pred_file Reading/LLaMA-2-7b-hf/WebQSP_webqsp_100_7_32_16/evaluation_beam/beam_test_top_k_predictions.json
CUDA_VISIBLE_DEVICES=0 python -u eval_final.py --dataset WebQSP --pred_file Reading/LLaMA-2-7b-hf/WebQSP_webqsp_100_7_32_16/evaluation_beam/beam_test_top_k_predictions.json --golden_ent


# cwq
# CUDA_VISIBLE_DEVICES=0 bash run_ft.sh CWQ LLaMA-2-13b-hf cwq_4_16_32_16 train-test 4 16 1 32 16 8
# sleep 1m
# CUDA_VISIBLE_DEVICES=0 bash run_ft.sh CWQ LLaMA-2-13b-hf cwq_4_16_32_16 train 4 16 1 32 16 8
# CUDA_VISIBLE_DEVICES=0 python -u eval_final.py --dataset CWQ --pred_file Reading/LLaMA-2-13b-hf/CWQ_cwq_4_16_32_16/evaluation_beam/beam_test_top_k_predictions.json
# CUDA_VISIBLE_DEVICES=0 python -u eval_final.py --dataset CWQ --pred_file Reading/LLaMA-2-13b-hf/CWQ_cwq_4_16_32_16/evaluation_beam/beam_test_top_k_predictions.json --golden_ent
