# AMAR

This is the code of the paper 'Harnessing Large Language Models for Knowledge Graph Question Answering via Adaptive Multi-Aspect Retrieval-Augmentation' (Accepted by AAAI 2025).

# Environment Setup
    conda create -n amar python=3.8
    pip install -r requirement.txt

### Freebase Setup
Both datasets use Freebase as the knowledge source. You may refer to [Freebase Virtuoso Setup](https://github.com/dki-lab/Freebase-Setup) to set up a Virtuoso triplestore service. We briefly list some key steps below:


Download OpenLink Virtuoso from https://github.com/openlink/virtuoso-opensource/releases, and put it in `Amar/`

Env setting:

    sudo apt install unixodbc unixodbc-dev

Download Database:

    cd Freebase-Setup
    wget https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip
    tar -zxvf virtuoso_db.zip

to start the Virtuoso service:
    
    python3 virtuoso.py start 3001 -d virtuoso_db

and to stop a currently running service at the same port:
    
    python3 virtuoso.py stop 3001


## KGQA Datasets and Retrieval data

Download from [Google drive](https://drive.google.com/drive/folders/1uOcpPoBcFeL2JWE6-Wpj6kuR-7Vdgiyz?usp=sharing) or [Baidu drive](https://pan.baidu.com/s/12w4bCqFhDKp6iPVW6i2cFw?pwd=p9da), and unzip data.zip to `Amar/data`


More details of entity/relation retrieval can be found in [GMT-KBQA](https://github.com/HXX97/GMT-KBQA), and subgraph retrieval can be found in [DECAF](https://github.com/awslabs/decode-answer-logical-form).

## Reproduction
Change the `--model_name_or_path` in `run_ft.sh` to your LLM checkpoint path. 

Reproduce the results for CWQ and WebQSP by executing the following:

    bash run_all.sh

Alternatively, you can run the commands step-by-step as shown below:
### Finetuning 
    CUDA_VISIBLE_DEVICES=0 bash run_ft.sh WebQSP LLaMA-2-7b-hf webqsp_100_7_32_16 train 100 7 0 32 16 15
    CUDA_VISIBLE_DEVICES=0 bash run_ft.sh CWQ LLaMA-2-13b-hf cwq_4_16_32_16 train 4 16 0 32 16 8

### Inference
    CUDA_VISIBLE_DEVICES=0 bash run_ft.sh WebQSP LLaMA-2-7b-hf webqsp_100_7_32_16 test 100 7 0 32 16 15
    CUDA_VISIBLE_DEVICES=0 bash run_ft.sh CWQ LLaMA-2-13b-hf cwq_4_16_32_16 test 4 16 0 32 16 8

### Querying on Freebase
    CUDA_VISIBLE_DEVICES=0 python -u eval_final.py --dataset WebQSP --pred_file Reading/LLaMA-2-7b-hf/WebQSP_webqsp_100_7_32_16/evaluation_beam/beam_test_top_k_predictions.json
    CUDA_VISIBLE_DEVICES=0 python -u eval_final.py --dataset CWQ --pred_file Reading/LLaMA-2-13b-hf/CWQ_cwq_4_16_32_16/evaluation_beam/beam_test_top_k_predictions.json

Querying with golden entity:

    CUDA_VISIBLE_DEVICES=0 python -u eval_final.py --dataset WebQSP --pred_file Reading/LLaMA-2-7b-hf/WebQSP_webqsp_100_7_32_16/evaluation_beam/beam_test_top_k_predictions.json --golden_ent
    CUDA_VISIBLE_DEVICES=0 python -u eval_final.py --dataset CWQ --pred_file Reading/LLaMA-2-13b-hf/CWQ_cwq_4_16_32_16/evaluation_beam/beam_test_top_k_predictions.json --golden_ent


This repo refers to [ChatKBQA](https://github.com/LHRLAB/ChatKBQA), [GMT-KBQA](https://github.com/HXX97/GMT-KBQA) and [DECAF](https://github.com/awslabs/decode-answer-logical-form). Thanks for their great jobs!
