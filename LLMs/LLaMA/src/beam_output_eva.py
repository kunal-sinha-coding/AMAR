from llmtuner import ChatModel
import json
from tqdm import tqdm
import random
import re
import os
from llmtuner.tuner.core import get_infer_args
from pro_model.totoken import data_load_retrieval, get_extra_input_ids

def dump_json(obj, fname, indent=4, mode='w' ,encoding="utf8", ensure_ascii=False):
    if "b" in mode:
        encoding = None
    with open(fname, "w", encoding=encoding) as f:
        return json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)

def main():
    model_args, data_args, _, _ = get_infer_args()
    id2rel, id2ent, id2sub = data_load_retrieval(data_args)
    chat_model = ChatModel()
    output_data = []
    with open(os.path.join(data_args.dataset_dir,data_args.dataset,'examples.json'), 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        # random.shuffle(json_data)
        total_lines = 0
        matched_lines = 0
        will_matched_lines = 0
        
        # 2. 读取每一行
        for data in tqdm(json_data):
            total_lines += 1
            query = data['instruction']+data['input']
            id = data['ID']
            entity = id2ent[id]
            relation = id2rel[id]
            subgraph = id2sub[id]
            predict = chat_model.chat_beam(query,entity,relation,subgraph)
            predict = [p[0] for p in predict]
            output_data.append({'label':data['output'],'predict':predict})
            for p in predict:
                # 4. 检查"label"和"predict"的值是否相等
                if data['output'] == p:
                    matched_lines += 1
                    break
            for p in predict:
                # 4. 检查"label"和"predict"的值是否相等
                if re.sub(r'\[.*?\]', '', data['output']) == re.sub(r'\[.*?\]', '', p):
                    will_matched_lines += 1
                    break
       

    # 5. 计算相等的行的数量
    print(f"Total lines: {total_lines}")
    print(f"Matched lines: {matched_lines}")
    print(f"Will Matched lines: {will_matched_lines}")

    # 6. 计算相等行的占比
    percentage = (matched_lines / total_lines) * 100
    print(f"Percentage of matched lines: {percentage:.2f}%")
    # 6. 计算相等行的占比
    will_percentage = (will_matched_lines / total_lines) * 100
    print(f"Percentage of will matched lines: {will_percentage:.2f}%")
    
    
    output_dir = os.path.join(os.path.dirname(model_args.checkpoint_dir[0]),'evaluation_beam/generated_predictions.jsonl')
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))
    # with open(output_dir, 'w') as f:
    #     for item in output_data:
    #         json_string = json.dumps(item)
    #         f.write(json_string + '\n')
    run_prediction(output_data,os.path.dirname(output_dir),output_predictions=True)

def run_prediction(output_data,output_dir,output_predictions=True):
    print()
    print('Start predicting ')
            
    ex_cnt = 0
    contains_ex_cnt = 0
    output_list = []
    real_total = 0
    for i,pred in enumerate(output_data):
        predictions = pred['predict']
        gen_label = pred['label']

        output_list.append({
            'predictions':predictions,
            'gen_label':gen_label,
        })

        if predictions[0].lower()==gen_label.lower():
            ex_cnt+=1

        if any([x.lower()==gen_label.lower() for x in predictions]):
            contains_ex_cnt+=1
        
        if gen_label.lower()!='null':
            real_total+=1

    
    print(f"""total:{len(output_list)}, 
                    ex_cnt:{ex_cnt}, 
                    ex_rate:{ex_cnt/len(output_list)}, 
                    real_ex_rate:{ex_cnt/real_total}, 
                    contains_ex_cnt:{contains_ex_cnt}, 
                    contains_ex_rate:{contains_ex_cnt/len(output_list)}
                    real_contains_ex_rate:{contains_ex_cnt/real_total}
                    """)

        
    if output_predictions:
        file_path = os.path.join(output_dir,f'beam_test_top_k_predictions.json')
        
        gen_statistics_file_path = os.path.join(output_dir,f'beam_test_gen_statistics.json')
        gen_statistics = {
            'total':len(output_list),
            'exmatch_num': ex_cnt,
            'exmatch_rate': ex_cnt/len(output_list),
            'real_exmatch_rate':ex_cnt/real_total, 
            'contains_ex_num':contains_ex_cnt,
            'contains_ex_rate':contains_ex_cnt/len(output_list),
            'real_contains_ex_rate':contains_ex_cnt/real_total
        }
        dump_json(output_list, file_path, indent=4)
        dump_json(gen_statistics, gen_statistics_file_path,indent=4)



if __name__ == "__main__":
    main()
