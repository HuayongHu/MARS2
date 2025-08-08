import json
import os
def add_question_field():
    all_json_path = r'root/studies/competitions/MARS2/VR-Ads/all.json'
    question_file_path = r'/root/studies/competitions/MARS2/adsqa_full_set/adsqa_question_file.json'
    output_path = r'root/studies/competitions/MARS2/VR-Ads/all_with_questions.json'
    try:
        with open(all_json_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        with open(question_file_path, 'r', encoding='utf-8') as f:
            question_data = json.load(f)
        question_map = {item['question_id']: item['question'] for item in question_data}
        for item in all_data:
            question_id = item['question_id']
            if question_id in question_map:
                item['question'] = question_map[question_id]
            else:
                item['question'] = "问题未找到"
                print(f"警告: 未找到 question_id 为 {question_id} 的问题")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        print(f"成功添加 question 字段到 {output_path}")
        print(f"共处理 {len(all_data)} 条记录")
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
if __name__ == "__main__": #  补上之前忘掉的question字段内容
    add_question_field()