import os
import json
def process_and_merge_json():
    root_dir = r'root/studies/competitions/MARS2/VR-Ads/result_competition'
    output_path = r'root/studies/competitions/MARS2/VR-Ads/all.json'
    merged_data = []
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            json_files = [f for f in os.listdir(folder_path)
                          if f.endswith('.json')]
            if not json_files:
                print(f"警告: 文件夹 {folder_name} 中没有找到JSON文件")
                continue
            json_path = os.path.join(folder_path, json_files[0])
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    prediction = data[0].get("prediction", "")
                    new_entry = {
                        "question_id": folder_name,
                        "answer": prediction
                    }
                    merged_data.append(new_entry)
                else:
                    print(f"警告: {json_path} 格式不符合预期")

            except Exception as e:
                print(f"处理文件 {json_path} 时出错: {str(e)}")
    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(merged_data, outfile, indent=2, ensure_ascii=False)
        print(f"成功生成合并文件: {output_path}")
        print(f"共处理 {len(merged_data)} 条记录")
    except Exception as e:
        print(f"写入输出文件时出错: {str(e)}")
if __name__ == "__main__":
    process_and_merge_json()