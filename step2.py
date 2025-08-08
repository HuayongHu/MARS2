import argparse
import time
import os
import json
import random
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import numpy as np


class VideoQAProcessor:
    """视频问答处理器类"""
    
    def __init__(self, model_dir):
        self.model = None
        self.processor = None
        self.current_video_name = ""
        self.video_inputs = None
        self.video_kwargs = None
        self._load_model(model_dir)
    
    def _load_model(self, model_dir):
        """加载模型和处理器"""
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_dir)
    
    def _prepare_video_inputs(self, video_path, video_name):
        """准备视频输入数据"""
        if self.current_video_name != video_name:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": int(float(768 * 28 * 28 / 4 * 1.2)),
                            "fps": 2.0,
                        },
                    ],
                }
            ]
            image_inputs, self.video_inputs, self.video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            self.current_video_name = video_name
    
    def response(self, video_name, question):
        """生成视频问答响应"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_name,
                        "max_pixels": 360 * 420 * 4,
                        "fps": 2.0,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        print(messages)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=None,
            videos=self.video_inputs,
            padding=True,
            return_tensors="pt",
            **self.video_kwargs,
        )

        inputs = inputs.to("cuda")

        # 推理生成
        id = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=1,
            max_new_tokens=2048
        )
        
        trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, id)
        ]
        
        output = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output[0].strip()


def load_data(file_path):
    """加载JSON文件数据"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def initialize_random_seeds(seed_value):
    """初始化所有随机数种子以确保结果可复现"""
    # 设置PyTorch的随机数种子
    torch.manual_seed(seed_value)

    # 如果使用CUDA（GPU），还需要设置CUDA的随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # 如果使用多GPU
        torch.backends.cudnn.deterministic = True  # 确保卷积操作的结果确定
        torch.backends.cudnn.benchmark = False  # 关闭自动优化，保证结果一致

    # 设置Python内置随机数种子
    random.seed(seed_value)

    # 设置NumPy的随机数种子
    np.random.seed(seed_value)

    print(f"随机数种子设置为: {seed_value}")


def create_output(directory_path):
    """创建输出目录（如果不存在）"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def save(output_path, predictions):
    """保存预测结果到JSON文件"""
    results = []
    for prediction in predictions:
        results.append({
            "prediction": prediction
        })
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)


def add_input_prompt(question):
    """构建输入提示词"""
    return (f"You are an expert in video analysis and advertising content understanding. "
            f"Please analyze the provided video frames and answer the question accurately.\n\n "
            f"Please provide a detailed and accurate answer based on the visual content shown "
            f"in the video frames. Focus on the key elements, actions, and messages conveyed "
            f"in the advertisement.\n Question: {question}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, 
                       default='/root/studies/competitions/MARS2/VR-Ads/datasets')  # 视频文件存放目录
    parser.add_argument('--file_dir', type=str, 
                       default='../adsqa_full_set')                                 # 视频语音识别文本、视频问答数据的目录
    parser.add_argument('--model_dir', type=str, 
                       default="/root/studies/models/Qwen2.5-VL-72B-Instruct")      # 模型路径
    parser.add_argument('--model_name', type=str, 
                       default='qwen2d5-vl-72b')                                    # 以model的名字命名存储，方便区分模型效果。
    args = parser.parse_args()

    # 初始化随机种子
    initialize_random_seeds(42)

    # 加载数据
    asr_results_data = load_data(f'./{args.file_dir}/asr_set.json')
    test_questions_data = load_data(f'./{args.file_dir}/adsqa_question_file.json')

    # 初始化视频问答处理器
    qa_processor = VideoQAProcessor(args.model_dir)

    # 处理每个问题
    for index, question_item in enumerate(test_questions_data):
        print(f"处理第 {index} 个问题")
        start_time = time.time()
        
        video_name = question_item['video']
        question_text = question_item['question']
        question_id = question_item['question_id']

        # 获取语音识别结果
        asr_text = asr_results_data[video_name + '.mp4']
        if asr_text != '':
            asr_text = f"Voiceover: {asr_text}\n"

        # 检查结果文件是否已存在
        result_file_path = os.path.join('./result_competition', question_id, f'{args.model_name}.json')
        if os.path.exists(result_file_path):
            continue

        # 构建视频路径
        video_file_path = os.path.join(args.video_dir, video_name + '.mp4')

        # 准备视频输入
        qa_processor._prepare_video_inputs(video_file_path, video_name)
        
        print(f"视频输入形状: {qa_processor.video_inputs[0].shape}")
        print(f"ASR类型: {type(asr_text)}, 值: {asr_text}")
        print(f"问题类型: {type(question_text)}, 值: {question_text}")

        # 构建输入提示
        input_prompt = add_input_prompt(question_text)

        # 生成预测结果
        predictions_list = []
        for iteration in range(1):
            prediction = qa_processor.response(video_name, input_prompt)
            predictions_list.append(prediction)
        output_dir = os.path.join('./result_competition', question_id)
        create_output(output_dir)
        save(result_file_path, predictions_list)
        print(f"处理时间: {time.time() - start_time:.2f}秒")
if __name__ == "__main__":
    main()