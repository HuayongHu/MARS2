from transformers import WhisperProcessor, WhisperForConditionalGeneration
import whisper
import openai
import re
import time
import os
import json
from typing import Optional
openai.api_key = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
openai.api_base = "https://api3.apifans.com/v1"


def initialize_openai(api_key: str):
    openai.api_key = api_key


def Check(text: str) -> bool: # 对文本进行检查
    cleaned = re.sub(r'[^\w]', '', text.lower())
    if len(cleaned) < 3:
        return True
    if len(set(cleaned)) == 1:
        return True
    if len(cleaned) > 6:
        for i in range(min(3, len(cleaned) - 2)):
            pattern = cleaned[i:i + 3]
            if cleaned.count(pattern) > len(cleaned) * 0.4:
                return True
    return False


def process_text(text: str) -> Optional[str]:  # 对文本进行进一步处理转换成英文格式方便统一处理。
    if not text or text.isspace():
        return None
    processed_text = ' '.join(text.strip().split())
    if Check(processed_text):
        return None
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intelligent text processor.  Process the user's ASR-generated text strictly according to the following rules:\n"
                        "1. **English Input:** If the input text is entirely in English, return the original text unchanged.\n"
                        "2.  **Non-English Input:** If the input text is in any language other than English, translate it accurately into English.\n"
                        "3. **Noise/Repetition:** If the input consists solely of meaningless repetition, gibberish, or non-linguistic noise, return `[FILTERED]`.\n"
                        "**Output Requirements:**"
                        "1.*   Return ONLY the processed text (either the original English, the English translation, or `[FILTERED]`).\n"
                        "2.*   DO NOT include any explanations, notes, additional text, or formatting beyond the required output.\n"
                        "3.*   Preserve the original casing and punctuation (where applicable after translation).\n"
                    )
                },
                {"role": "user", "content": processed_text}
            ],
            temperature=0.1,
            max_tokens=2048
        )

        result = response.choices[0].message.content.strip()
        if result == "[FILTERED]":
            return None

        return result if result else None

    except Exception as e:
        print(f"Error processing text: {e}")
        return None
def read_json(ppath):
    with open(ppath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():
    """主函数：执行ASR处理和翻译的完整流程"""

    print("=== 开始ASR语音识别处理 ===")
    print("正在加载Whisper模型...")

    processor = WhisperProcessor.from_pretrained("/root/studies/models/whisper-large-v3-turbo")
    model = WhisperForConditionalGeneration.from_pretrained("/root/studies/models/whisper-large-v3-turbo").cuda()
    model.config.forced_decoder_ids = None
    print("Whisper模型加载完成")
    path_list = os.listdir('/root/studies/competitions/MARS2/VR-Ads/datasets')
    # 第一阶段：ASR处理
    print("进行第一阶段处理...")
    processed_count = 0
    for filee in path_list:
        if filee.endswith('.md') or os.path.exists(os.path.join('../asr_results', filee.replace('.mp4', ''), 'asr_original.json')):  # 中断处理后重新运行时直接加载已经处理过内容，省时间
            continue
        try:
            vdir = filee.replace('.mp4', '')
            print(f"正在处理ASR: {filee}")
            # 加载音频并进行语音识别
            audio = whisper.load_audio(os.path.join('/root/studies/competitions/MARS2/VR-Ads/datasets', vdir + '.mp4'))
            input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.cuda()
            id = model.generate(input_features)
            # 解码为文本
            transcription = processor.batch_decode(id, skip_special_tokens=True)
            asr_results = {vdir: transcription[0]}
            print(f"ASR结果: {transcription[0]}")
            # 创建输出目录并保存结果
            if not os.path.exists(os.path.join('../asr_results', vdir)):
                os.makedirs(os.path.join('../asr_results', vdir))
            with open(os.path.join('../asr_results', vdir, 'asr_original.json'), 'w', encoding='utf-8') as ff:
                json.dump(asr_results, ff, ensure_ascii=False)
            processed_count += 1
        except RuntimeError as e:
            print(f"ASR处理错误 {filee}: {e}")
    print(f"ASR处理完成，共处理 {processed_count} 个文件")
    
    # 第二阶段：翻译处理
    print("\n=== 开始翻译处理 ===")
    path_list5 = os.listdir('/root/studies/competitions/MARS2/VR-Ads/datasets')
    path_list_clean = [pp.replace('.mp4', '') for pp in path_list5 if pp.endswith('.mp4')]
    translation_processed_count = 0
    for ii, vpath in enumerate(path_list_clean):
        if os.path.exists(os.path.join('../asr_results', vpath, 'asr_results_cleaned.json')) or '.md' in vpath:
            continue
        print(f"正在处理翻译 ({ii+1}/{len(path_list_clean)}): {vpath}")
        start = time.time()
        try:
            # 读取ASR原始结果
            asr_raw_path = os.path.join('../asr_results', vpath, 'asr_original.json')
            if not os.path.exists(asr_raw_path):
                print(f"ASR文件不存在，跳过: {vpath}")
                continue
            rawasr_data = read_json(asr_raw_path)
            rawasr = list(rawasr_data.values())[0] if rawasr_data else ""
            print(f"原始ASR文本: {rawasr}")
            result = process_text(rawasr)
            print(f"翻译结果: {result}")
            # 保存翻译结果
            if result is not None:
                result_data = {vpath: result}
            else:
                result_data = {vpath: ""}
            if not os.path.exists(os.path.join('../asr_results', vpath)):
                os.makedirs(os.path.join('../asr_results', vpath))
                
            with open(os.path.join('../asr_results', vpath, 'asr_results_cleaned.json'), 'w', encoding='utf-8') as ff:
                json.dump(result_data, ff, ensure_ascii=False, indent=4)
            translation_processed_count += 1
            processing_time = time.time() - start
            print(f"处理时间: {processing_time:.2f}秒")
        except Exception as e:
            print(f"翻译处理错误 {vpath}: {e}")
    print(f"\n翻译处理完成，共处理 {translation_processed_count} 个文件")
    print("=== 所有处理完成 ===")

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"\n总处理时间: {total_time:.2f}秒")