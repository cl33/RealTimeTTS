import os
import queue
import uuid

from RealtimeSTT import AudioToTextRecorder
from indextts.infer import IndexTTS
import json
import requests
import sounddevice as sd
import torchaudio
import threading


OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-R1:8b"

# 全局音频队列和播放线程
audio_queue = queue.Queue()
play_thread_running = True  # 控制播放线程的标志

def play_audio_thread():
    """播放线程：逐个播放队列中的音频"""
    while play_thread_running:
        try:
            audio_data, sample_rate = audio_queue.get()
            sd.play(audio_data, sample_rate)
            sd.wait()  # 等待当前音频播放完成
            audio_queue.task_done()
        except queue.Empty:
            pass

def stream_text_response(prompt):
    """流式获取Ollama模型回复（过滤<think>标签）"""
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "max_tokens": 50
        },
        stream=True
    )
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8').strip()
            data = json.loads(line)
            if "response" in data:
                if "<think>" in data["response"] or "</think>" in data["response"]:
                    data["response"] = data["response"].replace("<think>", "")
                    data["response"] = data["response"].replace("</think>", "")
                yield data["response"]

def find_split_point(text):
    """查找文本中的第一个分割符（句号、逗号、换行符等）的位置"""
    for idx, char in enumerate(text):
        if char in {".", ",", "\n", "，", "。", "?", "？", "！", "！"}:
            return idx
    return -1  # 未找到分割符

def synthesize_and_play(text, tts_model, reference_voice):
    """合成语音并加入播放队列"""
    if not text:
        return
    print(f"Synthesizing: {text}")
    uuid_name = uuid.getnode()
    voice_name = f"{uuid_name}.wav"
    tts_model.infer(
        reference_voice,
        text,
        output_path=voice_name
    )
    # 加载音频并加入队列
    audio, sample_rate = torchaudio.load(voice_name)
    audio_np = audio.numpy().squeeze()
    audio_queue.put((audio_np, sample_rate))
    # 删除临时文件
    os.remove(voice_name)

def real_time_interactive(recorder, tts_model, reference_voice):
    while True:
        text = recorder.text()
        if text:
            print(f"Transcribed Text: {text}")
            current_segment = ""  # 当前累积的文本段
            for chunk in stream_text_response(text):
                print(f"Streamed Text: {chunk}")
                current_segment += chunk  # 累积新文本
                while True:
                    split_pos = find_split_point(current_segment)
                    if split_pos == -1:
                        break  # 无分割符，继续累积
                    # 提取到分割符的段落
                    segment = current_segment[:split_pos + 1].strip()
                    remaining = current_segment[split_pos + 1:]  # 剩余文本
                    current_segment = remaining  # 更新缓冲区
                    # 合成并加入播放队列
                    synthesize_and_play(segment, tts_model, reference_voice)
            # 处理剩余未分割的文本
            if current_segment:
                synthesize_and_play(current_segment, tts_model, reference_voice)

def main():
    tts_instance = None
    stt_instance = None

    def load_index_tts():
        nonlocal tts_instance
        try:
            tts_instance = IndexTTS(
                model_dir="checkpoints",
                cfg_path="checkpoints/config.yaml"
            )
            print("IndexTTS loaded successfully")
        except Exception as e:
            print(f"Error loading IndexTTS: {e}")

    def load_realtime_stt():
        nonlocal stt_instance
        try:
            stt_instance = AudioToTextRecorder(
                model="iic/faster-whisper-medium",
                download_root="./iic",
                device="cuda",
                gpu_device_index=0,
            )
            print("RealtimeSTT loaded successfully")
        except Exception as e:
            print(f"Error loading RealtimeSTT: {e}")

    # 启动模型加载线程
    tts_thread = threading.Thread(target=load_index_tts)
    stt_thread = threading.Thread(target=load_realtime_stt)
    tts_thread.start()
    stt_thread.start()
    tts_thread.join(30)
    stt_thread.join(30)

    if not (tts_instance and stt_instance):
        print("Failed to load models. Exiting.")
        return

    # 启动播放线程
    play_thread = threading.Thread(target=play_audio_thread, daemon=True)
    play_thread.start()

    reference_voice = "1.wav"
    real_time_interactive(stt_instance, tts_instance, reference_voice)

    # 等待队列处理完毕
    audio_queue.join()
    global play_thread_running
    play_thread_running = False

if __name__ == "__main__":
    main()