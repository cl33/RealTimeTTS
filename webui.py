import os
import shutil
import sys
import threading
import time
import webbrowser

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import gradio as gr
from utils.webui_utils import next_page, prev_page

from indextts.infer import IndexTTS
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="zh_CN")
MODE = 'local'
tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)

def infer(voice, text, output_path=None):
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    tts.infer(voice, text, output_path)
    return output_path

def gen_single(prompt, text):
    output_path = infer(prompt, text)
    return gr.update(value=output_path, visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button


with gr.Blocks(theme=gr.themes.Soft(), css="""
body {
    background-color: #0f0f0f;
    color: #f5f5f5;
    font-family: 'Helvetica Neue', sans-serif;
}

h2, .gr-button {
    color: #f5f5f5;
}

.gr-button {
    background: #ff5f1f;
    border-radius: 8px;
    font-weight: bold;
}

.gr-textbox input {
    background-color: #1f1f1f;
    color: #fff;
    border: 1px solid #444;
}

.gr-audio {
    border: 1px solid #444;
    border-radius: 10px;
    background: #181818;
}
""") as demo:

    mutex = threading.Lock()
    gr.HTML('''
<div style="text-align: center; padding-top: 10px; line-height: 1.6;">
    <h1 style="font-size: 28px; font-weight: bold; color: #FF3C3C;">王知风 · IndexTTS 整合包</h1>
    <p style="font-size: 16px; color: #cccccc;">全网首发 · 高阶可控 · 零样本语音克隆系统</p>
    <p style="font-size: 14px; color: #777;">声音生成不是技术展示，是人格注入。</p>
    <p style="margin-top: 6px;">
        <a href="https://wangzhifeng.vip" target="_blank" style="color: #999; text-decoration: underline;">AI工具大全：wangzhifeng.vip</a>
    </p>
</div>
''')


    with gr.Tab("音频生成"): 
        with gr.Row():
            prompt_audio = gr.Audio(label="上传参考音频", key="prompt_audio",
                                    sources=["upload", "microphone"], type="filepath")
            input_text_single = gr.Textbox(label="请输入目标文本", key="input_text_single",
                                           placeholder="请在此输入您想表达的句子，例如：我想要一种温柔中带点犀利的笑声。")

        with gr.Row():
            gen_button = gr.Button("生成语音", key="gen_button", interactive=True)
            output_audio = gr.Audio(label="生成结果", visible=False, key="output_audio")

    prompt_audio.upload(update_prompt_audio,
                        inputs=[],
                        outputs=[gen_button])

    gen_button.click(gen_single,
                     inputs=[prompt_audio, input_text_single],
                     outputs=[output_audio])

if __name__ == "__main__":
    port = 7860
    webbrowser.open(f"http://127.0.0.1:{port}")
    demo.queue(20)
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)