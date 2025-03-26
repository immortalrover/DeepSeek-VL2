from logging import debug
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from typing import AsyncGenerator, Any
import asyncio
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM
from fastapi.templating import Jinja2Templates
import tempfile
import os
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.serve.app_modules.utils import parse_ref_bbox
from contextlib import asynccontextmanager
import PIL.Image
import uvicorn
import base64
from io import BytesIO
import binascii
import json
# Global state to hold the model and processor
model_state = {
    # "vl_gpt": None,
    # "vl_chat_processor": None,
    # "tokenizer": None,
    # "dtype": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI to load and unload the model."""
    dtype = torch.bfloat16
    model_path = "deepseek-vl2-tiny"

    # Load the model and processor
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype
    ).cuda().eval()
    model_state["vl_chat_processor"] = vl_chat_processor
    model_state["tokenizer"] = tokenizer
    model_state["vl_gpt"] = vl_gpt
    model_state["dtype"] = dtype

    yield  # FastAPI will keep the model loaded during runtime

    # Cleanup (optional, but good practice)
    del vl_gpt, vl_chat_processor, tokenizer
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

class MediaType(str):
    VIDEO = "video"
    IMAGE = "image"

class MediaProcessor:
    def __init__(self, data: bytes, question: str):
        self.raw_data = data
        self.question = question
        self._validate()

    def _validate(self):
        if not isinstance(self.raw_data, bytes):
            raise MediaProcessingError("Input data must be bytes", self.raw_data)

class VideoProcessor(MediaProcessor):
    async def async_process(self) -> AsyncGenerator[dict, None]:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as tmp_video:
            tmp_video.write(self.raw_data)  # 将原始视频数据写入临时文件
            tmp_video.flush()  # 确保数据写入磁盘

            fps = 1/4

            frame_dir = tempfile.mkdtemp()
            cmd = [
                'ffmpeg', '-i', tmp_video.name,  # 输入视频文件
                '-vf', f'fps={fps}',  # 设置帧率过滤器
                '-q:v', '2',  # 设置输出质量(2表示高质量)
                os.path.join(frame_dir, 'frame_%04d.jpg')  # 输出帧文件名格式
            ]
            process = await asyncio.create_subprocess_exec(*cmd)  # 启动ffmpeg进程
            await process.wait()  # 等待帧提取完成

            for frame_file in sorted(os.listdir(frame_dir)):  # 按文件名排序处理
                with open(os.path.join(frame_dir, frame_file), 'rb') as f:
                    frame_data = f.read()  # 读取帧数据
                async for result in ImageProcessor(frame_data).async_process():
                    yield result
                await asyncio.sleep(0)  # 确保异步执行

            os.unlink(tmp_video.name)  # 删除临时视频文件
            for frame_file in os.listdir(frame_dir):
                os.unlink(os.path.join(frame_dir, frame_file))  # 删除所有帧文件
            os.rmdir(frame_dir)  # 删除临时目录

class ImageProcessor(MediaProcessor):
    def _validate(self):
        if not (self.raw_data.startswith(b'\xFF\xD8\xFF') or  # JPEG
                self.raw_data.startswith(b'\x89PNG\r\n\x1A\n') or  # PNG
                self.raw_data.startswith(b'GIF87a') or  # GIF
                self.raw_data.startswith(b'GIF89a') or  # GIF
                self.raw_data.startswith(b'BM') or  # BMP
                self.raw_data.startswith(b'II*\x00') or  # TIFF
                self.raw_data.startswith(b'MM\x00*')):  # TIFF
            raise ImageDecodeError("Invalid image format", self.raw_data)

    async def async_process(self) -> AsyncGenerator[dict, None]:
        chunk_size = -1
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n<|grounding|>{self.question}",
                "images": "",
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        images = [PIL.Image.open(BytesIO(self.raw_data)).convert("RGB")]
        prepare_inputs = model_state["vl_chat_processor"].__call__(
            conversations=conversation,
            images=images,
            force_batchify=True,
            system_prompt=""
        ).to(model_state["vl_gpt"].device, dtype=model_state["dtype"])

        with torch.no_grad():

            if chunk_size == -1:
                inputs_embeds = model_state["vl_gpt"].prepare_inputs_embeds(**prepare_inputs)
                past_key_values = None
            else:
                inputs_embeds, past_key_values = model_state["vl_gpt"].incremental_prefilling(
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    chunk_size=chunk_size
                )

            # run the model to get the response
            outputs = model_state["vl_gpt"].generate(
                inputs_embeds=inputs_embeds,
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                past_key_values=past_key_values,

                pad_token_id=model_state["tokenizer"].eos_token_id,
                bos_token_id=model_state["tokenizer"].bos_token_id,
                eos_token_id=model_state["tokenizer"].eos_token_id,
                max_new_tokens=512,
                # do_sample=False,
                # repetition_penalty=1.1,
                use_cache=True,
            )

            answer = model_state["tokenizer"].decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)
            yield {
                "type": "image",
                "answer": answer,
                "data": base64.b64encode(self.raw_data).decode('utf-8'),
                "question": self.question,
            }

        await asyncio.sleep(0)

@app.websocket("/media-process")
async def media_processing_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            for b64_str, filename, _type , question in zip(payload["images"], payload["filenames"], payload["types"], payload["question"]):
                if ';base64,' in b64_str:
                    header, data = b64_str.split(';base64,', 1)
                else:
                    data = b64_str

                # 解码并保存
                file_data = base64.b64decode(data)
                if (_type == "image"):
                    async for result in ImageProcessor(file_data, question).async_process():
                        await websocket.send_json(result)
                elif (_type == "video"):
                    async for result in VideoProcessor(file_data, question).async_process():
                        await websocket.send_json(result)

    finally:
        await websocket.close()


# 异常体系
class MediaProcessingError(Exception):
    def __init__(self, msg: str, data: bytes):
        super().__init__(f"{msg} (data: {len(data)} bytes)")
        self.data = data

class VideoDecodeError(MediaProcessingError): pass
class ImageDecodeError(MediaProcessingError): pass

# 前端页面
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>WebSocket Image Upload</title>
        <style>
        #container {
            padding: 20px;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border: 1px solid #ccc;
        }
        img {
            max-width: 300px;
            margin-top: 5px;
        }
        </style>
    </head>
    <body>
        <h1>Upload Multiple Images via WebSocket</h1>
        <input type="file" id="fileInput" multiple accept="image/*">
        <div class="input-container">
            <input
                type="text"
                id="input-box"
                placeholder="请输入内容..."
                maxlength="50"
            >
        </div>
        <button onclick="sendImages()">Send Images</button>
        <div id="container"></div>
        <script>
            const ws = new WebSocket('ws://' + window.location.host + '/media-process');
            const container = document.getElementById('container');
            function createMessageElement(type) {
                const element = document.createElement('div');
                element.className = `message ${type}`;
                return element;
            }
            function handleTextMessage(data) {
                const element = createMessageElement('text');
                element.textContent = `文本消息: ${data}`;
                container.appendChild(element);
            }
            function handleImageMessage(base64Data) {
                const element = createMessageElement('image');
                element.textContent = '图片消息:';
                const img = document.createElement('img');
                img.src = `data:image/png;base64,${base64Data}`;
                element.appendChild(img);
                container.appendChild(element);
            }
            async function sendImages() {
                const files = document.getElementById('fileInput').files;
                if (files.length === 0) {
                    alert('请选择至少一张图片');
                    return;
                };
                const question = document.getElementById('input-box').value;
                if (question.length === 0) {
                    question = "描述图片";
                };

                const readers = Array.from(files).map(file => {
                    return new Promise((resolve) => {
                        const reader = new FileReader();
                        reader.onload = () => resolve(reader.result);
                        reader.readAsDataURL(file);
                    });
                });

                try {
                    const base64Results = await Promise.all(readers);
                    const payload = {
                        images: base64Results,
                        filenames: Array.from(files).map(f => f.name),
                        types: Array.from(files).map(f => {
                            const ext = f.name.split('.').pop().toLowerCase();
                            return (['jpg','jpeg','png'].includes(ext)) ? 'image' : 'video';
                        }),
                        question: question
                    };
                    ws.send(JSON.stringify(payload));
                    alert('发送成功！');
                } catch (error) {
                    alert('文件读取失败: ' + error);
                }
            }

            ws.onmessage = function(event) {
                console.log('服务器响应:', event.data);
                try {
                    const message = JSON.parse(event.data);

                    switch (message.type.toLowerCase()) {
                        case 'text':
                            handleTextMessage(message.data);
                            break;
                        case 'image':
                            handleImageMessage(message.data);
                            handleTextMessage(message.question);
                            handleTextMessage(message.answer);
                            break;
                        default:
                            console.warn('未知消息类型:', message.type);
                    }
                } catch (error) {
                    console.error('消息解析失败:', error);
                }
            };
        </script>
    </body>
</html>
"""

@app.get("/")
async def root():
    return HTMLResponse(html)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
