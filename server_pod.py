# -*- coding: utf-8 -*-
"""
Complimentary Machine - RunPod Pods Server
直接在 RunPod Pods 上运行的 FastAPI 服务器
从 Hugging Face Hub 加载模型
"""

import os
import io
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

# --- Configuration ---
MODEL_ID = os.environ.get("HF_MODEL_ID", "GRMD/cm-gallery-vlm")
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # 私有仓库需要
PORT = int(os.environ.get("PORT", 8000))

app = FastAPI(title="Complimentary Machine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Model Variables ---
model = None
processor = None
device = None


def load_model():
    """Load model from Hugging Face Hub"""
    global model, processor, device
    
    print(f"🚀 Loading VLM Model from Hugging Face: {MODEL_ID}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            token=HF_TOKEN
        )
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True,
            token=HF_TOKEN
        )
        
        model.eval()
        print(f"✅ Model Loaded Successfully on {device}!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e


def generate_compliment(pil_image, user_text=None):
    """Generate compliment from image using the fine-tuned VLM."""
    if user_text:
        text_content = f"给这个场景一个赞美。用户说：{user_text}"
    else:
        text_content = "给这个场景一个温暖的赞美。"
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": text_content}
        ]
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=True, 
            temperature=0.8,
            top_p=0.9
        )
        
    trim_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trim_ids, skip_special_tokens=True)[0].strip()


@app.post("/compliment")
async def api_generate_compliment(
    image: UploadFile = File(...),
    user_text: str = Form(None)
):
    if not model: 
        return {"error": "Model not loaded"}
        
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        compliment = generate_compliment(pil_image, user_text)
        return {"compliment": compliment}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok" if model else "error",
        "model": MODEL_ID,
        "device": str(device)
    }


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
