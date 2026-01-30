# -*- coding: utf-8 -*-
"""
Complimentary Machine - PC Server (Merged VLM Version)
Runs on your powerful PC with NVIDIA GPU.
Uses a merged fine-tuned VLM model directly without separate LoRA adapter.
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
MODEL_PATH = "GRMD/complimentary-machine-vlm"  # Merged fine-tuned VLM model
PORT = 8000

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
print("Loading Merged VLM Model...")

# Device selection: CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
    print("🖥️  Using NVIDIA CUDA GPU")
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16  # MPS works better with float16
    print("🍎 Using Apple Silicon MPS (M1/M2/M3/M4)")
else:
    device = "cpu"
    dtype = torch.float32
    print("💻 Using CPU (this will be slow)")

try:
    # Load merged model directly (no LoRA needed)
    # Note: device_map="auto" works best for CUDA; for MPS/CPU we load then move
    if device == "cuda":
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto"
        )
    else:
        # For MPS and CPU, load without device_map then move to device
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model.eval()
    print(f"✅ Model Loaded Successfully on {device.upper()}!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


def generate_compliment(pil_image, user_text=None):
    """
    Generate compliment directly from image using the fine-tuned VLM.
    The merged model has learned to give compliments based on images.
    """
    # Build prompt - the model was trained to handle images directly
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
        # Read Image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Generate compliment directly from image
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
        "model": MODEL_PATH,
        "device": device
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
