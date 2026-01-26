# -*- coding: utf-8 -*-
"""
Complimentary Machine - RunPod Serverless Handler
Uses a fine-tuned VLM model from Hugging Face Hub.
"""

import os
import io
import base64
import requests
import torch
import runpod
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# --- Configuration ---
# 替换成你的 Hugging Face 模型仓库路径
MODEL_ID = os.environ.get("HF_MODEL_ID", "your-username/cm-gallery-vlm")
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # 如果是私有仓库需要token

# --- Global Model Variables (loaded once during cold start) ---
model = None
processor = None
device = None


def load_model():
    """
    Load model from Hugging Face Hub.
    This function is called once during cold start.
    """
    global model, processor, device
    
    if model is not None:
        return  # Model already loaded
    
    print(f"🚀 Loading VLM Model from Hugging Face: {MODEL_ID}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    try:
        # Load model from Hugging Face Hub
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


def load_image(image_input):
    """
    Load image from base64 string or URL.
    
    Args:
        image_input: Either a base64 encoded string or a URL
        
    Returns:
        PIL Image object
    """
    if image_input.startswith(('http://', 'https://')):
        # Load from URL
        response = requests.get(image_input, timeout=30)
        response.raise_for_status()
        pil_image = Image.open(io.BytesIO(response.content))
    else:
        # Assume base64 encoded
        # Handle data URI format: "data:image/jpeg;base64,..."
        if ',' in image_input and image_input.startswith('data:'):
            image_input = image_input.split(',', 1)[1]
        
        image_bytes = base64.b64decode(image_input)
        pil_image = Image.open(io.BytesIO(image_bytes))
    
    return pil_image.convert("RGB")


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


def handler(job):
    """
    RunPod Serverless handler function.
    
    Expected input format:
    {
        "input": {
            "image": "<base64_string or URL>",
            "user_text": "optional user text"  # optional
        }
    }
    
    Returns:
    {
        "compliment": "generated compliment text"
    }
    """
    try:
        job_input = job.get("input", {})
        
        # Validate input
        image_input = job_input.get("image")
        if not image_input:
            return {"error": "Missing required field: 'image' (base64 string or URL)"}
        
        user_text = job_input.get("user_text", None)
        
        # Load image
        try:
            pil_image = load_image(image_input)
        except Exception as e:
            return {"error": f"Failed to load image: {str(e)}"}
        
        # Generate compliment
        compliment = generate_compliment(pil_image, user_text)
        
        return {
            "compliment": compliment
        }
        
    except Exception as e:
        return {"error": str(e)}


# --- RunPod Serverless Entry Point ---
if __name__ == "__main__":
    # Load model during cold start
    load_model()
    
    # Start serverless handler
    runpod.serverless.start({
        "handler": handler
    })
