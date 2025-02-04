from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
import torch
import gc  # For clearing memory
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BlipProcessor, BlipForConditionalGeneration
)
from PIL import Image
import io
from diffusers import StableDiffusionPipeline
from datetime import datetime
import random
import string

# Initialize FastAPI App
app = FastAPI(title="AI Model API", version="3.1", description="Serving DeepSeek-R1 & Janus-Pro-7B with FastAPI")

from fastapi.staticfiles import StaticFiles
import os

# Ensure 'static' directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Device Selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths to Local Models
DEEPSEEK_MODEL_PATH = "/home/ubuntu/models/deepseek_r1_distill_qwen_7b"
JANUS_MODEL_PATH = "/home/ubuntu/models/janus_pro_7b"
JANUS_VISION_MODEL = "Salesforce/blip-image-captioning-large"
JANUS_TEXT2IMAGE_MODEL = "CompVis/stable-diffusion-v1-4"

EC2_PUBLIC_IP = "3.80.102.182"

# Optional API Key
API_KEY = "1234a"

# Request Schema
class TextRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.7

# 🔹 Helper function to free GPU memory
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

# ✅ Helper function for API key validation
def validate_api_key(api_key: str = Header(None)):
    if api_key and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# ✅ Function to generate a unique filename
def generate_unique_filename():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=5))  # 5-char random string
    return f"generated_image_{timestamp}_{random_suffix}.png"

# ✅ Load & Unload DeepSeek-R1
def process_deepseek_text(request: TextRequest):
    try:
        print("Loading DeepSeek-R1 model...")
        tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(DEEPSEEK_MODEL_PATH, torch_dtype=torch.float16).to(device)

        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_length=request.max_length, temperature=request.temperature)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Free up GPU memory
        del model, tokenizer
        clear_gpu_memory()

        return {"model": "deepseek", "generated_text": generated_text}
    except Exception as e:
        return {"error": str(e)}

# ✅ Load & Unload Janus-Pro-7B (Text)
def process_janus_text(request: TextRequest):
    try:
        print("Loading Janus-Pro-7B (Text) model...")
        tokenizer = AutoTokenizer.from_pretrained(JANUS_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(JANUS_MODEL_PATH, torch_dtype=torch.float16).to(device)

        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_length=request.max_length, temperature=request.temperature)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Free up GPU memory
        del model, tokenizer
        clear_gpu_memory()

        return {"model": "janus", "generated_text": generated_text}
    except Exception as e:
        return {"error": str(e)}

# ✅ Load & Unload Janus-Pro-7B (Vision)
def process_janus_image_caption(file: UploadFile):
    try:
        print("Loading Janus-Pro-7B (Vision) model...")
        processor = BlipProcessor.from_pretrained(JANUS_VISION_MODEL)
        model = BlipForConditionalGeneration.from_pretrained(JANUS_VISION_MODEL).to(device)

        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

        # Free up GPU memory
        del model, processor
        clear_gpu_memory()

        return {"model": "janus", "image_caption": caption}
    except Exception as e:
        return {"error": str(e)}

# ✅ Enhance Text-to-Image Prompts using DeepSeek
def enhance_prompt_with_deepseek(request: TextRequest):
    """
    Uses DeepSeek model to enhance the input prompt for better image generation.
    """
    try:
        print("Enhancing prompt using DeepSeek...")
        tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(DEEPSEEK_MODEL_PATH, torch_dtype=torch.float16).to(device)

        # Modify prompt for better image generation
        enhanced_input = (
            f"Enhance this prompt for a high-quality AI-generated art: {request.prompt}. "
            f"Ensure the output is a single, focused image with no duplicates or multiple iterations."
            f"STRICTLY ensure that the output is a single focused and high-quality image. DO NOT output duplicates or multiple iterations."
            f"Always output a SINGLE image if it is an avatar, ONLY output ONE iteration."
            f"Generate a SINGLE, high-quality, AI-generated image based on this prompt: {request.prompt}. "
            f"Do NOT generate multiple iterations. The output must be ONE SINGLE image, NOT variations."
        )
        inputs = tokenizer(enhanced_input, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_length=request.max_length, temperature=request.temperature)
        enhanced_prompt = tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"Original Prompt: {request.prompt}")
        print(f"Enhanced Prompt: {enhanced_prompt}")

        # Free up GPU memory
        del model, tokenizer
        clear_gpu_memory()

        return enhanced_prompt

    except Exception as e:
        print(f"Error enhancing prompt: {e}")
        return request.prompt  # If enhancement fails, use original prompt

# ✅ Load & Unload Janus-Pro-7B (Text-to-Image)
def process_janus_text2image(request: TextRequest):
    try:
        print("Loading Janus-Pro-7B Text-to-Image model...")

        # Enhance prompt using DeepSeek
        enhanced_prompt = enhance_prompt_with_deepseek(request)

        # Check GPU availability
        if torch.cuda.is_available():
            print("Running Stable Diffusion on GPU...")
            pipeline = StableDiffusionPipeline.from_pretrained(
                JANUS_TEXT2IMAGE_MODEL,
                revision="fp16",
                torch_dtype=torch.float16
            ).to("cuda")
        else:
            print("Running Stable Diffusion on CPU...")
            pipeline = StableDiffusionPipeline.from_pretrained(
                JANUS_TEXT2IMAGE_MODEL
            ).to("cpu")  # ✅ No float16 for CPU

        # ✅ Ensure only one image is generated
        image = pipeline(
            enhanced_prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            num_images_per_prompt=1  # ✅ This ensures only one image is generated
        ).images[0]

        # ✅ Save the image in the "static" folder
        static_dir = "static"
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        # Generate a unique filename
        image_filename = generate_unique_filename()
        image_path = os.path.join(static_dir, image_filename)
        image.save(image_path)

        # ✅ Free memory
        del pipeline
        clear_gpu_memory()

        return {"model": "janus", "image_url": f"http://{EC2_PUBLIC_IP}:8000/{image_path}"}

    except torch.cuda.OutOfMemoryError:
        return {"error": "CUDA Out of Memory. Please reduce batch size or try again later."}
    except Exception as e:
        return {"error": str(e)}


# 🚀 API Endpoints
# 🚀 API Endpoints with Optional API Key
@app.post("/deepseek/generate-text")
async def generate_text_deepseek(request: TextRequest, api_key: str = Header(None)):
    validate_api_key(api_key)  # ✅ Validate API key (optional)
    return process_deepseek_text(request)

@app.post("/janus/generate-text")
async def generate_text_janus(request: TextRequest, api_key: str = Header(None)):
    validate_api_key(api_key)  # ✅ Validate API key (optional)
    return process_janus_text(request)

@app.post("/janus/generate-image")
async def generate_image_janus(request: TextRequest, api_key: str = Header(None)):
    validate_api_key(api_key)  # ✅ Validate API key (optional)
    return process_janus_text2image(request)

@app.get("/")
def health_check():
    return {
        "status": "API is running!",
        "available_models": ["deepseek-text", "janus-text", "janus-text-to-image"],
    }
