

# import os
# import io
# import base64
# import traceback
# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.concurrency import run_in_threadpool
# from dotenv import load_dotenv
# from PIL import Image
# import httpx

# from transformers import BlipProcessor, BlipForConditionalGeneration
# import torch

# # Load environment variables
# load_dotenv()

# app = FastAPI()

# API_KEY = os.getenv("OPENROUTER_API_KEY")
# if not API_KEY:
#     raise RuntimeError("OPENROUTER_API_KEY not set")

# # Constants
# OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# MODEL_NAME = "gpt-4o-mini"

# # Load BLIP model for image captioning
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# @app.post("/analyze-issue")
# async def analyze_issue(
#     image: UploadFile = File(...),
#     text: str = Form(default="")
# ):
#     try:
#         image_bytes = await image.read()

#         # Run BLIP captioning in a threadpool to avoid blocking event loop
#         def generate_caption():
#             pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#             inputs = processor(pil_image, return_tensors="pt")
#             out = model.generate(**inputs)
#             return processor.decode(out[0], skip_special_tokens=True)

#         caption = await run_in_threadpool(generate_caption)

#         # Build prompt
#         prompt = (
#             f"Here is an image description: {caption}\n"
#             f"User description: '{text}'\n"
#             "Please analyze the issue and answer the following:\n"
#             "1. Does image displays a valid civic issue? (Yes or No)\n"
#             "2. Provide a short description of the issue if it a valid civic issue else tell why the text is not related to the image\n"
#             "3. Which category does it belong to if it is civic issue otherwise ignore this part? (Road, Sanitation, Electricity, Water, Other)\n"
#         )

#         # OpenRouter API call using HTTPX
#         headers = {
#             "Authorization": f"Bearer {API_KEY}",
#             "Content-Type": "application/json"
#         }

#         json_data = {
#             "model": MODEL_NAME,
#             "messages": [
#                 {"role": "system", "content": "You are a helpful assistant that analyzes civic issues."},
#                 {"role": "user", "content": prompt}
#             ]
#         }

#         async with httpx.AsyncClient() as client:
#             response = await client.post(OPENROUTER_API_URL, headers=headers, json=json_data)
#             response.raise_for_status()
#             data = response.json()

#         ai_reply = data["choices"][0]["message"]["content"]

#         return JSONResponse(content={
#             "status": "success",
#             "image_caption": caption,
#             "ai_response": ai_reply
#         })

#     except httpx.HTTPStatusError as e:
#         traceback.print_exc()
#         return JSONResponse(
#             status_code=e.response.status_code,
#             content={"detail": f"OpenRouter API error: {e.response.text}"},
#         )
#     except Exception as e:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

import os
import io
import traceback
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv
from PIL import Image
import httpx

from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO

# Load environment variables
load_dotenv()

app = FastAPI()

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "gpt-4o-mini"

# Load BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load YOLOv8 model (using pretrained COCO weights)
yolo_model = YOLO('yolov8n.pt')  # lightweight model, replace with your custom if available

async def generate_caption(image_bytes: bytes) -> str:
    def _caption():
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(pil_image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
    return await run_in_threadpool(_caption)

async def detect_objects(image_bytes: bytes) -> str:
    def _detect():
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = yolo_model(pil_image)
        detected_labels = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = yolo_model.names[cls_id]
                detected_labels.append(f"{label} ({conf:.2f})")
        return ", ".join(detected_labels) if detected_labels else "No objects detected"
    return await run_in_threadpool(_detect)

@app.post("/analyze-issue")
async def analyze_issue(
    image: UploadFile = File(...),
    text: str = Form(default="")
):
    try:
        image_bytes = await image.read()

        # Run captioning and detection concurrently for speed
        caption_task = generate_caption(image_bytes)
        detection_task = detect_objects(image_bytes)

        caption, detected_objects = await caption_task, await detection_task

        # Build prompt for LLM
        prompt = (
            f"Image description: {caption}\n"
            f"Detected objects: {detected_objects}\n"
            f"User description: '{text}'\n"
            "Please analyze if this is a valid civic issue.\n"
            "1. does image shows a valid civic issue? (Yes or No)\n"
            "2. looking at caption,objects and text on scale of 0-100 what percentage is chance of it being a valid civic issue,you can be strict"
            "3. Provide a short description if yes, else explain why not.\n"
            "3. Categorize it: Road, Sanitation, Electricity, Water, Other or none.\n"
        )

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        json_data = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that analyzes civic issues."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4000
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=json_data)
            response.raise_for_status()
            data = response.json()

        ai_reply = data["choices"][0]["message"]["content"]

        return JSONResponse({
            "status": "success",
            "image_caption": caption,
            "detected_objects": detected_objects,
            "ai_response": ai_reply
        })

    except httpx.HTTPStatusError as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=e.response.status_code,
            content={"detail": f"OpenRouter API error: {e.response.text}"}
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    
