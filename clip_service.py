from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from PIL import Image
import open_clip
import torch
import io
import os

from dotenv import load_dotenv
load_dotenv()

PORT = int(os.getenv("PORT", 1337))
MODEL_NAME = os.getenv("MODEL_NAME", "ViT-L-14")
PRETRAINED = os.getenv("PRETRAINED", "laion2b_s32b_b82k")

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model.eval()

app = FastAPI(title="OpenCLIP Embedding Service")

class Texts(BaseModel):
    texts: List[str]

@app.post("/text-embedding")
async def text_embedding(payload: Texts):
    tokens = tokenizer(payload.texts)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return {"vectors": emb.cpu().tolist()}

@app.post("/image-embedding")
async def image_embedding(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img_t = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_image(img_t)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return {"vector": emb.cpu().tolist()[0]}

@app.get("/healthstatus")
async def healthstatus():
    return {"status": "It works"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("clip_service:app", host="0.0.0.0", port=PORT)
