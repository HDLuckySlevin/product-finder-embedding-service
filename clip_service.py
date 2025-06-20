import os
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from PIL import Image
import torch
import io
import openai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Embedding Service")

# ENV-Variablen
PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openclip")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PORT = int(os.getenv("PORT"))

if PROVIDER == "openclip":
    import open_clip
    MODEL_NAME = os.getenv("MODEL_NAME", "ViT-L-14")
    PRETRAINED = os.getenv("PRETRAINED", "laion2b_s32b_b82k")

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()

if PROVIDER == "openai":
    openai.api_key = OPENAI_API_KEY


class Texts(BaseModel):
    texts: List[str]

@app.post("/text-embedding")
async def text_embedding(payload: Texts):
    if PROVIDER == "openclip":
        tokens = tokenizer(payload.texts)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return {"vectors": emb.cpu().tolist()}

    elif PROVIDER == "openai":
        response = openai.embeddings.create(
            input=payload.texts,
            model=OPENAI_MODEL
        )
        vectors = [d.embedding for d in response.data]
        return {"vectors": vectors}

    else:
        return {"error": "Invalid EMBEDDING_PROVIDER setting"}

@app.post("/image-embedding")
async def image_embedding(file: UploadFile = File(...)):
    if PROVIDER != "openclip":
        return {"error": "Image embedding only available with OpenCLIP"}

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