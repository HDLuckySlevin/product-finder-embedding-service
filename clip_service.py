import os
import io
import base64
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from PIL import Image
import torch
import openai
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    filename="vector_output.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Load .env
load_dotenv()

# ENV
PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openclip")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
OPENAI_MODEL_IMAGE = os.getenv("OPENAI_MODEL_IMAGE", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROMPT = os.getenv("PROMPT", "Beschreibe präzise und sachlich ausschließlich das sichtbare Produkt auf dem Bild. Konzentriere dich auf Produktname, Farben, Materialien, sichtbare Strukturen, Formen und typische Nutzungshinweise. Wenn Text, Inhaltsstoffe, Logos, Marken oder Nummern wie EAN-Codes sichtbar sind, gib diese vollständig wieder. Vermeide jede Beschreibung von Personen, Körperteilen oder deren Interaktion mit dem Produkt. Nutze klare, maschinell interpretierbare Sprache in vollständigen Sätzen. Beschreibe den Hintergrund nur, wenn er für die Nutzung oder Erkennbarkeit des Produkts relevant ist. Nenne ausschließlich Merkmale, die im Bild eindeutig zu erkennen sind. Verzichte auf Interpretationen oder Bewertungen.")
PORT = int(os.getenv("PORT", "1337"))

print(f"🔧 Using embedding provider: {PROVIDER}")

model = None
preprocess = None
tokenizer = None

if PROVIDER == "openclip":
    import open_clip
    MODEL_NAME = os.getenv("MODEL_NAME", "ViT-L-14")
    PRETRAINED = os.getenv("PRETRAINED")

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()

elif PROVIDER == "openai":
    if not OPENAI_API_KEY:
        raise ValueError("❌ OPENAI_API_KEY is not set.")
    openai.api_key = OPENAI_API_KEY


@asynccontextmanager
async def lifespan(app: FastAPI):
    if PROVIDER == "openai":
        try:
            test_embed = openai.embeddings.create(input=["Hello"], model=OPENAI_MODEL)
            print(f"✅ OpenAI embedding model '{OPENAI_MODEL}' is reachable.")
        except Exception as e:
            print(f"❌ Failed to reach OpenAI embedding model '{OPENAI_MODEL}': {e}")

        try:
            if OPENAI_MODEL_IMAGE:
                test_chat = openai.chat.completions.create(
                    model=OPENAI_MODEL_IMAGE,
                    messages=[{"role": "user", "content": "Ping"}],
                    max_tokens=5
                )
                print(f"✅ OpenAI vision/chat model '{OPENAI_MODEL_IMAGE}' is reachable.")
        except Exception as e:
            print(f"❌ Failed to reach OpenAI image/chat model '{OPENAI_MODEL_IMAGE}': {e}")
    yield


app = FastAPI(title="Embedding Service", lifespan=lifespan)


class Texts(BaseModel):
    texts: List[str]


@app.post("/text-embedding")
async def text_embedding(payload: Texts):
    if PROVIDER == "openclip":
        tokens = tokenizer(payload.texts)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        vectors = emb.cpu().tolist()
        logging.info(f"TextEmbedding | Provider=openclip | VectorLength={len(vectors[0])} | Vectors={json.dumps(vectors)}")
        return {"vectors": vectors}

    elif PROVIDER == "openai":
        response = openai.embeddings.create(
            input=payload.texts,
            model=OPENAI_MODEL
        )
        vectors = [d.embedding for d in response.data]
        logging.info(f"TextEmbedding | Provider=openai | VectorLength={len(vectors[0])} | Vectors={json.dumps(vectors)}")
        return {"vectors": vectors}

    return {"error": "Invalid EMBEDDING_PROVIDER setting"}


@app.post("/image-embedding")
async def image_embedding(file: UploadFile = File(...)):
    data = await file.read()

    if PROVIDER == "openclip":
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img_t = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            emb = model.encode_image(img_t)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        vector = emb.cpu().tolist()[0]
        logging.info(f"ImageEmbedding | Provider=openclip | VectorLength={len(vector)} | Vector={json.dumps(vector)}")
        return {"vector": vector}

    elif PROVIDER == "openai":
        base64_img = base64.b64encode(data).decode("utf-8")
        try:
            response = openai.chat.completions.create(
                model=OPENAI_MODEL_IMAGE,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": "high"}}
                        ]
                    }
                ],
                max_tokens=300
            )
            description = response.choices[0].message.content.strip()

            embed_response = openai.embeddings.create(
                input=[description],
                model=OPENAI_MODEL
            )
            vector = embed_response.data[0].embedding
            logging.info(f"ImageEmbedding | Provider=openai | Description=\"{description}\" | VectorLength={len(vector)} | Vector={json.dumps(vector)}")

            return {
                "description": description,
                "vector": vector
            }

        except Exception as e:
            return {"error": f"OpenAI image description failed: {e}"}

    return {"error": "Invalid EMBEDDING_PROVIDER setting"}


@app.get("/healthstatus")
async def healthstatus():
    return {"status": "It works", "provider": PROVIDER}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("clip_service:app", host="0.0.0.0", port=PORT)
