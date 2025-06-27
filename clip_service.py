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
    filename="debug.log",
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
MODEL_NAME = os.getenv("MODEL_NAME", "ViT-L-14")  # ensure defined for all providers
PRETRAINED = os.getenv("PRETRAINED")
PROMPT = os.getenv("PROMPT", "Beschreibe präzise und sachlich ausschließlich das sichtbare Produkt auf dem Bild. Konzentriere dich auf Produktname, Farben, Materialien, sichtbare Strukturen, Formen und typische Nutzungshinweise. Wenn Text, Inhaltsstoffe, Logos, Marken oder Nummern wie EAN-Codes sichtbar sind, gib diese vollständig wieder. Vermeide jede Beschreibung von Personen, Körperteilen oder deren Interaktion mit dem Produkt. Nutze klare, maschinell interpretierbare Sprache in vollständigen Sätzen. Beschreibe den Hintergrund nur, wenn er für die Nutzung oder Erkennbarkeit des Produkts relevant ist. Nenne ausschließlich Merkmale, die im Bild eindeutig zu erkennen sind. Verzichte auf Interpretationen oder Bewertungen. Schreibe Am ende die Produkt-Kategorie dazu und den Produkt-Namen. Antworte auf deutsch")
PORT = int(os.getenv("PORT", "1337"))
DIMENSION_OPENCLIP = int(os.getenv("DIMENSION_OPENCLIP"))
DIMENSION_OPENAI = int(os.getenv("DIMENSION_OPENAI"))

print(f"🔧 Using embedding provider: {PROVIDER}")

model = None
preprocess = None
tokenizer = None

if PROVIDER == "openclip":
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME or "ViT-L-14", pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME or "ViT-L-14")
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

class ChangeModel(BaseModel):
    embedding_provider: str
    model_name: str

@app.get("/activeembeddingmodell")
async def activeembeddingmodell():
    logging.info("/activeembeddingmodell | incoming")
    if PROVIDER == "openai":
        result = {"embedding_provider": "openai", "model_name": OPENAI_MODEL}
    elif PROVIDER == "openclip":
        result = {"embedding_provider": "openclip", "model_name": MODEL_NAME}
    else:
        result = {"error": "Invalid EMBEDDING_PROVIDER setting"}
    logging.info(f"/activeembeddingmodell | result={json.dumps(result)}")
    return result

@app.post("/changeembeddingmodell")
async def changeembeddingmodell(payload: ChangeModel):
    logging.info(f"/changeembeddingmodell | incoming={payload.json()}")
    global PROVIDER, OPENAI_MODEL, MODEL_NAME, model, preprocess, tokenizer
    provider = payload.embedding_provider.lower()
    model_name = payload.model_name

    if provider == "openai":
        PROVIDER = "openai"
        OPENAI_MODEL = model_name
        os.environ["EMBEDDING_PROVIDER"] = "openai"
        os.environ["OPENAI_MODEL"] = model_name
        model = None
        preprocess = None
        tokenizer = None
        result = {"embedding_provider": PROVIDER, "model_name": OPENAI_MODEL}
        logging.info(f"/changeembeddingmodell | result={json.dumps(result)}")
        return result

    elif provider == "openclip":
        PROVIDER = "openclip"
        MODEL_NAME = model_name
        os.environ["EMBEDDING_PROVIDER"] = "openclip"
        os.environ["MODEL_NAME"] = model_name
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        model.eval()
        result = {"embedding_provider": PROVIDER, "model_name": MODEL_NAME}
        logging.info(f"/changeembeddingmodell | result={json.dumps(result)}")
        return result

    result = {"error": "Invalid embedding_provider"}
    logging.info(f"/changeembeddingmodell | result={json.dumps(result)}")
    return result

@app.get("/dimension")
async def dimension():
    logging.info("/dimension | incoming")
    if PROVIDER == "openclip":
        logging.info(f"dimension | Provider=openclip | dimension={json.dumps(DIMENSION_OPENCLIP)}")
        result = {"dimension": DIMENSION_OPENCLIP}
    elif PROVIDER == "openai":
        logging.info(f"dimension | Provider=openai | dimension={json.dumps(DIMENSION_OPENAI)}")
        result = {"dimension": DIMENSION_OPENAI}
    else:
        result = {"error": "Invalid DIMENSION setting"}
    logging.info(f"/dimension | result={json.dumps(result)}")
    return result


@app.post("/text-embedding")
async def text_embedding(payload: Texts):
    logging.info(f"/text-embedding | incoming={payload.json()}")
    if PROVIDER == "openclip":
        tokens = tokenizer(payload.texts)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        vectors = emb.cpu().tolist()
        logging.info(f"TextEmbedding | Provider=openclip | VectorLength={len(vectors[0])} | Vectors={json.dumps(vectors)}")
        result = {"vectors": vectors}
        logging.info(f"/text-embedding | result={json.dumps(result)}")
        return result

    elif PROVIDER == "openai":
        response = openai.embeddings.create(
            input=payload.texts,
            model=OPENAI_MODEL
        )
        vectors = [d.embedding for d in response.data]
        logging.info(f"TextEmbedding | Provider=openai | VectorLength={len(vectors[0])} | Vectors={json.dumps(vectors)}")
        result = {"vectors": vectors}
        logging.info(f"/text-embedding | result={json.dumps(result)}")
        return result

    result = {"error": "Invalid EMBEDDING_PROVIDER setting"}
    logging.info(f"/text-embedding | result={json.dumps(result)}")
    return result


@app.post("/image-embedding")
async def image_embedding(file: UploadFile = File(...)):
    logging.info(f"/image-embedding | incoming filename={file.filename}")
    data = await file.read()

    if PROVIDER == "openclip":
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img_t = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            emb = model.encode_image(img_t)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        vector = emb.cpu().tolist()[0]
        logging.info(f"ImageEmbedding | Provider=openclip | VectorLength={len(vector)} | Vector={json.dumps(vector)}")
        result = {
            "description": "test",
            "vector": vector,
            "provider": "openclip"
        }
        logging.info(f"/image-embedding | result={json.dumps(result)}")
        return result

    elif PROVIDER == "openai":
        base64_img = base64.b64encode(data).decode("utf-8")
        try:
            response = openai.chat.completions.create(
                model=OPENAI_MODEL_IMAGE,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Beschreibe ausschließlich das sichtbare physische Produkt auf dem Bild sachlich und vollständig. Gib alle sichtbaren Merkmale wie Produktform, Farbe, Kameraanordnung, Materialien, Knöpfe, Logos und sichtbare Inhalte auf dem Display an. Wenn ein Logo sichtbar ist, nenne die zugehörige Marke, sofern sie durch Form, Farbe oder Gestaltung eindeutig erkennbar ist. Verwende keine unsicheren Begriffe wie „möglicherweise“ oder „könnte“. Nutze die Markenzuordnung nur, wenn diese auf dem Bild visuell eindeutig ist, z. B. bei einem „G“-Logo für Google oder einem Apfel-Logo für Apple.Beschreibe den Bildschirminhalt nur, wenn er sichtbar ist. Verwende klare, einfache Sätze.Beende die Beschreibung mit den Feldern: Produkt-Kategorie: [z. B. Smartphone] Produkt-Name: [Marke + Modell, falls eindeutig sichtbar, sonst: „nicht erkennbar“]"},
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

            result = {
                "description": description,
                "vector": vector,
                "provider": "openai"
            }
            logging.info(f"/image-embedding | result={json.dumps(result)}")
            return result

        except Exception as e:
            result = {"error": f"OpenAI image description failed: {e}"}
            logging.info(f"/image-embedding | result={json.dumps(result)}")
            return result
    result = {"error": "Invalid EMBEDDING_PROVIDER setting"}
    logging.info(f"/image-embedding | result={json.dumps(result)}")
    return result


@app.get("/healthstatus")
async def healthstatus():
    logging.info("/healthstatus | incoming")
    result = {"status": "It works", "provider": PROVIDER}
    logging.info(f"/healthstatus | result={json.dumps(result)}")
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("clip_service:app", host="0.0.0.0", port=PORT)