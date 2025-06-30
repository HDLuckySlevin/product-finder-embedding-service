# product-finder-embedding-service
product-finder-embedding-service is a lightweight Python API using FastAPI and OpenCLIP (ViT-L/14) to generate unified vector embeddings for both images and product descriptions. Ideal for product matching, semantic search, and use with vector databases like Milvus — fully local and token-free.

## Endpoints

- `/activeembeddingmodell` – Returns the currently active embedding provider and model name.
- `/changeembeddingmodell` – Change the embedding provider and model at runtime by sending a JSON body like:

```json
{
  "embedding_provider": "openai",
  "model_name": "text-embedding-ada-002"
}
```

- `/availablemodels` – Returns all configured models including their provider and dimensions.

### Debugging

Set `DEBUG_VECTORS=true` in your `.env` if you want the service to log full
vectors in `debug.log`. Otherwise only vector lengths are logged.

### Model configuration

All available embedding models are provided via the `MODEL_NAME` environment variable. The format lists providers and their models with the embedding dimension, e.g.:

```
MODEL_NAME={openclip{ViT-L-14(768)},openai{text-embedding-3-large(3072),text-embedding-ada-002(1536)}}
```

In this example the OpenAI provider has two models and OpenCLIP has one.

