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

