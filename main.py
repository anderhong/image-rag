from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import io

# ============ 1. 建立 FastAPI App ============
app = FastAPI(
    title="Image RAG API",
    description="Text -> Image & Image -> Image Search using CLIP",
    version="1.0.0"
)

# ============ 2. 載入 CLIP + Embeddings ============
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP loaded")

# 載入 Embeddings
if not os.path.exists("embeddings.pt"):
    raise RuntimeError("embeddings.pt not found. Run build_index.py first.")

data = torch.load("embeddings.pt", weights_only=False)
image_embeddings = data["embeddings"]
image_paths = data["paths"]
image_metadata = data.get("metadata", [{} for _ in image_paths])

print(f"Loaded {len(image_paths)} image embeddings")


# ============ 3. Helper Functions ============
def embed_text(text: str):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model.get_text_features(**inputs)
    if hasattr(outputs, "pooler_output"):
        embedding = outputs.pooler_output
    else:
        embedding = outputs
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding


def embed_image(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    if hasattr(outputs, "pooler_output"):
        embedding = outputs.pooler_output
    else:
        embedding = outputs
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding


def search_by_embedding(query_embedding, top_k: int = 5, filters: dict = None):
    similarities = torch.cosine_similarity(query_embedding, image_embeddings)
    sorted_indices = torch.argsort(similarities, descending=True)

    results = []
    for idx in sorted_indices:
        idx_int = idx.item()
        meta = image_metadata[idx_int]

        if filters:
            match = all(meta.get(k) == v for k, v in filters.items())
            if not match:
                continue

        results.append({
            "path": image_paths[idx_int],
            "similarity": round(similarities[idx_int].item(), 4),
            "metadata": meta
        })

        if len(results) >= top_k:
            break

    return results


# ============ 4. Request Models ============
class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict = {}


# ============ 5. API Endpoints ============
@app.get("/")
def root():
    """Health Check"""
    return {
        "status": "ok",
        "message": "Image RAG API",
        "total_images": len(image_paths)
    }


@app.post("/search/text")
def search_by_text(request: TextSearchRequest):
    """用文字搜尋圖片"""
    try:
        query_embedding = embed_text(request.query)
        results = search_by_embedding(
            query_embedding,
            top_k=request.top_k,
            filters=request.filters if request.filters else None
        )
        return {
            "query": request.query,
            "filters": request.filters,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = 5):
    """用圖片搜尋圖片"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        query_embedding = embed_image(image)
        results = search_by_embedding(query_embedding, top_k=top_k)
        return {
            "filename": file.filename,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images")
def list_images():
    """列出所有 Index 咗嘅圖"""
    return {
        "total": len(image_paths),
        "images": [
            {"path": p, "metadata": m}
            for p, m in zip(image_paths, image_metadata)
        ]
    }