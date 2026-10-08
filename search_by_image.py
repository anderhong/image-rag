from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

# ============ 1. Load CLIP ============
print("📥 Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP loaded\n")


def embed_images(image_paths):
    """Generate an embedding from an image."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = clip_processor(images=images, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    if hasattr(outputs, "pooler_output"):
        embeddings = outputs.pooler_output
    else:
        embeddings = outputs
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings


# ============ 2. Load embeddings and metadata ============
if not os.path.exists("embeddings.pt"):
    print("❌ embeddings.pt not found. Please run build_index.py first.")
    exit(1)

data = torch.load("embeddings.pt", weights_only=False)
image_embeddings = data["embeddings"]
image_paths = data["paths"]
image_metadata = data.get("metadata", [{} for _ in image_paths])

print(f"📦 Loaded {len(image_paths)} image embeddings")
print(f"   Embedding shape: {image_embeddings.shape}\n")


# ============ 3. Query by Image ============
def search_by_image(query_image_path: str, top_k: int = 5, filters: dict = None):
    """
    Search for similar images using a query image.
    
    Args:
        query_image_path: Path to the query image.
        top_k: Number of results to return.
        filters: Metadata filters, for example {"category": "car", "brand": "BMW"}.
    """
    # Step 1: Embed Query Image
    query_embedding = embed_images([query_image_path])
    
    # Step 2: Calculate cosine similarity.
    similarities = torch.cosine_similarity(query_embedding, image_embeddings)
    
    # Step 3: Sort the results.
    sorted_indices = torch.argsort(similarities, descending=True)
    
    # Step 4: Filter and collect results.
    results = []
    for idx in sorted_indices:
        idx_int = idx.item()
        meta = image_metadata[idx_int]
        
        # Metadata Filter
        if filters:
            match = all(meta.get(k) == v for k, v in filters.items())
            if not match:
                continue
        
        results.append({
            "path": image_paths[idx_int],
            "similarity": similarities[idx_int].item(),
            "metadata": meta
        })
        
        if len(results) >= top_k:
            break
    
    # Step 5: Display the results.
    print(f"🖼️ Query Image: {query_image_path}")
    if filters:
        print(f"   Filters: {filters}")
    print(f"{'='*70}")
    
    if not results:
        print("   ❌ No results found.")
    else:
        for rank, r in enumerate(results):
            meta_str = ", ".join(f"{k}={v}" for k, v in r["metadata"].items() if k != "filename")
            print(f"   {rank+1}. {r['path']} (Similarity: {r['similarity']:.3f})")
            if meta_str:
                print(f"      📋 {meta_str}")
    print()
    
    return results


# ============ 4. Interactive Loop ============
if __name__ == "__main__":
    print("="*50)
    print("🖼️ Image → Image Search")
    print("="*50)
    print("💡 用法:")
    print("   - 輸入圖片路徑: ./input_image/cat.jpg")
    print("   - 加 Filter: filter:category=car ./input_image/bmw.jpg")
    print("   輸入 'bye' 或 'exit' 結束")
    print("="*50)
    
    while True:
        user_input = input("\n🔍 Query: ").strip()
        
        if user_input.lower() in ["bye", "exit", "quit"]:
            print("👋 Bye!")
            break
        
        if not user_input:
            continue
        
        # Parse the filter syntax.
        filters = {}
        query_parts = []
        
        for token in user_input.split():
            if token.startswith("filter:"):
                filter_str = token.replace("filter:", "")
                if "=" in filter_str:
                    key, value = filter_str.split("=", 1)
                    filters[key] = value
            else:
                query_parts.append(token)
        
        image_path = " ".join(query_parts)
        
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            continue
        
        search_by_image(image_path, filters=filters if filters else None)
