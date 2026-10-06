from transformers import CLIPProcessor, CLIPModel
import torch
import os
import json

# ============ 1. 載入 CLIP ============
print("📥 Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP loaded\n")


def embed_text(text):
    """將文字轉做 Embedding"""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model.get_text_features(**inputs)
    if hasattr(outputs, "pooler_output"):
        embedding = outputs.pooler_output
    else:
        embedding = outputs
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding


# ============ 2. 載入 Embedding + Metadata ============
if not os.path.exists("embeddings.pt"):
    print("❌ embeddings.pt not found. Please run build_index.py first.")
    exit(1)

data = torch.load("embeddings.pt", weights_only=False)
image_embeddings = data["embeddings"]
image_paths = data["paths"]
image_metadata = data.get("metadata", [{} for _ in image_paths])

print(f"📦 Loaded {len(image_paths)} image embeddings")
print(f"   Embedding shape: {image_embeddings.shape}\n")


# ============ 3. Query Phase ============
def search(query: str, top_k: int = 5, filters: dict = None):
    """
    搜尋圖片
    
    Args:
        query: Text Query
        top_k: 回傳數量
        filters: Metadata Filter，例如 {"category": "car", "color": "red"}
    """
    text_embedding = embed_text(query)
    similarities = torch.cosine_similarity(text_embedding, image_embeddings)
    
    # 排序
    sorted_indices = torch.argsort(similarities, descending=True)
    
    # Filter + 收集結果
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
    
    # 顯示結果
    print(f"🔍 Query: '{query}'")
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
    print("🐾 Multi-modal Image Search")
    print("="*50)
    print("💡 Query 範例:")
    print("   - a cat")
    print("   - a red sports car")
    print("   - filter:category=car a black car")
    print("   - filter:category=animal color=white a bird")
    print("   輸入 'bye' 或 'exit' 結束")
    print("="*50)
    
    while True:
        user_input = input("\n🔍 Search: ").strip()
        
        if user_input.lower() in ["bye", "exit", "quit"]:
            print("👋 Bye!")
            break
        
        if not user_input:
            continue
        
        # 解析 Filter 語法：filter:key=value
        filters = {}
        query_parts = []
        
        for token in user_input.split():
            if token.startswith("filter:"):
                # 解析 filter:category=car
                filter_str = token.replace("filter:", "")
                if "=" in filter_str:
                    key, value = filter_str.split("=", 1)
                    filters[key] = value
            else:
                query_parts.append(token)
        
        query = " ".join(query_parts)
        
        if not query:
            print("⚠️ Please provide a search query.")
            continue
        
        search(query, filters=filters if filters else None)