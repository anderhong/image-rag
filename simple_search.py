from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

# ============ 1. 載入 CLIP ============
print("📥 Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP loaded\n")

def embed_images(image_paths):
    """將多張圖片轉做 Embedding"""
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


# ============ 2. Indexing Phase ============
image_folder = "./images"
image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".jpeg", ".png"))]
image_paths = [os.path.join(image_folder, f) for f in image_files]

print(f"📸 Loading {len(image_paths)} images...")
image_embeddings = embed_images(image_paths)
print(f"✅ Loaded all images\n")


# ============ 3. Query Phase ============
def search_by_text(query: str, top_k: int = 5):
    """用文字搜尋圖片"""
    text_embedding = embed_text(query)
    similarities = torch.cosine_similarity(text_embedding, image_embeddings)
    sorted_indices = torch.argsort(similarities, descending=True)[:top_k]
    
    print(f"🔍 Query: '{query}'")
    print(f"{'='*70}")
    for rank, idx in enumerate(sorted_indices):
        idx = idx.item()
        similarity = similarities[idx].item()
        print(f"   {rank+1}. {image_paths[idx]} (Similarity: {similarity:.3f})")
    print()


# ============ 4. Interactive Loop ============
if __name__ == "__main__":
    print("="*50)
    print("🐾 Simple Animal Image Search")
    print("輸入 'bye' 或 'exit' 結束")
    print("="*50)
    
    while True:
        query = input("\n🔍 Search: ").strip()
        if query.lower() in ["bye", "exit", "quit"]:
            print("👋 Bye!")
            break
        if not query:
            continue
        search_by_text(query)