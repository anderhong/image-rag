# uv run python build_index.py
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
    """Generate embeddings for multiple images."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = clip_processor(images=images, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    if hasattr(outputs, "pooler_output"):
        embeddings = outputs.pooler_output
    else:
        embeddings = outputs
    # Normalize
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings


# ============ 2. Indexing Phase ============
image_folder = "./images"
image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".jpeg", ".png"))]
image_paths = [os.path.join(image_folder, f) for f in image_files]

print(f"📸 Loading {len(image_paths)} images...")
image_embeddings = embed_images(image_paths)
print(f"✅ Loaded all images\n")


# ============ 3. Save embeddings and metadata ============
# Save the embedding tensor and image paths.
data = {
    "embeddings": image_embeddings,     # Tensor (N, 512)
    "paths": image_paths,                # List of str
}

torch.save(data, "embeddings.pt")
print(f"💾 Saved embeddings to embeddings.pt")
print(f"   Total images: {len(image_paths)}")
print(f"   Embedding shape: {image_embeddings.shape}")

# Also save a human-readable summary for debugging.
with open("embeddings_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Total images: {len(image_paths)}\n")
    f.write(f"Embedding shape: {image_embeddings.shape}\n\n")
    f.write("Image paths:\n")
    for i, path in enumerate(image_paths):
        f.write(f"  {i}. {path}\n")

print(f"📄 Also saved human-readable summary: embeddings_summary.txt")
