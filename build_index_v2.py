from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import json

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
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings


# ============ 2. Load metadata ============
metadata_file = "metadata.json"
if os.path.exists(metadata_file):
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata_map = json.load(f)
    print(f"📋 Loaded metadata for {len(metadata_map)} images")
else:
    metadata_map = {}
    print("⚠️ No metadata.json found. Using empty metadata.")


# ============ 3. Indexing Phase ============
image_folder = "./images"
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".jpg", ".jpeg", ".png"))])
image_paths = [os.path.join(image_folder, f) for f in image_files]

print(f"📸 Loading {len(image_paths)} images...")
image_embeddings = embed_images(image_paths)
print(f"✅ Loaded all images\n")

# Prepare metadata for each image.
image_metadata = []
for path in image_paths:
    filename = os.path.basename(path)
    meta = metadata_map.get(filename, {})
    meta["filename"] = filename
    image_metadata.append(meta)


# ============ 4. Save embeddings and metadata ============
data = {
    "embeddings": image_embeddings,
    "paths": image_paths,
    "metadata": image_metadata,
}

torch.save(data, "embeddings.pt")
print(f"💾 Saved embeddings to embeddings.pt")
print(f"   Total images: {len(image_paths)}")
print(f"   Embedding shape: {image_embeddings.shape}")
print(f"   Metadata entries: {len(image_metadata)}")
