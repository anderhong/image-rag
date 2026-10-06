# Image RAG API

Text → Image & Image → Image Search using CLIP + FastAPI.

## Features

- 🖼️ Text → Image Search (CLIP)
- 🔍 Image → Image Search (CLIP)
- 🏷️ Metadata Filter
- 🚀 FastAPI + Swagger UI

## Tech Stack

- **CLIP** (OpenAI) - Multi-modal Embedding
- **FastAPI** - Web Framework
- **Uvicorn** - ASGI Server
- **PyTorch** - Deep Learning

## Installation

\`\`\`bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/image-rag.git
cd image-rag

# Install dependencies
uv sync

# Build index (first time only)
uv run python build_index.py

# Run API
uv run uvicorn main:app --reload --port 8000
\`\`\`

## Usage

Open http://localhost:8000/docs to test the API.

### Text Search

\`\`\`bash
curl -X POST http://localhost:8000/search/text \\
  -H "Content-Type: application/json" \\
  -d '{"query": "a red sports car", "top_k": 3}'
\`\`\`

### Image Search

\`\`\`bash
curl -X POST http://localhost:8000/search/image \\
  -F "file=@./input_image/query.jpg"
\`\`\`

## Project Structure

\`\`\`
image-rag/
├── images/              # Image database
├── input_image/         # Query images
├── metadata.json        # Image metadata
├── build_index.py       # Build embedding index
├── search.py            # CLI search
├── main.py              # FastAPI app
└── README.md
\`\`\`

## License

MIT