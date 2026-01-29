# Cellpose Segmentation API

A FastAPI-based REST API for cell segmentation in histopathology images using [Cellpose](https://github.com/MouseLand/cellpose).

## Features

- **Nuclei Detection**: Using Cellpose's `nuclei` model
- **Cytoplasm Detection**: Using Cellpose's `cyto3` model
- **Polygon Output**: Returns cell boundaries as polygon coordinates for annotation
- **Multiple Input Formats**: Accepts file upload or base64-encoded images

## Quick Start

### Using Docker (Recommended)

```bash
# Build and run
docker-compose up --build

# API will be available at http://localhost:8000
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

## API Endpoints

### Health Check
```
GET /health
```
Returns API status and model loading state.

### Segment Image (File Upload)
```
POST /segment
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- model_type: "nuclei" or "cyto3" (default: "nuclei")
- diameter: Expected cell diameter in pixels (optional, auto-detect if not provided)
- min_area: Minimum cell area threshold (default: 100)
- channels: Cellpose channels, e.g., "0,0" for grayscale (default: "0,0")
```

### Segment Image (Base64)
```
POST /segment-base64
Content-Type: multipart/form-data

Parameters:
- image_data: Base64-encoded image (required)
- model_type: "nuclei" or "cyto3" (default: "nuclei")
- diameter: Expected cell diameter (optional)
- min_area: Minimum cell area threshold (default: 100)
- channels: Cellpose channels (default: "0,0")
```

## Response Format

```json
{
  "success": true,
  "cells": [
    {
      "id": "nucleus_1",
      "type": "nucleus",
      "points": [
        {"x": 100.0, "y": 150.0},
        {"x": 110.0, "y": 160.0},
        ...
      ],
      "area": 1234.5,
      "centroid": {"x": 105.0, "y": 155.0}
    }
  ],
  "total_cells": 42,
  "image_width": 1024,
  "image_height": 768,
  "model_used": "nuclei",
  "message": "Detected 42 cells using nuclei model"
}
```

## Integration with Label-Image App

Set the environment variable in your frontend:

```bash
VITE_CELLPOSE_API_URL=http://localhost:8000
```

Or for production deployment, point to your hosted API endpoint.

## GPU Support

To enable GPU acceleration, uncomment the GPU section in `docker-compose.yml` and ensure you have:
- NVIDIA Docker runtime installed
- A CUDA-compatible GPU

## Model Information

- **nuclei**: Optimized for detecting cell nuclei
- **cyto3**: Detects cytoplasm boundaries (newer, more accurate model)

For histopathology H&E stained images:
- Use `channels="2,3"` (Green for cytoplasm, Blue for nuclei)
- For grayscale, use `channels="0,0"`

## License

MIT
