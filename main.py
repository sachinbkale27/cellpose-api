"""
Cellpose API for cell segmentation in histopathology images.
Detects nuclei and cytoplasm, returns polygon coordinates for annotation.
"""

import io
import base64
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import cv2

# Cellpose import
from cellpose import models, utils

app = FastAPI(
    title="Cellpose Segmentation API",
    description="API for cell segmentation using Cellpose",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
nuclei_model = None
cyto_model = None


@app.on_event("startup")
async def load_models():
    """Load Cellpose models on startup."""
    global nuclei_model, cyto_model
    print("Loading Cellpose models...")
    nuclei_model = models.Cellpose(model_type='nuclei', gpu=False)
    cyto_model = models.Cellpose(model_type='cyto3', gpu=False)
    print("Models loaded successfully!")


class PolygonPoint(BaseModel):
    x: float
    y: float


class DetectedCell(BaseModel):
    id: str
    type: str  # 'nucleus' or 'cytoplasm'
    points: List[PolygonPoint]
    area: float
    centroid: PolygonPoint


class SegmentationResponse(BaseModel):
    success: bool
    cells: List[DetectedCell]
    total_cells: int
    image_width: int
    image_height: int
    model_used: str
    message: Optional[str] = None


def mask_to_polygons(masks: np.ndarray, min_area: int = 100) -> List[dict]:
    """
    Convert Cellpose masks to polygon coordinates.

    Args:
        masks: 2D array where each unique value represents a cell
        min_area: Minimum area threshold to filter small detections

    Returns:
        List of dictionaries containing polygon points
    """
    polygons = []
    unique_masks = np.unique(masks)

    for mask_id in unique_masks:
        if mask_id == 0:  # Skip background
            continue

        # Create binary mask for this cell
        cell_mask = (masks == mask_id).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            cell_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        # Simplify contour to reduce points (epsilon = 1% of perimeter)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Convert to list of points
        points = [{"x": float(p[0][0]), "y": float(p[0][1])} for p in approx]

        if len(points) < 3:
            continue

        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = points[0]["x"], points[0]["y"]

        polygons.append({
            "id": f"cell_{mask_id}",
            "points": points,
            "area": float(area),
            "centroid": {"x": cx, "y": cy}
        })

    return polygons


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": nuclei_model is not None and cyto_model is not None
    }


@app.post("/segment", response_model=SegmentationResponse)
async def segment_image(
    file: UploadFile = File(...),
    model_type: str = Form(default="nuclei"),
    diameter: Optional[float] = Form(default=None),
    min_area: int = Form(default=100),
    channels: str = Form(default="0,0")
):
    """
    Segment cells in an uploaded image using Cellpose.

    Args:
        file: Image file (PNG, JPG, TIFF)
        model_type: 'nuclei' or 'cyto3'
        diameter: Expected cell diameter in pixels (None for auto-detect)
        min_area: Minimum cell area to include
        channels: Cellpose channels (e.g., "0,0" for grayscale, "2,3" for G=cyto, B=nuclei)

    Returns:
        SegmentationResponse with detected cell polygons
    """
    global nuclei_model, cyto_model

    if model_type not in ["nuclei", "cyto3"]:
        raise HTTPException(status_code=400, detail="model_type must be 'nuclei' or 'cyto3'")

    model = nuclei_model if model_type == "nuclei" else cyto_model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to numpy array
        img_array = np.array(image)

        # Handle different image formats
        if len(img_array.shape) == 2:
            # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            # RGBA - remove alpha
            img_array = img_array[:, :, :3]

        # Parse channels
        try:
            ch = [int(c) for c in channels.split(",")]
            if len(ch) != 2:
                ch = [0, 0]
        except:
            ch = [0, 0]

        # Run Cellpose
        masks, flows, styles, diams = model.eval(
            img_array,
            diameter=diameter,
            channels=ch,
            flow_threshold=0.4,
            cellprob_threshold=0.0
        )

        # Convert masks to polygons
        cell_type = "nucleus" if model_type == "nuclei" else "cytoplasm"
        polygons = mask_to_polygons(masks, min_area=min_area)

        # Add type to each cell
        cells = []
        for i, poly in enumerate(polygons):
            cells.append(DetectedCell(
                id=f"{cell_type}_{i+1}",
                type=cell_type,
                points=[PolygonPoint(**p) for p in poly["points"]],
                area=poly["area"],
                centroid=PolygonPoint(**poly["centroid"])
            ))

        return SegmentationResponse(
            success=True,
            cells=cells,
            total_cells=len(cells),
            image_width=img_array.shape[1],
            image_height=img_array.shape[0],
            model_used=model_type,
            message=f"Detected {len(cells)} cells using {model_type} model"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.post("/segment-base64", response_model=SegmentationResponse)
async def segment_image_base64(
    image_data: str = Form(...),
    model_type: str = Form(default="nuclei"),
    diameter: Optional[float] = Form(default=None),
    min_area: int = Form(default=100),
    channels: str = Form(default="0,0")
):
    """
    Segment cells from a base64-encoded image.

    Args:
        image_data: Base64-encoded image (with or without data URL prefix)
        model_type: 'nuclei' or 'cyto3'
        diameter: Expected cell diameter in pixels
        min_area: Minimum cell area to include
        channels: Cellpose channels

    Returns:
        SegmentationResponse with detected cell polygons
    """
    global nuclei_model, cyto_model

    if model_type not in ["nuclei", "cyto3"]:
        raise HTTPException(status_code=400, detail="model_type must be 'nuclei' or 'cyto3'")

    model = nuclei_model if model_type == "nuclei" else cyto_model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]

        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to numpy array
        img_array = np.array(image)

        # Handle different image formats
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        # Parse channels
        try:
            ch = [int(c) for c in channels.split(",")]
            if len(ch) != 2:
                ch = [0, 0]
        except:
            ch = [0, 0]

        # Run Cellpose
        masks, flows, styles, diams = model.eval(
            img_array,
            diameter=diameter,
            channels=ch,
            flow_threshold=0.4,
            cellprob_threshold=0.0
        )

        # Convert masks to polygons
        cell_type = "nucleus" if model_type == "nuclei" else "cytoplasm"
        polygons = mask_to_polygons(masks, min_area=min_area)

        cells = []
        for i, poly in enumerate(polygons):
            cells.append(DetectedCell(
                id=f"{cell_type}_{i+1}",
                type=cell_type,
                points=[PolygonPoint(**p) for p in poly["points"]],
                area=poly["area"],
                centroid=PolygonPoint(**poly["centroid"])
            ))

        return SegmentationResponse(
            success=True,
            cells=cells,
            total_cells=len(cells),
            image_width=img_array.shape[1],
            image_height=img_array.shape[0],
            model_used=model_type,
            message=f"Detected {len(cells)} cells using {model_type} model"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
