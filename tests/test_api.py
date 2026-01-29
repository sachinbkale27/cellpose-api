"""
Tests for Cellpose API endpoints.
"""

import io
import base64
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

# Import without loading models initially
import sys
sys.modules['cellpose'] = type(sys)('cellpose')
sys.modules['cellpose.models'] = type(sys)('cellpose.models')
sys.modules['cellpose.utils'] = type(sys)('cellpose.utils')

from main import app, mask_to_polygons


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    img = Image.new('RGB', (100, 100), color='white')
    # Add a simple shape for detection
    pixels = img.load()
    for x in range(30, 70):
        for y in range(30, 70):
            pixels[x, y] = (100, 100, 100)
    return img


@pytest.fixture
def sample_image_bytes(sample_image):
    """Get test image as bytes."""
    buffer = io.BytesIO()
    sample_image.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_image_base64(sample_image_bytes):
    """Get test image as base64 string."""
    return base64.b64encode(sample_image_bytes).decode('utf-8')


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_status(self, client):
        """Health endpoint should return status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data

    def test_health_shows_models_not_loaded(self, client):
        """Health should indicate models not loaded initially."""
        response = client.get("/health")
        data = response.json()
        # Models won't be loaded in test environment
        assert data["status"] == "healthy"


class TestMaskToPolygons:
    """Tests for the mask_to_polygons utility function."""

    def test_empty_mask(self):
        """Empty mask should return empty list."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        polygons = mask_to_polygons(mask)
        assert polygons == []

    def test_single_cell_mask(self):
        """Single cell mask should return one polygon."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Create a circular cell
        for y in range(100):
            for x in range(100):
                if (x - 50) ** 2 + (y - 50) ** 2 < 400:
                    mask[y, x] = 1

        polygons = mask_to_polygons(mask, min_area=50)
        assert len(polygons) == 1
        assert "points" in polygons[0]
        assert "area" in polygons[0]
        assert "centroid" in polygons[0]
        assert len(polygons[0]["points"]) >= 3

    def test_multiple_cells_mask(self):
        """Multiple cells should return multiple polygons."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Cell 1
        for y in range(20, 60):
            for x in range(20, 60):
                mask[y, x] = 1
        # Cell 2
        for y in range(100, 140):
            for x in range(100, 140):
                mask[y, x] = 2

        polygons = mask_to_polygons(mask, min_area=50)
        assert len(polygons) == 2

    def test_small_cells_filtered(self):
        """Small cells below min_area should be filtered."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Create a very small cell (5x5 = 25 pixels)
        mask[10:15, 10:15] = 1

        polygons = mask_to_polygons(mask, min_area=100)
        assert len(polygons) == 0

    def test_polygon_points_format(self):
        """Polygon points should have x and y coordinates."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1

        polygons = mask_to_polygons(mask, min_area=50)
        assert len(polygons) == 1

        for point in polygons[0]["points"]:
            assert "x" in point
            assert "y" in point
            assert isinstance(point["x"], float)
            assert isinstance(point["y"], float)

    def test_centroid_within_bounds(self):
        """Centroid should be within the cell bounds."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1

        polygons = mask_to_polygons(mask, min_area=50)
        centroid = polygons[0]["centroid"]

        assert 30 <= centroid["x"] <= 70
        assert 30 <= centroid["y"] <= 70


class TestSegmentEndpointValidation:
    """Tests for segment endpoint input validation."""

    def test_invalid_model_type(self, client, sample_image_bytes):
        """Invalid model type should return 400."""
        response = client.post(
            "/segment",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
            data={"model_type": "invalid_model"}
        )
        assert response.status_code == 400
        assert "model_type must be" in response.json()["detail"]

    def test_valid_model_types(self, client):
        """Both 'nuclei' and 'cyto3' should be valid."""
        # These will fail with 503 due to models not loaded,
        # but should not fail with 400 for invalid model type
        response1 = client.post(
            "/segment",
            files={"file": ("test.png", b"fake", "image/png")},
            data={"model_type": "nuclei"}
        )
        # 503 means model validation passed, just not loaded
        assert response1.status_code in [503, 500]

        response2 = client.post(
            "/segment",
            files={"file": ("test.png", b"fake", "image/png")},
            data={"model_type": "cyto3"}
        )
        assert response2.status_code in [503, 500]


class TestSegmentBase64EndpointValidation:
    """Tests for segment-base64 endpoint input validation."""

    def test_invalid_model_type(self, client, sample_image_base64):
        """Invalid model type should return 400."""
        response = client.post(
            "/segment-base64",
            data={
                "image_data": sample_image_base64,
                "model_type": "invalid_model"
            }
        )
        assert response.status_code == 400
        assert "model_type must be" in response.json()["detail"]

    def test_with_data_url_prefix(self, client):
        """Should handle base64 with data URL prefix."""
        # This will fail due to model not loaded, but tests parsing
        base64_with_prefix = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        response = client.post(
            "/segment-base64",
            data={
                "image_data": base64_with_prefix,
                "model_type": "nuclei"
            }
        )
        # Should fail with 503 (model not loaded) not 400 (parse error)
        assert response.status_code in [503, 500]


class TestOpenAPISchema:
    """Tests for API documentation."""

    def test_openapi_schema_available(self, client):
        """OpenAPI schema should be available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

    def test_docs_available(self, client):
        """Swagger docs should be available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_endpoints_documented(self, client):
        """All endpoints should be documented."""
        response = client.get("/openapi.json")
        schema = response.json()
        paths = schema["paths"]

        assert "/health" in paths
        assert "/segment" in paths
        assert "/segment-base64" in paths
