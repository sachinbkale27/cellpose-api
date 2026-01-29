"""
AWS Lambda handler for Cellpose API
Uses Mangum to wrap FastAPI for Lambda compatibility
"""

from mangum import Mangum
from main import app

# Create Lambda handler
handler = Mangum(app, lifespan="off")
