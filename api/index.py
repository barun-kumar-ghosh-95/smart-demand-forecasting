"""
Vercel Serverless Function Entry Point for FastAPI.
Exposes the FastAPI 'app' instance for Vercel's Python runtime.
"""
import os
import sys

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.main import app

# Vercel looks for 'app' as the ASGI application entry point
