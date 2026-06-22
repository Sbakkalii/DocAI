"""
Celery worker for async multi-page document processing (Track B).
Start with: celery -A workers.celery_app worker --loglevel=info
"""

from workers.celery_app import app as celery_app

__all__ = ["celery_app"]
