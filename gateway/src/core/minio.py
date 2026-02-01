"""s3"""

from minio.api import Minio

from src.core.config import settings

client = Minio(
    endpoint=settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=False,
    region="us-east-1" # fucking shit minio
)

public_client = Minio(
    endpoint=settings.MINIO_PUBLIC_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=False,
    region="us-east-1" # fucking shit minio
)