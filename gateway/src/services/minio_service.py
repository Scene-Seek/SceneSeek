from datetime import timedelta

from minio.api import Minio

from fastapi import UploadFile

from src.core.minio import client, public_client

class MinioService:
    def __init__(self, *, client: Minio, public_client: Minio):
        self.client = client
        self.public_client = public_client
        self.BUCKET_VIDEOS_IN="videos-in-bucket"
        self.BUCKETS = [self.BUCKET_VIDEOS_IN]

    def create_buckets(self):
        """
        Создает бакеты, если их нет
        """
        for bucket in self.BUCKETS:
            isExisting = self.client.bucket_exists(bucket)
            if not isExisting:
                self.client.make_bucket(bucket_name=bucket)
                print("Created bucket", isExisting)
            else:
                print("Bucket", isExisting, "already exists")

    def save_obj(self, obj: UploadFile, bucket: str):
        """
        Кладёт в bucket объект
        """
        self.client.put_object(
            bucket_name=bucket,
            object_name=obj.filename,
            data=obj.file,
            length=obj.size,
            content_type=obj.content_type
        )

    def get_presigned_url_internal(
        self,
        *,
        bucket: str,
        object_name: str,
        expires_seconds: int = 3600
    ) -> str:
        """
        Возвращает временную ссылку для скачивания/просмотра объекта.
        """
        return self.client.presigned_get_object(
            bucket_name=bucket,
            object_name=object_name,
            expires=timedelta(seconds=expires_seconds)
        )

    def get_presigned_url(
        self,
        *,
        bucket: str,
        object_name: str,
        expires_seconds: int = 3600
    ) -> str:
        """
        Возвращает временную ссылку для скачивания/просмотра объекта.
        """
        return self.public_client.presigned_get_object(
            bucket_name=bucket,
            object_name=object_name,
            expires=timedelta(seconds=expires_seconds)
        )

    def get_video_url(self, *, object_name: str, expires_seconds: int = 3600) -> str:
        """
        Удобный метод для видео в default bucket.
        """
        return self.get_presigned_url(
            bucket=self.BUCKET_VIDEOS_IN,
            object_name=object_name,
            expires_seconds=expires_seconds
        )
    def get_video_url_internal(self, *, object_name: str, expires_seconds: int = 3600) -> str:
        """
        Удобный метод для видео в default bucket.
        """
        return self.get_presigned_url_internal(
            bucket=self.BUCKET_VIDEOS_IN,
            object_name=object_name,
            expires_seconds=expires_seconds
        )


minio_service = MinioService(client=client, public_client=public_client)