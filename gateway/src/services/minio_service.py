from minio.api import Minio

from fastapi import UploadFile

from src.core.minio import client

class MinioService:
    def __init__(self, client: Minio):
        self.client = client
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


minio_service = MinioService(client=client)