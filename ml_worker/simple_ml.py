import cv2
import numpy as np
import torch
import imagehash
from PIL import Image
from transformers import AutoProcessor, ProcessorMixin, AutoModel, PreTrainedModel


def extract_frames(video_path: str, threshold: int = 5) -> tuple[list, list]:
    """
    Извлекает уникальные кадры (BGR) и их таймкоды.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видеофайл {video_path}")
        return [], []

    prev_hash = None
    saved_frames = []
    saved_timestamps = []

    print("Индексация началась...")
    while True:
        success, frame = cap.read()
        if not success:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        current_hash = imagehash.phash(pil_image)

        if prev_hash is None:
            prev_hash = current_hash
            saved_frames.append(frame)
            saved_timestamps.append(timestamp)
            continue

        diff = current_hash - prev_hash

        if diff > threshold:
            saved_frames.append(frame)
            saved_timestamps.append(timestamp)
            prev_hash = current_hash

    cap.release()
    print(f"Индексация завершена. Выбрано кадров: {len(saved_frames)}")
    return saved_frames, saved_timestamps


def extract_embeddings_from_video(frames: list, model: PreTrainedModel, processor: ProcessorMixin, DEVICE: str, batch_size: int = 8) -> torch.Tensor:
    """
    Принимает список кадров (BGR), конвертирует в RGB и возвращает эмбеддинги.
    """
    if not frames:
        return torch.tensor([]).to(DEVICE)

    embeddings = []

    for i in range(0, len(frames), batch_size):
        batch_frames_bgr = frames[i : i + batch_size]

        #  BGR -> RGB перед CLIP
        batch_pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in batch_frames_bgr]

        with torch.inference_mode():
            inputs = processor(images=batch_pil, return_tensors="pt").to(DEVICE)

            image_features = model.get_image_features(**inputs)

            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(image_features)

    if not embeddings:
        return torch.tensor([]).to(DEVICE)

    return torch.cat(embeddings, dim=0)


def extract_embeddings_from_text(text: str, model: PreTrainedModel, processor: ProcessorMixin, DEVICE: str) -> torch.Tensor:
    if not text:
        return torch.tensor([]).to(DEVICE)

    with torch.inference_mode():
        inputs = processor(text=[text], return_tensors="pt", truncation=True).to(DEVICE)
        text_features = model.get_text_features(**inputs)

        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    return text_features


def get_text_frame_similarities(text_embeddings: torch.Tensor, frame_embeddings: torch.Tensor, model: PreTrainedModel = None) -> np.array:
    if frame_embeddings.nelement() == 0 or text_embeddings.nelement() == 0:
        return np.array([])

    similarities = (text_embeddings @ frame_embeddings.T).squeeze(0)

    # Для siglip
    # logits = (text_embeddings @ frame_embeddings.T).squeeze(0)
    # logits = logits * model.logit_scale.exp() + model.logit_bias
    # similarities = torch.sigmoid(logits)

    return similarities.detach().cpu().numpy().flatten()


def simple_ml(video_path: str, query: str) -> dict:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_NAME = "openai/clip-vit-large-patch14"

    print(f"Загрузка модели {MODEL_NAME} на {DEVICE}...")

    try:
        model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return {}

    # MODEL_PATH = "siglip2-so400m-patch16-384"

    # try:
    #     model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)
    #     processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    # except Exception as e:
    #     print(f"Ошибка загрузки модели: {e}")
    #     return {}

    print("\n--- Этап 1: Индексация и векторизация видео ---")
    indexed_frames, frame_timestamps = extract_frames(video_path, threshold=10)

    if not indexed_frames:
        print("Кадры не найдены.")
        return {}

    print("\n--- Этап 2: Извлечение эмбеддингов кадров видео ---")

    frame_embeddings = extract_embeddings_from_video(indexed_frames[:5], model, processor, DEVICE)

    if frame_embeddings.nelement() > 0:
        print("\n--- Этап 3: Извлечение эмбеддингов запроса ---")
        query_embeddings = extract_embeddings_from_text(query, model, processor, DEVICE)

        print("\n--- Этап 4: Вычисление сходства ---")
        scores = get_text_frame_similarities(query_embeddings, frame_embeddings, model)

        print(f"\nРезультаты для запроса: '{query}'")

        top_k = min(5, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        result_dictionary = {}

        for idx in top_indices:
            timestamp = frame_timestamps[idx]
            score = scores[idx]

            result_dictionary[round(timestamp, 2)] = round(float(score), 4)

            time_str = f"{int(timestamp // 60):02}:{int(timestamp % 60):02}"
            print(f"  - Время {time_str} ({timestamp:.2f}s): Score = {score:.4f}")

        return result_dictionary
    else:
        print("Не удалось извлечь эмбеддинги из видео.")
        return {}


if __name__ == "__main__":
    uploaded_video = "video.mp4"
    query_text = "a little monkey"

    results = simple_ml(uploaded_video, query_text)
    # print(results)
