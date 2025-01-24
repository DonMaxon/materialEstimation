import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def calculate_psnr(image1, image2):
    """Вычисляет PSNR между двумя изображениями."""
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')  # Идентичные изображения
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(image1, image2):
    """Вычисляет SSIM между двумя изображениями."""
    min_side = min(image1.shape[:2])
    
    # Устанавливаем значение win_size
    win_size = min(7, min_side)  # Используем 7 или меньшее значение
    ssim_value, _ = ssim(image1, image2, full=True, multichannel=True, win_size=win_size, channel_axis=-1)
    return ssim_value

def calculate_clip_similarity(image1_path, image2_path):
    """Вычисляет CLIP similarity между двумя изображениями."""
    # Загружаем модель и процессор CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Загружаем изображения
    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")
    
    # Преобразуем изображения
    inputs1 = processor(images=image1, return_tensors="pt", padding=True)
    inputs2 = processor(images=image2, return_tensors="pt", padding=True)
    
    # Извлекаем эмбеддинги
    with torch.no_grad():
        image_features1 = model.get_image_features(**inputs1)
        image_features2 = model.get_image_features(**inputs2)
    
    # Нормализуем эмбеддинги
    image_features1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)
    image_features2 = image_features2 / image_features2.norm(dim=-1, keepdim=True)
    
    # Вычисляем косинусное сходство
    similarity = (image_features1 @ image_features2.T).item()
    return similarity