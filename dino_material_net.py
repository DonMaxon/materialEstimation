import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class DinoMaterialNet(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", dino_model_name="facebook/dino-vits16", 
                 clip_embedding_dim=512, dino_embedding_dim=384, hidden_dim=512):
        """
        Модель для предсказания значений metallic и roughness с использованием эмбеддингов CLIP и DINO.

        :param clip_model_name: Название предобученной модели CLIP.
        :param dino_model_name: Название предобученной модели DINO.
        :param clip_embedding_dim: Размерность эмбеддингов CLIP.
        :param dino_embedding_dim: Размерность эмбеддингов DINO.
        :param hidden_dim: Размерность скрытых слоев в MLP.
        """
        super(DinoMaterialNet, self).__init__()
        # Загрузка предобученной модели CLIP
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Загрузка предобученной модели DINO
        self.dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        for param in self.dino_model.parameters():
            param.requires_grad = False

        # Размер объединенных эмбеддингов
        combined_embedding_dim = clip_embedding_dim + dino_embedding_dim

        # MLP для предсказания metallic и roughness
        self.mlp = nn.Sequential(
            nn.Linear(combined_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 2 выхода: metallic и roughness
            nn.Sigmoid()
        )

    def forward(self, image):
        """
        Подает изображение через CLIP и DINO, объединяет эмбеддинги и передает через MLP.
        
        :param image: Входное изображение.
        :return: Предсказанные значения metallic и roughness.
        """
        # Получение эмбеддингов CLIP
        clip_embedding = self.clip_model.get_image_features(image)

        # Получение эмбеддингов DINO
        dino_embedding = self.dino_model(image)

        # Объединение эмбеддингов
        combined_embedding = torch.cat((clip_embedding, dino_embedding), dim=-1)

        # Пропуск через MLP
        output = self.mlp(combined_embedding)
        return output
