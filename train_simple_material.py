import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset

class SimpleMaterialDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Инициализация датасета.

        :param root_dir: Корневая папка, содержащая материалы.
        :param transform: Преобразования, применяемые к изображениям.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Словарь для меток и значений metallic и roughness
        self.materials = {
            "metal": (1.0, 0.45),
            "plastic": (0.0, 0.4),
            "cloth": (0.0, 0.6),
            "wood": (0.0, 0.7),
            "fur": (0.0, 0.8),
        }

        # Сканируем папки
        for material, (metallic, roughness) in self.materials.items():
            folder_path = os.path.join(root_dir, material)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.data.append((file_path, metallic, roughness))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, metallic, roughness = self.data[idx]
        image = Image.open(file_path).convert("RGB")  # Загружаем изображение

        # Применяем преобразования
        if self.transform:
            image = self.transform(image)

        # Возвращаем изображение и целевые значения
        target = torch.tensor([metallic, roughness], dtype=torch.float32)
        return image, target

def train_material_net(model, train_dataset, val_dataset, device, epochs=10, batch_size=32, learning_rate=1e-4, save_path="material_net.pth"):
    """
    Функция для обучения модели MaterialNet.
    
    :param model: Экземпляр модели MaterialNet.
    :param train_dataset: Датасет для обучения.
    :param val_dataset: Датасет для валидации.
    :param device: Устройство (CPU или GPU).
    :param epochs: Количество эпох.
    :param batch_size: Размер батча.
    :param learning_rate: Скорость обучения.
    :param save_path: Путь для сохранения весов модели.
    """
    # Подготовка DataLoader-ов
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Определение функции потерь и оптимизатора
    criterion = nn.MSELoss()  # Среднеквадратичная ошибка для регрессии
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Обучение
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            images, targets = batch  # images: [batch_size, 3, H, W], targets: [batch_size, 2]
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(images)

        train_loss /= len(train_loader.dataset)
        print(f"Training Loss: {train_loss:.4f}")

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images, targets = batch
                images, targets = images.to(device), targets.to(device)

                predictions = model(images)
                loss = criterion(predictions, targets)
                val_loss += loss.item() * len(images)

        val_loss /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model weights saved to {save_path}")

    print("Training complete.")

def train_beta_material_net(model, train_dataset, val_dataset, device, epochs=10, batch_size=32, learning_rate=1e-4, save_path="material_net.pth"):
    """
    Функция для обучения модели MaterialNet.
    
    :param model: Экземпляр модели MaterialNet.
    :param train_dataset: Датасет для обучения.
    :param val_dataset: Датасет для валидации.
    :param device: Устройство (CPU или GPU).
    :param epochs: Количество эпох.
    :param batch_size: Размер батча.
    :param learning_rate: Скорость обучения.
    :param save_path: Путь для сохранения весов модели.
    """
    from material_net import MaterialNet, beta_log_likelihood

    # Подготовка DataLoader-ов
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Определение функции потерь и оптимизатора
    criterion = nn.MSELoss()  # Среднеквадратичная ошибка для регрессии
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    def beta_loss(params, target):
        return -beta_log_likelihood(params, target)
    
    model = model.to(device)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        # Обучение
        model.train()
        train_loss = 0.0
        min_loss = None
        for batch in tqdm(train_loader, desc="Training"):
            images, targets = batch  # images: [batch_size, 3, H, W], targets: [batch_size, 2]
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()

            roughness_params, metallic_params = model(images)

            roughness_loss = beta_loss(roughness_params, targets[0, 0])
            metallic_loss = beta_loss(metallic_params, targets[0, 1])
            train_loss = roughness_loss + metallic_loss*0.2

            if min_loss is None or train_loss.item() < min_loss.item():
                min_loss = train_loss

        # Обратное распространение и оптимизация
        if min_loss is not None:
            optimizer.zero_grad()
            min_loss.backward()
            optimizer.step()

            epoch_loss += min_loss.item()

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images, targets = batch
                images, targets = images.to(device), targets.to(device)

                # predictions = model(images)
                # loss = criterion(predictions, targets)
                # val_loss += loss.item() * len(images)
                roughness_params, metallic_params = model(images)

                roughness_loss = beta_loss(roughness_params, targets[0, 0])
                metallic_loss = beta_loss(metallic_params, targets[0, 1])
                val_loss += roughness_loss + metallic_loss*0.2
        val_loss = val_loss.item()
        val_loss /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model weights saved to {save_path}")

    print("Training complete.")

if __name__=='__main__':
    from torch.utils.data import random_split
    # from simple_material_net import SimpleMaterialNet
    from material_net import MaterialNet
    from dino_material_net import DinoMaterialNet
    from torchvision import transforms
    # # Создаем экземпляр модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = DinoMaterialNet().to(device)
    transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Изменение размера
            transforms.ToTensor(),         # Преобразование в тензор
        ])

    # # Загрузка датасета и разбиение на обучение/валидацию
    dataset = SimpleMaterialDataset(root_dir="data2", transform=transform)  # Замените на путь к вашим данным
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Обучение модели
    train_material_net(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        epochs=15,
        batch_size=16,
        learning_rate=1e-4,
        save_path="dino_material_net15.pth"
    )

