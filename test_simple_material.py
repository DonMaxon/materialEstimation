if __name__=='__main__':
    import os
    import torch
    from torchvision import transforms
    from PIL import Image
    from simple_material_net import SimpleMaterialNet  # Импорт вашей модели
    from material_net import predict_mode, MaterialNet
    from dino_material_net import DinoMaterialNet
    # Параметры
    MODEL_PATH = "dino_material_net10.pth"  # Путь к сохранённой модели
    FOLDER_PATH = "diff_test"  # Путь к папке с изображениями
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Предобработка изображений
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Изменение размера до 224x224, как требует CLIP
        transforms.ToTensor(),          # Преобразование в тензор
    ])

    # Загрузка модели
    model = DinoMaterialNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # Установка модели в режим оценки

    def process_image(image_path):
        """
        Обрабатывает одно изображение, возвращая предсказанные metallic и roughness.
        """
        try:
            # Загрузка изображения
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)  # Добавляем batch dimension
            # Предсказание
            with torch.no_grad():
                output = model(image_tensor)
            metallic, roughness = output[0].cpu().numpy()  # Извлечение значений
            # roughness_params, metallic_params = model(image_tensor)
            # roughness = predict_mode(roughness_params)
            # metallic = predict_mode(metallic_params)
            # return metallic.item(), roughness.item()
            return metallic, roughness
        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {e}")
            return None, None

    # Прогон по всем изображениям в папке
    def process_folder(folder_path):
        """
        Прогоняет модель по всем изображениям в заданной папке.
        """
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):  # Проверяем расширение файла
                    image_path = os.path.join(root, file_name)
                    metallic, roughness = process_image(image_path)
                    if metallic is not None and roughness is not None:
                        print(f"Обработано: {image_path} | Metallic: {metallic:.4f}, Roughness: {roughness:.4f}")
                    else:
                        print(f"Не удалось обработать: {image_path}")

    # Запуск
    process_folder(FOLDER_PATH)
