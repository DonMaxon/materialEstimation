import os
from PIL import Image

def find_problematic_images(folder_path):
    """
    Проходит по всем файлам в указанной папке и проверяет, могут ли они быть открыты как изображения.
    :param folder_path: Путь к папке для проверки
    """
    problematic_files = []
    
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Проверяем валидность изображения
            except (Image.UnidentifiedImageError, OSError) as e:
                print(f"Ошибка для файла: {file_path} -> {e}")
                problematic_files.append(file_path)
    
    if problematic_files:
        print("\nПроблемные файлы:")
        for file in problematic_files:
            print(file)
    else:
        print("Все изображения валидны!")

# Укажите путь к папке с изображениями
folder_path = "data2/wood"
find_problematic_images(folder_path)
