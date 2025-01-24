import os
import cv2
import numpy as np

def crop_objects(input_folder, output_folder, label=1):
    # Проверяем, существует ли целевая папка, если нет — создаем
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Проходим по всем папкам train, test и val
    for subset in ['train', 'test', 'valid']:
        subset_path = os.path.join(input_folder, subset)
        if not os.path.exists(subset_path):
            print(f"Папка {subset} не найдена в {input_folder}. Пропускаем.")
            continue

        # Проходим по всем изображениям в текущей папке
        for img_name in os.listdir(os.path.join(subset_path, 'images')):
            img_path = os.path.join(subset_path, 'images', img_name)
            label_path = os.path.join(subset_path, 'labels', img_name.replace('.jpg', '.txt'))

            if not os.path.exists(label_path):
                print(f"Аннотация для {img_name} не найдена. Пропускаем.")
                continue

            # Чтение изображения
            img = cv2.imread(img_path)
            if img is None:
                print(f"Не удалось прочитать изображение {img_name}. Пропускаем.")
                continue

            # Чтение аннотаций
            with open(label_path, 'r') as file:
                annotations = file.readlines()

            # Процесс обрезки объектов
            for annotation in annotations:
                parts = annotation.strip().split()
                if int(parts[0]) == label:
                    # Преобразование координат из нормализованных в пиксельные
                    x_center, y_center, width, height = map(float, parts[1:])
                    img_height, img_width = img.shape[:2]
                    x_min = int((x_center - width / 2) * img_width)
                    y_min = int((y_center - height / 2) * img_height)
                    x_max = int((x_center + width / 2) * img_width)
                    y_max = int((y_center + height / 2) * img_height)

                    # Обрезка изображения
                    cropped_img = img[y_min:y_max, x_min:x_max]
                    if cropped_img.size == 0:
                        print(f"Обрезанное изображение для {img_name} пусто. Пропускаем.")
                        continue

                    # Сохранение обрезанного изображения
                    output_img_name = f"{img_name.replace('.jpg', '')}_label{label}.jpg"
                    output_img_path = os.path.join(output_folder, output_img_name)
                    cv2.imwrite(output_img_path, cropped_img)
                    print(f"Обрезанное изображение сохранено как {output_img_name}")



src_folder = r'D:\PythonProjects\point2cad\COCO\merge.v1i.yolov5pytorch'
dst_folder = r'D:\PythonProjects\point2cad\COCO\met'
prefix = 2

crop_objects(src_folder, dst_folder, prefix)