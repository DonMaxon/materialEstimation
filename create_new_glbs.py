import bpy
import os

# Данные для настройки материалов
files = {
    "car": {"metallic": 0.0226, "roughness": 0.6556},
    "flag": {"metallic": 0.3114, "roughness": 0.5623},
    "house": {"metallic": 0.0000, "roughness": 0.6721},
    "my1": {"metallic": 0.9001, "roughness": 0.5272},
    "my2": {"metallic": 0.1846, "roughness": 0.5713},
    "test1": {"metallic": 0.0000, "roughness": 0.6419},
    "test2": {"metallic": 0.0002, "roughness": 0.6074}
}

# Укажите путь к корневой папке
root_path = "/path/to/root_directory"

def process_glb_files():
    # Проход по всем подпапкам
    for dirpath, dirnames, filenames in os.walk(root_path):
        for folder_name in files.keys():
            if folder_name in dirnames:  # Проверка наличия папки
                folder_path = os.path.join(dirpath, folder_name)
                glb_file_name = f"{folder_name}.glb"
                glb_file_path = os.path.join(folder_path, glb_file_name)

                # Проверяем, существует ли файл
                if os.path.exists(glb_file_path):
                    output_file_path = os.path.join(folder_path, f"new_{glb_file_name}")

                    # Очищаем сцену Blender
                    bpy.ops.wm.read_factory_settings(use_empty=True)

                    # Импортируем GLB файл
                    bpy.ops.import_scene.gltf(filepath=glb_file_path)

                    # Настройка материалов
                    for obj in bpy.context.selected_objects:
                        if obj.type == 'MESH':
                            # Создаем новый материал
                            material = bpy.data.materials.new(name="CustomMaterial")
                            material.use_nodes = True

                            # Получаем ноды материала
                            nodes = material.node_tree.nodes
                            bsdf = nodes.get("Principled BSDF")

                            if bsdf:
                                # Устанавливаем значения metallic и roughness, оставляя цвет прежним
                                bsdf.inputs['Metallic'].default_value = files[folder_name]["metallic"]
                                bsdf.inputs['Roughness'].default_value = files[folder_name]["roughness"]

                            # Применяем материал к объекту
                            if len(obj.data.materials):
                                obj.data.materials[0] = material
                            else:
                                obj.data.materials.append(material)

                    # Экспортируем модифицированный файл
                    bpy.ops.export_scene.gltf(filepath=output_file_path)

# Запускаем процесс
process_glb_files()
