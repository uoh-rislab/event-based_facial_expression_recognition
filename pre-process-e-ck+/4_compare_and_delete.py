import os
import re

dir_100fps = '../output/e-ck+_frames_process_100fps'
dir_30fps = '../output/e-ck+_frames_process_30fps'

# Recorrer todos los sujetos en la estructura del 100fps
for root, _, files_100 in os.walk(dir_100fps):
    # Construir la ruta relativa para encontrar el equivalente en 30fps
    rel_path = os.path.relpath(root, dir_100fps)
    path_30fps = os.path.join(dir_30fps, rel_path)

    if not os.path.exists(path_30fps):
        continue  # Saltar si no hay equivalente

    # Leer los nombres base válidos desde 30fps
    valid_basenames = set()
    for fname in os.listdir(path_30fps):
        match = re.match(r'frame_(\d{4})\.png', fname)
        if match:
            valid_basenames.add(match.group(1))

    # Revisar cada archivo en 100fps
    for fname in files_100:
        match = re.match(r'frame_(\d{4})\.\d\.png', fname)
        if match:
            base = match.group(1)
            if base not in valid_basenames:
                # Eliminar archivo no deseado
                full_path = os.path.join(root, fname)
                os.remove(full_path)
                print(f"Eliminado: {full_path}")
