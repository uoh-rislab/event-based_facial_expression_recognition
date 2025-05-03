import os
import numpy as np
from glob import glob

# Rutas
bbox_dir = '../../ddbb_cropped/ck+_frames_process_30fps/bboxes'           # .txt con bbox normalizados
input_root = '../../data/e-ck+_events_npy_30fps'                      # eventos completos
output_root = '../../ddbb_cropped/e-ck+_numpys_process_30fps'            # eventos recortados
os.makedirs(output_root, exist_ok=True)

# Resolución de los eventos (DVS)
img_w, img_h = 346, 260  # asegúrate de que estos valores coincidan con los eventos originales

# Recorrer Train_Set y Test_Set
for split in ['Train_Set', 'Test_Set']:
    split_path = os.path.join(input_root, split)

    for class_dir in os.listdir(split_path):
        class_path = os.path.join(split_path, class_dir)
        if not os.path.isdir(class_path):
            continue

        for subject_dir in os.listdir(class_path):
            subject_path = os.path.join(class_path, subject_dir)
            if not os.path.isdir(subject_path):
                continue

            # Leer bbox normalizado desde archivo
            bbox_txt = os.path.join(bbox_dir, f"{subject_dir}.txt")
            if not os.path.exists(bbox_txt):
                print(f"[WARN] BBox no encontrado para {subject_dir}")
                continue

            with open(bbox_txt, 'r') as f:
                line = f.readline().strip()
                x_norm, y_norm, w_norm, h_norm = map(float, line.split())

            # Convertir bbox normalizada a píxeles
            x = int(x_norm * img_w)
            y = int(y_norm * img_h)
            w = int(w_norm * img_w)
            h = int(h_norm * img_h)
            x2 = min(x + w, img_w)
            y2 = min(y + h, img_h)

            # Crear carpeta de salida
            cropped_subject_dir = os.path.join(output_root, split, class_dir, subject_dir)
            os.makedirs(cropped_subject_dir, exist_ok=True)

            # Recorrer los archivos .npy
            npy_paths = sorted(glob(os.path.join(subject_path, 'frame_*.npy')))
            for npy_path in npy_paths:
                try:
                    events = np.load(npy_path)  # shape: (N, 4)
                except Exception as e:
                    print(f"[ERROR] No se pudo cargar: {npy_path} -> {e}")
                    continue

                # Filtrar eventos dentro del bbox
                mask = (
                    (events[:, 0] >= x) & (events[:, 0] < x2) &
                    (events[:, 1] >= y) & (events[:, 1] < y2)
                )
                cropped_events = events[mask]

                # Guardar eventos recortados
                frame_name = os.path.basename(npy_path)
                output_path = os.path.join(cropped_subject_dir, frame_name)
                np.save(output_path, cropped_events)

            print(f"[OK] Eventos recortados guardados para {subject_dir} en {cropped_subject_dir}")
