import numpy as np
import os
from glob import glob

# Parámetros de la cámara
width, height = 346, 260
fps = 30
frame_interval = 1 / fps  # segundos por frame

# Directorios de entrada y salida
input_root = '../data/e-ck+_346_full_ms'
output_root = f'../output/e-ck+_events_npy_{fps}fps'

# Recorrer Train_Set y Test_Set
for split in ['Train_Set', 'Test_Set']:
    split_path = os.path.join(input_root, split)

    # Subdirectorios 1-7
    for class_dir in os.listdir(split_path):
        class_path = os.path.join(split_path, class_dir)
        if not os.path.isdir(class_path):
            continue

        # Archivos .npy
        for npy_file in glob(os.path.join(class_path, '*.npy')):
            # Cargar eventos
            data = np.load(npy_file)

            x = data[:, 0].astype(np.int32)
            y = data[:, 1].astype(np.int32)
            t = data[:, 2]
            p = data[:, 3].astype(np.int8)

            # Frame metadata
            t_start = t[0]
            t_end = t[-1]
            current_time = t_start
            frame_idx = 0

            # Generar ruta de salida
            relative_path = os.path.relpath(npy_file, input_root)
            base_name = os.path.splitext(os.path.basename(npy_file))[0]
            output_dir = os.path.join(output_root, os.path.dirname(relative_path), base_name)
            os.makedirs(output_dir, exist_ok=True)

            # Guardar eventos por intervalo
            while current_time < t_end:
                mask = (t >= current_time) & (t < current_time + frame_interval)
                events = np.stack((x[mask], y[mask], t[mask], p[mask]), axis=-1)

                if fps <= 30:
                    frame_label = int(round((current_time - t_start) * 30))
                    frame_name = f'frame_{frame_label:04d}.npy'
                else:
                    base_frame_duration = 1 / 30
                    relative_time_from_start = current_time - t_start
                    base_frame = int(relative_time_from_start // base_frame_duration)

                    time_in_base = relative_time_from_start - base_frame * base_frame_duration
                    sub_index = int(round(time_in_base / frame_interval))

                    frame_name = f'frame_{base_frame:04d}.{sub_index}.npy'

                frame_path = os.path.join(output_dir, frame_name)
                np.save(frame_path, events)

                frame_idx += 1
                current_time += frame_interval
