import numpy as np
import cv2
import os

# Cargar los eventos
data = np.load('../data/S005_001.npy')

# Extraer valores
x = data[:, 0].astype(np.int32)
y = data[:, 1].astype(np.int32)
t = data[:, 2]
p = data[:, 3].astype(np.int8)

# Definir resolución de la cámara (DAVIS346)
width, height = 346, 260

# Parámetros de frame
frame_interval = 1 / 30  # 30 FPS
t_start = t[0]
t_end = t[-1]

# Crear carpeta de salida
os.makedirs('../output/frames_output', exist_ok=True)

frame_idx = 0
current_time = t_start

while current_time < t_end:
    # Seleccionar eventos del intervalo actual
    mask = (t >= current_time) & (t < current_time + frame_interval)
    x_frame = x[mask]
    y_frame = y[mask]
    p_frame = p[mask]

    # Crear imagen base gris
    frame = np.full((height, width), 128, dtype=np.uint8)

    # Aplicar eventos
    for xi, yi, pi in zip(x_frame, y_frame, p_frame):
        if 0 <= xi < width and 0 <= yi < height:
            if pi == 1:
                frame[yi, xi] = 255  # blanco
            elif pi == -1:
                frame[yi, xi] = 0    # negro

    # Guardar el frame
    frame_name = f'../output/frames_output/frame_{frame_idx:04d}.png'
    cv2.imwrite(frame_name, frame)

    frame_idx += 1
    current_time += frame_interval
