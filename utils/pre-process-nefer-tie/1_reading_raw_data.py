#!/usr/bin/env python3.8
# 1_reading_raw_data.py
# Lectura básica de un .raw de Prophesee (NEFER) usando Metavision
# + resolución espacial
# + número de ventanas de 15 ms

from metavision_core.event_io import EventsIterator
import numpy as np
import os

# TODO: ajusta esta ruta a un .raw real de NEFER
RAW_PATH = "data/user00_2022-06-08_11-52-05.raw"

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(RAW_PATH)

# ================================
# Parámetro: tamaño ventana en ms
# ================================
WINDOW_MS = 15
WINDOW_US = WINDOW_MS * 1000

# Leemos por bloques de 200k eventos
it = EventsIterator(RAW_PATH, mode="n_events", n_events=200_000)

total = 0
first_ts = None
last_ts = None

# Para obtener resolución espacial
min_x, max_x = 999999, -1
min_y, max_y = 999999, -1

for i, events in enumerate(it):
    if events.size == 0:
        continue

    total += events.size
    xs = events["x"]
    ys = events["y"]
    ts = events["t"]

    # Actualizamos resolución
    min_x = min(min_x, int(xs.min()))
    max_x = max(max_x, int(xs.max()))
    min_y = min(min_y, int(ys.min()))
    max_y = max(max_y, int(ys.max()))

    # Actualizamos timestamps globales
    if first_ts is None:
        first_ts = int(ts.min())
    last_ts = int(ts.max())

    # Print opcional por bloques
    print(f"Bloque {i:04d}: {events.size:7d} eventos | "
          f"t: {int(ts.min())} → {int(ts.max())} | "
          f"x:[{xs.min()}, {xs.max()}], y:[{ys.min()}, {ys.max()}]")

# ================================
#     CÁLCULOS FINALES
# ================================

duration_us = last_ts - first_ts
duration_ms = duration_us / 1000.0
duration_s  = duration_us / 1e6

# Número de ventanas de 15 ms
num_windows = duration_us // WINDOW_US

# Resolución espacial
width  = max_x - min_x + 1
height = max_y - min_y + 1

# ================================
#        IMPRESIÓN FINAL
# ================================

print("\n========================")
print("        RESUMEN")
print("========================")
print(f"Total eventos:           {total}")
print(f"Rango temporal (µs):     {first_ts} → {last_ts}")
print(f"Duración total:          {duration_us} µs")
print(f"                         {duration_ms:.3f} ms")
print(f"                         {duration_s:.3f} s")
print("")
print(f"Resolución espacial:     {width} × {height}")
print(f"(x: {min_x}–{max_x}, y: {min_y}–{max_y})")
print("")
print(f"Ventanas de {WINDOW_MS} ms: {num_windows} ventanas")
print("========================\n")
