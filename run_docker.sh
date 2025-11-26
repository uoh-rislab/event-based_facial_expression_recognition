#!/bin/bash

docker rm -fv ignacio_event_fer

docker run -it \
  --gpus '"device=0"' \
  --name ignacio_event_fer \
  --shm-size=8g \
  -v /home/ignacio.bugueno/cachefs/event_fer/input:/app/input \
  -v /home/ignacio.bugueno/cachefs/event_fer/output:/app/results \
  ignacio_event_fer
