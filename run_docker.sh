#!/bin/bash

docker rm -fv ignacio_event_fer

docker run -it --name ignacio_event_fer -v /home/ignacio.bugueno/cachefs/event_fer/input:/app/input -v /home/ignacio.bugueno/cachefs/event_fer/output:/app/results ignacio_event_fer
