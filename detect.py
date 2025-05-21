# -*- coding:utf-8 -*-

from ultralytics import YOLO

# Load a model
model = YOLO('./runs/detect/train/weights/best.pt')
# Run batched inference on a list of images
model(r"./img", imgsz=640, save=True, device=0)
