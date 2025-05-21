# -*- coding:utf-8 -*-

from ultralytics import YOLO

if __name__ == '__main__':
    # build from YAML and transfer weights
    model = YOLO('yolov8n.yaml').load('./weights/yolov8n.pt')
    # Train the model
    model.train(data='./VOCData/mydata.yaml', epochs=100, imgsz=640)
