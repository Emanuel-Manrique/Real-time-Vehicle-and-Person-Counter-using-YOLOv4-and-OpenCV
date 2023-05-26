import cv2
import numpy as np


class ObjectDetector:
    def __init__(self, model_weights, model_cfg, class_file):
        with open(class_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.net = cv2.dnn.readNet(model_weights, model_cfg)

        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

    def detect_objects(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        outs = self.net.forward(self.output_layers)

        return outs
