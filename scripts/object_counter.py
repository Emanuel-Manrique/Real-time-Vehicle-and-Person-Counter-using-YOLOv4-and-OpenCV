import numpy as np


class ObjectCounter:
    def __init__(self, classes):
        self.classes = classes
        self.objects = {}
        self.counters = {class_name: 0 for class_name in classes}

    def process_detections(self, outs, frame):
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y, w, h = (detection[0:4] * np.array(
                        [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype('int')
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    match_found = False
                    for object_id, object_info in self.objects.items():
                        if abs(object_info['x'] - x) < 20 and abs(object_info['y'] - y) < 20:
                            match_found = True
                            self.objects[object_id]['x'] = x
                            self.objects[object_id]['y'] = y
                            self.objects[object_id]['w'] = w
                            self.objects[object_id]['h'] = h
                            self.objects[object_id]['frames_since_seen'] = 0
                            break

                    if not match_found:
                        self.objects[max(self.objects.keys()) + 1 if self.objects else 0] = {
                            'x': x, 'y': y, 'w': w, 'h': h, 'frames_since_seen': 0, 'class_id': class_id}

    def remove_repeated_objects(self, frame):
        object_ids_to_delete = []
        for object_id, object_info in self.objects.items():
            object_info['frames_since_seen'] += 1
            if object_info['frames_since_seen'] > 20:
                if object_info['y'] > frame.shape[0] / 2:
                    self.counters[self.classes[object_info['class_id']]] += 1
                object_ids_to_delete.append(object_id)

        for object_id in object_ids_to_delete:
            del self.objects[object_id]

    def get_objects(self):
        return self.objects

    def get_counters(self):
        return self.counters
