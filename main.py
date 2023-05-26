from scripts.object_detector import ObjectDetector
from scripts.object_counter import ObjectCounter
import cv2


def main():
    # I am using yolov4-tiny for better performance, you can use any model that you want changing
    detector = ObjectDetector('models/yolov4-tiny.weights', 'models/yolov4.cfg', 'models/coco.names')
    counter = ObjectCounter(detector.classes)

    # Change the URL with your rtsp url. Documentation: https://dahuawiki.com/Remote_Access/RTSP_via_VLC
    cap = cv2.VideoCapture(
        'rtsp://admin:admin@10.7.6.67:554/cam/realmonitor?channel=1&subtype=1')

    frame_id = 0
    while True:
        ret, frame = cap.read()
        frame_id += 1

        if not ret or frame_id % 3 != 0:  # If you want to improve the precision change the number 4, smaller numbers
            # equals more fps.
            continue

        outs = detector.detect_objects(frame)

        counter.process_detections(outs, frame)

        objects = counter.get_objects()

        for object_info in objects.values():
            x = object_info['x']
            y = object_info['y']
            w = object_info['w']
            h = object_info['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        counter.remove_repeated_objects(frame)

        print(counter.get_counters())

        cv2.imshow('Image', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
