# 🚗🚶‍♂️ Real-time Vehicle and Person Counter using YOLOv4 and OpenCV 📹

This repository contains a simple implementation of a vehicle and person counter in real-time using the YOLO (You Only Look Once) object detection system. The project is organized into the following modules:

🔍 object_detector.py: Contains the ObjectDetector class, which encapsulates the logic for loading the YOLO network and processing video frames to detect objects.
📊 object_counter.py: Contains the ObjectCounter class, which keeps track of the detected objects across frames and maintains counters for the number of detected vehicles and people.
📝 main.py: This is the entry point of the application. It sets up the video feed, initializes the ObjectDetector and ObjectCounter classes, and runs the main loop that reads frames, detects and counts objects, and displays the results.

The current implementation provides basic counting functionality and may produce duplicate counts if the same object is detected in multiple frames. More advanced tracking methods such as the Kalman filter can be used to maintain a consistent identity for each detected object across frames, thereby reducing duplicate counts and improving the accuracy of the counter.

This is a basic implementation and serves as a starting point for more complex object counting and tracking applications. Contributions to improve the accuracy and functionality are welcome. 👍

💻 How to run the project:

1) Install the required Python libraries with pip install opencv-python numpy.
2) Download the YOLO weights and place it in models folder.
3) Run python main.py.
4) The application will start a video feed and display the frames with bounding boxes around detected vehicles and people, along with counters for the number of detected objects. The feed can be closed by pressing the 'Esc' key. 🎥

Feel free to contribute and enhance the project! 🌟
