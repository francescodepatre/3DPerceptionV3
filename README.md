<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Documentation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f9;
        }
        h1 {
            color: #2c3e50;
        }
        h2 {
            color: #34495e;
        }
        pre {
            background-color: #2d3436;
            color: white;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .section {
            margin-bottom: 20px;
        }
        .code-block {
            background-color: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
    </style>
</head>
<body>

    <h1>Project Documentation</h1>

    <div class="section">
        <h2>online_detection.py</h2>
        <h3>Overview</h3>
        <p>This Python script is designed to perform real-time detection of human faces using a YOLO model. It captures live video from a webcam and detects faces in the video stream, displaying bounding boxes and confidence scores for each detection.</p>
        
        <h3>Features</h3>
        <ul>
            <li>Real-time face detection using YOLOv5 model</li>
            <li>Draws bounding boxes around detected faces in the video stream</li>
            <li>Displays confidence scores for each detected face</li>
            <li>Provides an option to save the processed video with annotations</li>
        </ul>

        <h3>Requirements</h3>
        <pre>pip install opencv-python ultralytics</pre>

        <h3>Input</h3>
        <ul>
            <li>Live video stream (webcam feed)</li>
        </ul>

        <h3>Output</h3>
        <ul>
            <li>Processed video file with bounding boxes and confidence scores</li>
        </ul>

        <h3>Usage</h3>
        <pre>
python online_detection.py
        </pre>
    </div>

    <div class="section">
        <h2>offline_detection.py</h2>
        <h3>Overview</h3>
        <p>This Python script processes a pre-recorded video file to detect faces using the YOLO model. It operates similarly to the real-time detection script but works with an offline video source instead of a live feed.</p>
        
        <h3>Features</h3>
        <ul>
            <li>Offline face detection using YOLOv5 model</li>
            <li>Processes video frames and draws bounding boxes around detected faces</li>
            <li>Displays confidence scores for each detection</li>
            <li>Saves the processed video with annotations</li>
        </ul>

        <h3>Requirements</h3>
        <pre>pip install opencv-python ultralytics</pre>

        <h3>Input</h3>
        <ul>
            <li>Pre-recorded video file (e.g., `input_video.mp4`)</li>
        </ul>

        <h3>Output</h3>
        <ul>
            <li>Processed video file with bounding boxes and confidence scores</li>
        </ul>

        <h3>Usage</h3>
        <pre>
python offline_detection.py
        </pre>
    </div>

    <div class="section">
        <h2>post_processing.py</h2>
        <h3>Overview</h3>
        <p>This Python script processes RGB and depth data captured from Kinect sensors and uses a YOLOv11 model to detect human faces in the RGB frames. It calculates the real-world distance of detected faces using depth information, and then combines this with GPS coordinates to create a map of face locations. Additionally, the script logs the distance data and visualizes the results in both a video and an interactive map.</p>

        <h3>Features</h3>
        <ul>
            <li>Face Detection: Uses YOLOv11 for detecting human faces in RGB video frames</li>
            <li>Depth Calculation: Uses depth maps to calculate the real-world distance of detected faces</li>
            <li>Multimodal Prediction: Combines RGB images and numerical data (distance, GPS, and orientation) to make multimodal predictions using a custom LSTM model</li>
            <li>Map Generation: Generates an interactive map showing the locations of detected faces relative to the Kinect sensor, using the GPS coordinates and calculated distance</li>
            <li>Video Output: Saves the processed RGB video with bounding boxes and distance annotations</li>
            <li>CSV Logging: Saves the real and predicted distances of detected faces in a CSV file</li>
        </ul>

        <h3>Requirements</h3>
        <pre>pip install opencv-python folium pyproj shapely ultralytics torch torchvision pandas requests</pre>

        <h3>Input</h3>
        <ul>
            <li>RGB Video: A video file (`freesbee_rgb.mp4`) containing the RGB frames from the Kinect camera</li>
            <li>Depth Video: A video file (`freesbee_depth.mp4`) containing the depth maps from the Kinect camera</li>
            <li>GPS Location File: A text file (`GPS_location_data.txt`) containing the GPS coordinates (latitude and longitude) of the Kinect sensor</li>
        </ul>

        <h3>Output</h3>
        <ul>
            <li>CSV File: The real and predicted distances of detected faces saved in `distances.csv`</li>
            <li>Video File: A processed video file with annotated bounding boxes and predicted distances saved as `output_video_<timestamp>.mp4`</li>
            <li>Map File: An HTML file (`kinect_map<timestamp>.html`) displaying the detected faces on an interactive map with the Kinect sensor as the origin</li>
        </ul>

        <h3>Usage</h3>
        <pre>
python post_processing.py
        </pre>
    </div>

</body>
</html>
