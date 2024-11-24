import cv2
import numpy as np
import folium
from ultralytics import YOLO
import requests
import webbrowser
import datetime
from shapely.geometry import Point
from pyproj import Geod
import math
import MultiModalModel
import torch
import torchvision.transforms as transforms
from kinect_sensor import Kinect  

date = datetime.datetime.now()
current_timestamp = date.timestamp()

kinect = Kinect() 

model_yolo = YOLO("last.pt")

# Carica le classi
with open('obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

next_id = 0

def get_location(file_path):
    try:
        with open(file_path, 'r') as file:
            latitude = float(file.readline().strip())
            longitude = float(file.readline().strip())
        return latitude, longitude
    except (IOError, ValueError) as e:
        print(f"Errore nella lettura del file o nel parsing delle coordinate: {e}")
        return None, None
    
def compass_face(file_path):
    try:
        with open(file_path, 'r') as file:
            comp = float(file.readline().strip())
            return comp
    except (IOError, ValueError) as e:
        print(f"Errore nella lettura del file o nel parsing delle coordinate: {e}")
        return None
    
def calcola_angolo(x, width):
    cx = width / 2
    dx = x - cx
    angle = (dx / cx) * 30
    return angle

def update_map(kinect_lat, kinect_lon, face_positions, angle):
    m = folium.Map(location=[kinect_lat, kinect_lon], zoom_start=15)
    folium.Marker([kinect_lat, kinect_lon], popup='Kinect Location', icon=folium.Icon(color='blue')).add_to(m)
    geod = Geod(ellps="WGS84")
    
    for dist_meters, angle_rel, obj_id in face_positions:
        angle_rad = math.radians(angle + angle_rel)
        lon2, lat2, _ = geod.fwd(kinect_lon, kinect_lat, angle, dist_meters * 10)
        
        popup_text = f"Person ID: {obj_id}<br>Distance: {dist_meters:.2f} m"
        folium.Marker(
            location=[lat2, lon2],
            popup=folium.Popup(popup_text, max_width=250),
            icon=folium.Icon(color='green', icon='user', prefix='fa')
        ).add_to(m)
    
    m.save(f'./output_maps/kinect_map{current_timestamp}.html')

latitude, longitude = get_location('GPS_location_data.txt')
angle=compass_face('Compass.txt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiModalModel.MultiModalLSTMModel().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

while True:
    rgb_image = kinect.get_realtime_video()
    depth_map = kinect.get_realtime_depth()
    
    if rgb_image is None or depth_map is None:
        break

    height, width, _ = rgb_image.shape
    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    MAX_DEPTH = 5.5
    depth_map_meters = (depth_map / 255.0) * MAX_DEPTH

    results = model_yolo(rgb_image)
    face_positions = []
    detections = []
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            if conf > 0.5 and cls == 0: 
                person_center_x = x1 + (x2 - x1) // 2
                person_center_y = y1 + (y2 - y1) // 2
                detections.append((person_center_x, person_center_y, x1, y1, x2, y2))

    for (x, y, x1, y1, x2, y2) in detections:
        if latitude is not None and longitude is not None:
            if 0 <= y < depth_map_meters.shape[0] and 0 <= x < depth_map_meters.shape[1]:
                distance_meters = depth_map_meters[y, x]
                
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                cropped_face = rgb_image[y1:y2, x1:x2]
                if cropped_face.size == 0:
                    continue

                if len(cropped_face.shape) == 2 or cropped_face.shape[2] == 1:
                    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_GRAY2BGR)

                frame_rgb_tensor = transform(cropped_face).unsqueeze(0).to(device)

                numeric_data = [distance_meters, latitude, longitude, angle]
                numeric_tensor = torch.tensor(numeric_data, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    prediction = model(frame_rgb_tensor, numeric_tensor)
                    predicted_distance = prediction.item() * 2.5
                    angle_rel = calcola_angolo(x, width)
                    face_positions.append([predicted_distance, angle_rel, next_id])

                    cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(rgb_image, f"Human Face ID:{next_id}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(rgb_image, f"Distance: {predicted_distance:.2f} m",
                                (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    next_id += 1

    if latitude is not None and longitude is not None:
        update_map(latitude, longitude, face_positions, angle)

    cv2.imshow('Output', rgb_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
webbrowser.open(f'./output_maps/kinect_map{current_timestamp}.html')
