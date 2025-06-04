from ultralytics import YOLO

model = YOLO("models/best.pt")

result = model.predict("./input_videos/sinner.mp4", conf=0.3, save=True)