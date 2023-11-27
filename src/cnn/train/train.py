from roboflow import Roboflow
import os

# check if Driver-behaviors-10 is exist
if not os.path.exists("datasets/Driver-behaviors-10"):
    # rf = Roboflow(api_key=os.environ.get("ROBOFLOW_KEY"))
    # project = rf.workspace("jui").project("driver-behaviors")
    # dataset = project.version(10).download("yolov8")
    os.rename("Driver-behaviors-10", "datasets/Driver-behaviors-10")

print("Dataset loaded")

from ultralytics import YOLO

# if data.yaml not exist, move it from Driver-behaviors-10
if not os.path.exists("data.yaml"):
    os.rename("datasets/Driver-behaviors-10/data.yaml", "data.yaml")


model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=20,
    imgsz=640,
    batch=1,
    workers=16,
    project="driver-behaviors",
)

# move best.pt to ../Models/best.pt
os.rename("driver-behaviors/train/weights/best.pt", "../Models/best.pt")
