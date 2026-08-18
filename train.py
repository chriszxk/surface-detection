from ultralytics import YOLO

# Load a model
model = YOLO("../model/deanet.yaml")

# Train the model
train_results = model.train(
    data="/home/xxx/xxx/xxx/ultralytics/data/xxx.yaml",  # path to dataset YAML
    epochs=500,  # number of training epochs
    imgsz=1024,  # training image size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    project='save_weights/',
    workers=24,
    resume=True,
)