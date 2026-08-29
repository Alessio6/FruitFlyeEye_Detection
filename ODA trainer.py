from ultralytics import YOLO
import yaml
import shutil
import cv2
import numpy as np
from pathlib import Path

def makeconfig(datafolder):
    data = Path(datafolder)
    
    config = {
        'path': str(data.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'left_eye',
            1: 'right_eye',
            2: 'eye_top',
            3: 'antenna'
        },
        'nc': 4,
        'task': 'segment'
    }
    
    configfile = data / "data_config.yaml"
    with open(configfile, 'w') as f:
        yaml.dump(config, f)
    
    return configfile

def train(configfile, outfolder):
    print("Starting training...")
    model = YOLO('yolov8n-seg.pt')
    
    model.train(
        data=str(configfile),
        epochs=100,
        imgsz=640,
        batch=4,
        patience=20,
        save=True,
        device='cpu',
        project=outfolder,
        name='fly_eye_seg',
        exist_ok=True,
        plots=True,
        cache=False,
        #Turn off augmentations that mess up the tiny features
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
    )
    print("Done training")
    return model

script = Path(__file__).parent
org = "ODADataset"
labels = script / org/ "labels"
images = script / org/ "images"
dataset = script / "ODADataset"
output = script / "oda_training_seg"

print("Labels: " + str(labels))
print("Images: " + str(images))

if not images.exists():
    print("Images folder not found")
    exit()
if not labels.exists():
    print("Labels folder not found")
    exit()

config = makeconfig(dataset)
model = train(config, output)

best = output / "fly_oda_seg" / "weights" / "best_oda.pt"
print("Model: " + str(best))