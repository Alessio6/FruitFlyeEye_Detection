from ultralytics import YOLO
import yaml
import shutil
import cv2
import numpy as np
from pathlib import Path

def setupdata(labfolder, imgfolder, outfolder):
    out = Path(outfolder)
    labpath = Path(labfolder)
    imgpath = Path(imgfolder)
    
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "fly_labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "fly_labels" / "val").mkdir(parents=True, exist_ok=True)
    print("Setting up dataset...")
    
    labfiles = list(labpath.glob("*.txt"))
    if not labfiles:
        print("No labels found!")
        return False
    print("Found " + str(len(labfiles)) + " labels")
    
    #80% train, 20% validation
    split = int(len(labfiles) * 0.8)
    trainlabs = labfiles[:split]
    vallabs = labfiles[split:]
    
    train_ok = 0
    train_bad = 0
    for lab in trainlabs:
        shutil.copy(lab, out / "labels" / "train" / lab.name)
        
        imgname = lab.stem
        img = imgpath / (imgname + ".tiff")
        
        #Try stripping the number prefix if the image wasn't found
        if not img.exists():
            parts = imgname.split('_', 1)
            if len(parts) > 1 and parts[0].isdigit():
                imgname = parts[1]
                img = imgpath / (imgname + ".tiff")
        
        if img.exists():
            image = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            
            #Some tiffs are 16bit so this normalises to 8bit
            if image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            #Convert greyscale or RGBA to RGB
            if len(image.shape) == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                rgb = image
            elif image.shape[2] == 4:
                rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                continue
            
            cv2.imwrite(str(out / "images" / "train" / (lab.stem + ".jpg")), rgb)
            train_ok += 1
        else:
            train_bad += 1
    
    val_ok = 0
    val_bad = 0
    for lab in vallabs:
        shutil.copy(lab, out / "labels" / "val" / lab.name)
        
        imgname = lab.stem
        img = imgpath / (imgname + ".tiff")
        
        if not img.exists():
            parts = imgname.split('_', 1)
            if len(parts) > 1 and parts[0].isdigit():
                imgname = parts[1]
                img = imgpath / (imgname + ".tiff")
        
        if img.exists():
            image = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            
            if image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            if len(image.shape) == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                rgb = image
            elif image.shape[2] == 4:
                rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                continue
            
            cv2.imwrite(str(out / "images" / "val" / (lab.stem + ".jpg")), rgb)
            val_ok += 1
        else:
            val_bad += 1
    
    print("Train: " + str(train_ok) + " ok, " + str(train_bad) + " missing")
    print("Val: " + str(val_ok) + " ok, " + str(val_bad) + " missing")
    
    if train_ok == 0:
        print("No training images found!")
        return False
    
    return True

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
    
    configfile = data / "dataset.yaml"
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
labels = script / "fly_labels"
images = script / "images"
dataset = script / "fly_dataset_seg"
output = script / "fly_training_seg"

print("Labels: " + str(labels))
print("Images: " + str(images))

if not images.exists():
    print("Images folder not found")
    exit()
if not labels.exists():
    print("Labels folder not found")
    exit()

ok = setupdata(labels, images, dataset)
if not ok:
    exit()

config = makeconfig(dataset)
model = train(config, output)

best = output / "fly_eye_seg" / "weights" / "best.pt"
print("Model: " + str(best))
