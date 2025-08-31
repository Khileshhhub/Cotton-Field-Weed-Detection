# Cotton Weed Detection using YOLO11m

This project detects **weeds** and **crops** in cotton fields using a YOLO11m object detection model. It supports **training** a custom model on your dataset and **running detections** on images.

---

## Installation

Make sure Python 3.8+ is installed.

pip install ultralytics


If using Google Colab:

!pip install ultralytics


You can also install all required dependencies using the `requirements.txt` file provided:
[Download requirements.txt](https://drive.google.com/uc?id=YOUR_FILE_ID_HERE)
Then run:

pip install -r requirements.txt


---

## Dataset Setup

Your `data.yaml` should look like this:


train: /cotton-weed-detection/data/dataset/train/images
val: /cotton-weed-detection/data/dataset/val/images
test: /cotton-weed-detection/data/dataset/test/images

nc: 2
names: ['crop', 'weed']  # Add labels according to your dataset


---

## Training the Model

**Command-line:**

yolo detect train \
    model=yolo11m.pt \
    data=/path/to/data.yaml \
    epochs=50 \
    imgsz=640 \
    batch=16

**Python:**

from ultralytics import YOLO

model = YOLO("yolo11m.pt")  # start from pretrained
model.train(
    data="/path/to/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)


Output weights will be saved in:

runs/detect/train/weights/best.pt

---

## Running Detection

**Python:**

from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/cotton-weed-detection/models/yolo11m/best.pt")

results = model.predict(
    source="/content/drive/MyDrive/cotton-weed-detection/data/dataset/test/images",
    conf=0.25,
    imgsz=640,
    save=True,
    project="/content/drive/MyDrive/cotton-weed-detection/results",
    name="detection_exp_all_images",
    exist_ok=True
)

**Command-line:**

yolo predict \
    model=/path/to/best.pt \
    source=/path/to/images \
    conf=0.25 \
    imgsz=640

Predictions will be saved in:

/content/drive/MyDrive/cotton-weed-detection/results/detection_exp_all_images/
---

## Notes

* Lower `conf=0.1` if detections are missing.
* Make sure `names` in `data.yaml` match your labels.
* Use GPU (`device=0`) for faster speed.

---

## License

MIT License — you are free to use, modify, and distribute.
