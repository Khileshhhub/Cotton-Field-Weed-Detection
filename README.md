Cotton Weed Detection using YOLO11m

This project detects weeds and crops in cotton fields using a YOLO11m object detection model. It supports training a custom model on your dataset and running detections on new images or videos.

1. Project Structure

cotton-weed-detection/
│
├── data/
│   └── dataset/
│       ├── images/           # training/validation/test images
│       ├── labels/           # YOLO-format annotation text files
│       └── data.yaml         # dataset configuration
│
├── models/
│   └── yolo11m/
│       └── best.pt           # trained YOLO11m weights
│
├── outputs/                  # detection outputs
│
└── README.md

2. Installation

Make sure Python 3.8+ is installed.

pip install ultralytics

If using Google Colab:

!pip install ultralytics

3. Dataset Setup

Your data.yaml should look like this:

train: /cotton-weed-detection/data/dataset/train/images
val: /cotton-weed-detection/data/dataset/val/images
test: /cotton-weed-detection/data/dataset/test/images

nc: 2
names: ['crop', 'weed']  # Add labels according to the dataset

4. Training the Model

Command-line:

yolo detect train \
    model=yolo11m.pt \
    data=/path/to/data.yaml \
    epochs=50 \
    imgsz=640 \
    batch=16

Python:

from ultralytics import YOLO

model = YOLO("yolo11m.pt")  # start from pretrained
model.train(
    data="/path/to/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)

Output weights:

runs/detect/train/weights/best.pt

5. Evaluating Model Performance

yolo val model=/path/to/best.pt data=/path/to/data.yaml split=test

Python:

metrics = model.val(data="/path/to/data.yaml", split="test")
print(metrics)

📊 Example Metrics Table:

Class

Precision

Recall

mAP@0.5

mAP@0.5:0.95

Crop

0.92

0.90

0.93

0.78

Weed

0.89

0.91

0.92

0.75

Overall

0.91

0.905

0.925

0.765

6. Running Detection

Python:

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

Command-line:

yolo predict \
    model=/path/to/best.pt \
    source=/path/to/images \
    conf=0.25 \
    imgsz=640

Predictions will be saved in:

/content/drive/MyDrive/cotton-weed-detection/results/detection_exp_all_images/



7. Notes

Lower conf=0.1 if detections are missing.

Make sure names in data.yaml match your labels.

Use GPU (device=0) for faster speed.

8. License

MIT License — you are free to use, modify, and distribute.

