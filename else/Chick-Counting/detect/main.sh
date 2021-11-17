# Train YOLOv5s on COCO128 for 3 epochs
python train.py --img 1280 --batch 8 --epochs 100 --data chicken.yaml --weights weights/yolov5x.pt --name exp140_docker_train --device 0
