# YoloV3
Object detection with yolov3 algorithm using  Tensorflow-2 

Clone or Download the files.

1. Downlaod the coco  weight from ## YOLOv3 implementation in TensorFlow 2.0

![alt text][image]

#### Installation

```bash
pip3 install -r ./requirements.txt
wget https://pjreddie.com/media/files/yolov3.weights -O ./yolov3.weights
```

#### Detect

```bash
python detect.py 
```


```

#### Papers and thanks

- [YOLO website](https://pjreddie.com/darknet/yolo/)
- [YOLOv3 paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [NMS paper](https://arxiv.org/pdf/1704.04503.pdf)
- [NMS implementation](https://github.com/bharatsingh430/soft-nms)
- [GIOU Paper](https://giou.stanford.edu/GIoU.pdf)
- [DarkNet Implementation](https://github.com/pjreddie/darknet)
- [YOLO implementation](https://github.com/zzh8829/yolov3-tf2)


[image]: ./YOLOV3/output_0.jpg "Logo Title Text 2"
