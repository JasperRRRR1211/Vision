import argparse
import os
import sys
from pathlib import Path
import torchvision
import numpy

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/NGYF3.mov',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_crop=True,  # save cropped prediction boxes
        save_txt = False,
        project=ROOT / 'runs/detect',  # save results to project/name
        line_thickness = 3
        ):
    source = str(source) # get path of the video

    #save directory
    save_dir = increment_path(Path(project) / 'exp', exist_ok=True)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride = model.stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    frame = cv2.VideoCapture(source)
    assert frame.isOpened()
    w = int(frame.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(frame.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = frame.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    # Use different variable names to avoid confusion
    out = cv2.VideoWriter(save_dir, codec, fps, (w, h))

    #Run
    while True:
        ret, img0 = frame.read()
        im = torch.from_numpy(im).to(device)
        if img0 is None:
            break

        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(device)
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim


        # Inference
        pred = model(img, augment=False)[0]

        # NMS
        pred = non_max_suppression(pred, 0.5, 0.5, classes=None, agnostic=False)

        # Process predictions
        for i, det in enumerate(pred):  # per images
            annotator = Annotator(img0, line_width=line_thickness, example=str(names))
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = img0.copy() if save_crop else img0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = '%s %.2f' % (names[int(cls)], conf)

                    # 获取原始边界框坐标
                    x_min, y_min, x_max, y_max = xyxy

                    # 在这里添加调整边界框大小的代码
                    # 例如，可以将宽度和高度乘以一个比例因子
                    scale_factor = 1.2  # 你可以根据需要调整这个比例因子
                    width = x_max - x_min
                    height = y_max - y_min
                    new_width = width * scale_factor
                    new_height = height * scale_factor

                    # 计算新的边界框坐标
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    x_min = x_center - new_width / 2
                    y_min = y_center - new_height / 2
                    x_max = x_center + new_width / 2
                    y_max = y_center + new_height / 2

                    # 绘制边界框或保存调整后的边界框
                    annotator.box_label([x_min, y_min, x_max, y_max], label, color=colors(c, True))
                    #save crops
                    save_one_box([x_min, y_min, x_max, y_max], imc,
                                 file=save_dir / 'crops' / names[c] / f'.jpg', BGR=True)
        out.write(img0)

    frame.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    check_requirements(exclude=('tensorboard', 'thop'))
    run(source='data/NGYF3.mov')
