# Ultralytics YOLO 🚀, GPL-3.0 license
import cv2
import torch
import easyocr
import time
import pytesseract
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import os
import serial

#ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)

#os.environ['TESSDATA_PREFIX'] = '/home/peerawit/Desktop/Project/plate/tessdata'
reader = easyocr.Reader(['en'], gpu =True)

class Prediction:
    """Class for storing prediction information"""
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    score: float


def ocr_image(roi, coordinates):
    x, y, w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])
    frame = roi[y:h, x:w]
    result = reader.readtext(frame, add_margin=0.3, width_ths=0.9, text_threshold=0.7, link_threshold=0.5, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    # Sort the detected text by x-coordinate position
    result = sorted(result, key=lambda x: x[0][0][0])

    text = ""
    for res in result:
        if len(result) == 1:
            text = res[1]
        if len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2:
            text = res[1]
            #file_path = '/home/peerawit/Desktop/Project/plate/plate_after_detect/licens_plates' + str(file_number) + '.txt'
            #text = res[1]file_path = '/home/peerawit/Desktop/Project/plate/plate_after_detect/licens_plates' + str(file_number) + '.txt'
            #file_path = '/home/peerawit/Desktop/Project/plate/plate_after_detect/licens_plates1' + str(file_number) + '.txt'
            #with open(file_path, 'w') as f:
                #f.write(text)
    return text

"""
import pytesseract

def ocr_image(dist_transform, coordinates, files_number):
    x, y, w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])
    frame = dist_transform[y:h, x:w]

    # Set Tesseract config to use the Thai language model
    #config = '-c tessedit_char_whitelist=0123456789กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ --psm 6'
    config = '-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTWXYZ --psm 7'
    text = pytesseract.image_to_string(frame, lang='eng', config=config)

    # Split the text by lines and return only the first line
    #lines = text.strip().split('\n')
    #return lines[0] if len(lines) > 0 else ''
    file_path = '/home/peerawit/Desktop/Project/plate/plate_after_detect/licens_plates' + str(files_number) + '.txt'
    with open(file_path, 'w') as file:
        file.write(text)
    #if text != "":
        #ser.write("90\r".encode())
        #time.sleep(8)
        #ser.write("0\r".encode())

    return text
"""


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors
        predictions = []
        if len(det) == 0:
            return f'{log_string}(no detections), '
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # write
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.args.save_txt:  # Write to file
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                name = f'id:{int(d.id.item())} {self.model.names[c]}' if d.id is not None else self.model.names[c]
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                xyxy = d.xyxy.squeeze()
                text_ocr = ocr_image(im0, xyxy)
                label = text_ocr
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return f'{log_string}'


def predict(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or 'yolov8n.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()
