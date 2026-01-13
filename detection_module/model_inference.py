import numpy as np
from ultralytics import YOLO
import cv2

class PPEModel:
    def __init__(self, weights_path='weights/best.pt', device='cpu', conf=0.5, iou=0.45):
        self.model = YOLO(weights_path)
        self.device = device
        self.conf = conf
        self.iou_threshold = iou

        # Class mapping (from best.pt dataset)
        self.class_names = {
            0: 'Hardhat',
            1: 'Mask',
            2: 'NO-Hardhat',
            3: 'NO-Mask',
            4: 'NO-Safety Vest',
            5: 'Person',
            6: 'Safety Cone',
            7: 'Safety Vest'
        }

    def detect(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=img, conf=self.conf, iou=self.iou_threshold,
                                     verbose=False, device=self.device)
        r = results[0]
        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box.tolist())
                dets.append({'box': (x1, y1, x2, y2), 'cls': int(cls.item()), 'conf': float(conf.item())})
        return dets

    @staticmethod
    def iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area != 0 else 0.0

    def check_ppe_per_person(self, detections, iou_threshold=0.05, conf_margin=0.05):
        # Positive PPE classes
        pos_map = {0: 'Helmet', 1: 'Mask', 7: 'Safety-Vest'}
        # Negative PPE classes
        neg_map = {2: 'Helmet', 3: 'Mask', 4: 'Safety-Vest'}

        people = [d for d in detections if d['cls'] == 5]  # Person = class 5
        others = [d for d in detections if d['cls'] != 5]
        results = []

        for person in people:
            px1, py1, px2, py2 = person['box']
            p_h = py2 - py1

            present = {'Helmet': False, 'Safety-Vest': False, 'Mask': False}
            details = {k: {'pos_conf': 0.0, 'neg_conf': 0.0, 'pos_box': None, 'neg_box': None} for k in present.keys()}

            for item in others:
                cls = item['cls']
                iou_val = self.iou(person['box'], item['box'])
                if iou_val < iou_threshold:
                    continue

                cx = (item['box'][0] + item['box'][2]) / 2.0
                cy = (item['box'][1] + item['box'][3]) / 2.0

                if cls in pos_map:
                    ppe_name = pos_map[cls]
                    is_negative = False
                elif cls in neg_map:
                    ppe_name = neg_map[cls]
                    is_negative = True
                else:
                    continue  # Ignore non-PPE classes

                # Spatial checks
                spatial_ok = True
                if ppe_name == 'Helmet' and cy > py1 + 0.40 * p_h: spatial_ok = False
                if ppe_name == 'Safety-Vest' and not (py1 + 0.25 * p_h <= cy <= py1 + 0.85 * p_h): spatial_ok = False
                if ppe_name == 'Mask' and cy > py1 + 0.45 * p_h: spatial_ok = False
                if not spatial_ok:
                    continue

                if not is_negative:
                    if item['conf'] > details[ppe_name]['pos_conf']:
                        details[ppe_name]['pos_conf'] = item['conf']
                        details[ppe_name]['pos_box'] = item['box']
                else:
                    if item['conf'] > details[ppe_name]['neg_conf']:
                        details[ppe_name]['neg_conf'] = item['conf']
                        details[ppe_name]['neg_box'] = item['box']

            for k in present.keys():
                posc, negc = details[k]['pos_conf'], details[k]['neg_conf']
                if posc == 0 and negc == 0:
                    present[k] = False
                elif posc >= negc + conf_margin:
                    present[k] = True
                elif negc >= posc + conf_margin:
                    present[k] = False
                else:
                    present[k] = posc > negc

            missing = [f"No {k}" for k, v in present.items() if not v]

            results.append({'person_box': person['box'], 'present': present, 'missing': missing,
                            'details': details, 'conf': person.get('conf', 1.0)})

        return results

    def annotate_frame(self, frame, detections, ppe_results=None):
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d['box']
            cls = d['cls']
            name = self.class_names.get(cls, str(cls))
            if name == "Person":
                color = (0, 255, 0)
            elif name.startswith("NO-"):
                color = (0, 0, 255)
            elif name in ["Hardhat", "Mask", "Safety Vest"]:
                color = (0, 165, 255)
            else:
                color = (255, 0, 0)  # Other objects (cone, machinery, vehicle)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{name} {d['conf']:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if ppe_results:
            for idx, pr in enumerate(ppe_results):
                x1, y1, x2, y2 = pr['person_box']
                if all(pr['present'].values()):
                    text, color = "✅ All PPE OK", (0, 255, 0)
                else:
                    text, color = "❌ " + ", ".join(pr['missing']), (0, 0, 255)
                cv2.putText(out, text, (x1, y2 + 18 + idx * 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        return out
