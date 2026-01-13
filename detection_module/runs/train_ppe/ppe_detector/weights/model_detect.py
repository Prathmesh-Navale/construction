import cv2
from ultralytics import YOLO

class PPEModel:
    def __init__(self, weights_path="runs/train_ppe/ppe_detector/weights/best1.pt", device="cpu", conf=0.5, iou=0.45):
        self.model = YOLO(weights_path)
        self.device = device
        self.conf = conf
        self.iou_threshold = iou
        self.class_names = self.model.names  # directly load from model

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf, iou=self.iou_threshold, device=self.device, verbose=False)
        r = results[0]
        dets = []
        if r.boxes is not None:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box.tolist())
                dets.append({
                    "box": (x1, y1, x2, y2),
                    "cls": int(cls.item()),
                    "conf": float(conf.item())
                })
        return dets

    def annotate(self, frame, detections):
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            cls = d["cls"]
            label = f"{self.class_names[cls]} {d['conf']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return out


if __name__ == "__main__":
    model = PPEModel(r"C:\1 Python Project\Construction Safety\runs\train_ppe\ppe_detector\weights\best1.pt", device="cpu")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = model.detect(frame)
        annotated = model.annotate(frame, detections)

        cv2.imshow("PPE Color Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
