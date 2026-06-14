from detector import ObjectDetector


class ObjectTracker:
    def __init__(self, model="yolo11n.pt", conf=0.35, imgsz=960, tracker="bytetrack.yaml", device=None):
        self.det = ObjectDetector(model=model, conf=conf, imgsz=imgsz, tracker=tracker, device=device)

    def reset(self):
        self.det.reset()

    def track(self, frame, cls=None):
        return self.det.track(frame, cls=cls)
