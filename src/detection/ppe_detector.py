from ultralytics import YOLO
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.CONSTANTS import PPE_CLASSES
class PPEDetector:
    def __init__(self, model_path='../../models/ppe/best.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)
        return self.get_ppe_detections(results, PPE_CLASSES)
 
    
    def get_ppe_detections(self, results, ppe_classes=PPE_CLASSES):
        detections = []
        for result in results:
            for box in result.boxes:
                has_person = False
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                if cls_id != 5 and cls_id != 8 and cls_id != 9 and not has_person: # Exclude 'Person' class and 'machinery' and 'vehicle'
                    #search for the corresponding person box
                    for person_box in result.boxes:
                        person_cls_id = int(person_box.cls[0])
                        
                        if person_cls_id == 5:
                            #check if the ppe box is intersecting with the person box
                            px1, py1, px2, py2 = person_box.xyxy[0]
                            bx1, by1, bx2, by2 = box.xyxy[0]
                            if not (bx2 < px1 or bx1 > px2 or by2 < py1 or by1 > py2):
                                person_box=person_box
                                has_person = True
                                break
                            
                if cls_name in ppe_classes:
                    detections.append({
                        'class': cls_name,
                        'confidence': float(box.conf[0]),
                        'box': box.xyxy[0].tolist(),
                        'associated_person_box': person_box.xyxy[0].tolist() if has_person else None
                    })
        return detections
