import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ultralytics import YOLO
from utils.CONSTANTS import KEYPOINTS

class PoseEstimator:
    def __init__(self, model_path='../../models/yolov8n-pose.pt'):
        self.model = YOLO(model_path)

    def get_keypoint(self, keypoints, name, threshold=0.5):
        """Get keypoint coordinates if confidence is above threshold"""
        idx = KEYPOINTS[name]
        #if keypoints[idx][2] > threshold:  
        return (int(keypoints[idx][0]), int(keypoints[idx][1]))
        #return None
    
    def get_keypoints_coordinates(self, frame):
        results = self.model(frame)
        s=results[0].keypoints.xy
        keypoints = [kp.tolist() for kp in s]
        return keypoints,results