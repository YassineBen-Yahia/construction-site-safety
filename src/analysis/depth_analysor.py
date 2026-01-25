import torch
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, model_type="DPT_Large"):
        """Initialize depth estimation model"""
        # Using MiDaS
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
    
    def estimate_depth(self, frame):
        """
        Estimate depth map for frame
        Returns: depth map (closer objects have smaller values)
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        return depth_map
    
    def get_object_depth(self, depth_map, box):
        """
        Get average depth of object from bounding box
        box: (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = map(int, box)
        roi = depth_map[y1:y2, x1:x2]
        
        # Use median depth to avoid outliers
        return np.median(roi)


def calculate_distance_with_depth(person_box, vehicle_box, depth_map, depth_estimator):
    """
    Calculate distance between person and vehicle using depth information
    
    Returns:
        distance: Relative depth difference (smaller = closer)
        person_depth: Depth value at person location
        vehicle_depth: Depth value at vehicle location
    """
    person_depth = depth_estimator.get_object_depth(depth_map, person_box)
    vehicle_depth = depth_estimator.get_object_depth(depth_map, vehicle_box)
    
    # Calculate depth difference
    depth_difference = abs(person_depth - vehicle_depth)
    
    # Calculate 2D distance (lateral distance)
    px1, py1, px2, py2 = person_box
    vx1, vy1, vx2, vy2 = vehicle_box
    
    person_center = np.array([(px1 + px2) / 2, (py1 + py2) / 2])
    vehicle_center = np.array([(vx1 + vx2) / 2, (vy1 + vy2) / 2])
    
    lateral_distance = np.linalg.norm(person_center - vehicle_center)
    
    return {
        'depth_difference': depth_difference,
        'lateral_distance': lateral_distance,
        'person_depth': person_depth,
        'vehicle_depth': vehicle_depth,
        'is_dangerous': depth_difference < 0.1  # Threshold for danger
    }