import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.depth_analysor import calculate_distance_with_depth

def calculate_angle(point1, point2, point3):
    """Calculate angle at point2 formed by point1-point2-point3"""
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(np.array(point1) - np.array(point2))


def detect_vehicle_near_person(person_box, detections, depth_map, 
                                          depth_estimator, vehicle_classes=['vehicle', 'machinery'],
                                          depth_threshold=0.15, lateral_threshold=200):
    """
    Check if vehicle is dangerously close using depth estimation
    
    Args:
        depth_threshold: Maximum allowed depth difference (relative units)
        lateral_threshold: Maximum allowed lateral distance in pixels
    """
    px1, py1, px2, py2 = person_box
    person_depth = depth_estimator.get_object_depth(depth_map, person_box)
    
    for other_det in detections:
        if other_det['class'] in vehicle_classes:
            vehicle_box = other_det['box']
            vehicle_depth = depth_estimator.get_object_depth(depth_map, vehicle_box)
            
            # Calculate depth difference
            depth_diff = abs(person_depth - vehicle_depth)
            
            # Calculate lateral distance
            person_center_x = (px1 + px2) / 2
            person_center_y = (py1 + py2) / 2
            vx1, vy1, vx2, vy2 = vehicle_box
            vehicle_center_x = (vx1 + vx2) / 2
            vehicle_center_y = (vy1 + vy2) / 2
            
            lateral_dist = calculate_distance(
                (person_center_x, person_center_y),
                (vehicle_center_x, vehicle_center_y)
            )
            
            # Check if both depth and lateral distance are close
            if depth_diff < depth_threshold and lateral_dist < lateral_threshold:
                return True, {
                    'depth_difference': depth_diff,
                    'lateral_distance': lateral_dist,
                    'person_depth': person_depth,
                    'vehicle_depth': vehicle_depth
                }
    
    return False, None