import numpy as np
def calculate_angle(point1, point2, point3):
    """Calculate angle at point2 formed by point1-point2-point3"""
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle