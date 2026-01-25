from ultralytics import YOLO
import cv2
import numpy as np
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.CONSTANTS import KEYPOINTS
from utils.custom_utils import calculate_angle
from detection.pose_estimator import PoseEstimator

class DangerousPoseDetector:
    """Detect dangerous poses in construction sites"""
    
    # Keypoint indices
    
    
    def __init__(self, estimator=None):
        """Initialize pose detection model"""
        if isinstance(estimator, PoseEstimator):
            self.model = estimator
        else:
            self.estimator = PoseEstimator()
            self.model = self.estimator
 
    
    def detect_bending_over(self, keypoints, threshold=140):
        """
        Detect if person is bending over (back bent)
        Dangerous for lifting heavy objects
        """
        shoulder = self.estimator.get_keypoint(keypoints, 'left_shoulder') or \
                   self.estimator.get_keypoint(keypoints, 'right_shoulder')
        hip = self.estimator.get_keypoint(keypoints, 'left_hip') or \
              self.estimator.get_keypoint(keypoints, 'right_hip')
        knee = self.estimator.get_keypoint(keypoints, 'left_knee') or \
               self.estimator.get_keypoint(keypoints, 'right_knee')
        
        if shoulder and hip and knee:
            angle = calculate_angle(shoulder, hip, knee)
            if angle and angle < threshold:
                return True, angle, "Person bending over - potential back injury risk"
        return False, None, None
    
    def detect_overhead_work(self, keypoints):
        """
        Detect if person is working overhead
        Dangerous without proper support
        """
        nose = self.estimator.get_keypoint(keypoints, 'nose')
        wrist_left = self.estimator.get_keypoint(keypoints, 'left_wrist')
        wrist_right = self.estimator.get_keypoint(keypoints, 'right_wrist')
        shoulder_left = self.estimator.get_keypoint(keypoints, 'left_shoulder')
        shoulder_right = self.estimator.get_keypoint(keypoints, 'right_shoulder')
        
        if nose and (wrist_left or wrist_right) and (shoulder_left or shoulder_right):
            # Check if wrists are significantly above head
            wrist = wrist_left if wrist_left else wrist_right
            shoulder = shoulder_left if shoulder_left else shoulder_right
            
            if wrist[1] < nose[1] - 50:  # Y-axis decreases upward
                return True, "Overhead work detected - fall/strain risk"
        return False, None
    
    def detect_squatting(self, keypoints, max_angle=100):
        """
        Detect deep squatting posture
        Can be dangerous if prolonged
        """
        hip = self.estimator.get_keypoint(keypoints, 'left_hip') or \
              self.estimator.get_keypoint(keypoints, 'right_hip')
        knee = self.estimator.get_keypoint(keypoints, 'left_knee') or \
               self.estimator.get_keypoint(keypoints, 'right_knee')
        ankle = self.estimator.get_keypoint(keypoints, 'left_ankle') or \
                self.estimator.get_keypoint(keypoints, 'right_ankle')
        
        if hip and knee and ankle:
            angle = calculate_angle(hip, knee, ankle)
            if angle and angle < max_angle:
                return True, angle, "Deep squat detected - prolonged strain risk"
        return False, None, None
    
    
    
    def detect_imbalanced_stance(self, keypoints):
        """
        Detect imbalanced stance normalized by shoulder width
        """
        left_ankle = self.estimator.get_keypoint(keypoints, 'left_ankle')
        right_ankle = self.estimator.get_keypoint(keypoints, 'right_ankle')
        left_hip = self.estimator.get_keypoint(keypoints, 'left_hip')
        right_hip = self.estimator.get_keypoint(keypoints, 'right_hip')
        left_shoulder = self.estimator.get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self.estimator.get_keypoint(keypoints, 'right_shoulder')
        
        if all([left_ankle, right_ankle, left_hip, right_hip, 
                left_shoulder, right_shoulder]):
            
            # Calculate shoulder width as reference scale
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            
            # Avoid division by zero
            if shoulder_width < 10:
                return False, None, None
            
            # Calculate centers
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            ankle_center_x = (left_ankle[0] + right_ankle[0]) / 2
            
            # Calculate offset
            offset = abs(hip_center_x - ankle_center_x)
            
            # Normalize by shoulder width
            normalized_offset = offset / shoulder_width
            
            # Threshold: offset should not exceed 0.8x shoulder width
            if normalized_offset > 0.65:
                return True, normalized_offset, f"Imbalanced stance - fall risk (offset: {normalized_offset:.2f}x shoulder width)"
        
        return False, None, None
    
    def analyze_frame(self, frame):
        """
        Analyze a single frame for dangerous poses
        Returns: frame with annotations, list of dangers detected
        """
        all_keypoints,results = self.model.get_keypoints_coordinates(frame)

        dangers = []
        
    
        if all_keypoints is not None and len(all_keypoints) > 0:
            for idx,keypoints in enumerate(all_keypoints):
                #for idx, keypoints in enumerate(result):
    #                    keypoints = keypoints.cpu().numpy()
                person_dangers = []
                
                # Check for various dangerous poses
                is_bending, bend_angle, bend_msg = self.detect_bending_over(keypoints)
                if is_bending:
                    person_dangers.append(bend_msg)
                
                is_overhead, overhead_msg = self.detect_overhead_work(keypoints)
                if is_overhead:
                    person_dangers.append(overhead_msg)
                
                is_squatting, squat_angle, squat_msg = self.detect_squatting(keypoints)
                if is_squatting:
                    person_dangers.append(squat_msg)
                
                is_imbalanced, offset, imbalance_msg = self.detect_imbalanced_stance(keypoints)
                if is_imbalanced:
                    person_dangers.append(imbalance_msg)
                
                if person_dangers:
                    dangers.append({
                        'person_id': idx,
                        'dangers': person_dangers,
                        'keypoints': keypoints
                    })
    
        # Draw results
        #annotated_frame = results[0].plot()
        
        # Add danger warnings
        #y_offset = 30
        #for danger in dangers:
        #    for msg in danger['dangers']:
        #        cv2.putText(annotated_frame, msg, (10, y_offset),
        #                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #        y_offset += 25
        
        return dangers
