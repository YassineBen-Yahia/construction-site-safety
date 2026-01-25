import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.custom_utils import detect_vehicle_near_person
from analysis.pose_analysor import DangerousPoseDetector
from analysis.depth_analysor import DepthEstimator
from detection.ppe_detector import PPEDetector
ppe= PPEDetector("C:/ML/construction site safety/models/ppe/best.pt")
pose_detector=DangerousPoseDetector()
depth_estimator=DepthEstimator()


def group_detections_by_person(detections):
    person_detections = {}
    for detection in detections:
        if detection['class'] == 'Person':
            person_box = tuple(detection['box'])
            person_detections[person_box] = []
    
    for detection in detections:
        if detection['class'] != 'Person' and detection.get('associated_person_box'):
            associated_box = tuple(detection['associated_person_box'])
            if associated_box in person_detections:
                person_detections[associated_box].append(detection)
    
    return person_detections


def link_keypoints_to_detections(detections, pose_keypoints):
    linked_data = {}
    for person_box, ppe_list in group_detections_by_person(detections).items():
        # Find corresponding keypoints
        for keypoint in pose_keypoints:
            kp_x = keypoint[0][0]
            kp_y = keypoint[0][1]
            px1, py1, px2, py2 = person_box
            if px1 <= kp_x <= px2 and py1 <= kp_y <= py2:
                linked_data[person_box] = {
                    'ppe': ppe_list,
                    'keypoints': keypoint
                }
                break
    return linked_data


def process_frame(frame):
    """Process a single frame for risk assessment"""
    detections = ppe.detect(frame)
    keypoints_list, pose_results = pose_detector.model.get_keypoints_coordinates(frame)
    
    linked_data = link_keypoints_to_detections(detections, keypoints_list)
    
    dangers = []
    for person_box, data in linked_data.items():
        person_dangers = []
        keypoints = data['keypoints']
        
        # Check for dangerous poses
        bending, angle, msg = pose_detector.detect_bending_over(keypoints)
        if bending:
            person_dangers.append(msg)
        
        overhead, msg = pose_detector.detect_overhead_work(keypoints)
        if overhead:
            person_dangers.append(msg)

        squatting, angle, msg = pose_detector.detect_squatting(keypoints)
        if squatting:
            person_dangers.append(msg)

        imbalanced, offset, msg = pose_detector.detect_imbalanced_stance(keypoints)
        if imbalanced:
            person_dangers.append(msg)
        
        ppe_list = [f['class'] for f in data['ppe']]

        depth_map = depth_estimator.estimate_depth(frame)
        
        # Check for vehicles/machinery nearby
        s, details = detect_vehicle_near_person(
            person_box, detections, depth_map, depth_estimator
        )
        if s:
            person_dangers.append("Vehicle or machinery detected near person - risk of accidents")
        
        dangers.append({
            'person_box': person_box,
            'dangers': person_dangers,
            'ppe': ppe_list,
         })
    
    return dangers