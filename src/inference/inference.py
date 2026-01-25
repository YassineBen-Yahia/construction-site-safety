import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from analysis.detection_pipeline import process_frame
from inference.danger_scoring import score_safety, danger_to_msg


def get_color_by_risk_level(risk_level):
    """Return BGR color based on risk level"""
    colors = {
        'LOW': (0, 255, 0),        # Green
        'MODERATE': (0, 165, 255), # Orange
        'HIGH': (0, 69, 255),      # Red-Orange
        'CRITICAL': (0, 0, 255)    # Red
    }
    return colors.get(risk_level, (255, 255, 255))


def draw_person_box_and_info(frame, person_box, score, risk_level, dangers):
    """
    Draw colored bounding box around person and display danger info
    
    Args:
        frame: Video frame (BGR)
        person_box: Tuple of (x1, y1, x2, y2)
        score: Safety score (0-100)
        risk_level: Risk level string
        dangers: List of danger messages
    """
    x1, y1, x2, y2 = [int(v) for v in person_box]
    color = get_color_by_risk_level(risk_level)
    
    # Draw thick bounding box around person
    thickness = 3
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw score and risk level on top of box
    score_text = f"Score: {score}/100"
    risk_text = f"Risk: {risk_level}"
    
    # Background for text readability
    cv2.rectangle(frame, (x1, y1 - 60), (x1 + 250, y1), color, -1)
    cv2.putText(frame, score_text, (x1 + 5, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, risk_text, (x1 + 5, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw dangers list below the box
    #if dangers:
    #    y_offset = y2 + 25
    #    for i, danger in enumerate(dangers):
    #        # Truncate long danger messages
    #        danger_text = danger[:50] + "..." if len(danger) > 50 else danger
    #        cv2.putText(frame, f"• {danger_text}", (x1, y_offset + i * 25), 
    #                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return frame


def process_video(video_path, output_path=None, display=True):
    """
    Process a video file and color persons based on safety scoring
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video (optional)
        display: Whether to display video while processing
    
    Returns:
        Dict with processing statistics
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    stats = {
        'total_frames': total_frames,
        'frames_processed': 0,
        'avg_score': 0,
        'risk_distribution': {'LOW': 0, 'MODERATE': 0, 'HIGH': 0, 'CRITICAL': 0}
    }
    
    all_scores = []
    
    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame for dangers
        dangers_list = process_frame(frame)
        
        # Process each person in the frame
        for danger_info in dangers_list:
            person_box = danger_info['person_box']
            detected_dangers = danger_info['dangers']
            ppe = danger_info['ppe']
            
            # Create detection pipeline output format
            detection_output = {
                'dangers': detected_dangers,
                'ppe': ppe
            }
            
            # Score safety
            score, _, risk_level = score_safety(detection_output)
            #all_scores.append(score)
            #stats['risk_distribution'][risk_level] += 1
            
            # Draw person box with info
            frame = draw_person_box_and_info(frame, person_box, score, risk_level, detected_dangers)
        
        # Display frame
        if display:
            cv2.imshow('Safety Monitoring', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Write frame to output video
        if writer:
            writer.write(frame)
        
        stats['frames_processed'] = frame_count
        
        if frame_count % 30 == 0:
            print(f"Processed: {frame_count}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    
    # Calculate statistics
    if all_scores:
        stats['avg_score'] = np.mean(all_scores)
        stats['min_score'] = np.min(all_scores)
        stats['max_score'] = np.max(all_scores)
    
    print(f"\nProcessing complete!")
    print(f"Average score: {stats.get('avg_score', 0):.2f}/100")
    print(f"Risk distribution: {stats['risk_distribution']}")
    
    if output_path:
        print(f"Output saved to: {output_path}")
    
    return stats


def process_webcam(display=True, save_output=False):
    """
    Process real-time webcam feed for safety monitoring
    
    Args:
        display: Whether to display the feed
        save_output: Whether to save the feed to a video file
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise ValueError("Cannot open webcam")
    
    # Setup video writer if saving
    writer = None
    if save_output:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('safety_monitoring_output.mp4', fourcc, fps, (width, height))
    
    print("Webcam monitoring started. Press 'q' to quit.")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame for dangers
        dangers_list = process_frame(frame)
        
        # Process each person in the frame
        for danger_info in dangers_list:
            person_box = danger_info['person_box']
            detected_dangers = danger_info['dangers']
            ppe = danger_info['ppe']
            
            # Create detection pipeline output format
            detection_output = {
                'dangers': detected_dangers,
                'ppe': ppe
            }
            
            # Score safety
            score, _, risk_level = score_safety(detection_output)
            
            # Draw person box with info
            frame = draw_person_box_and_info(frame, person_box, score, risk_level, detected_dangers)
        
        # Display frame
        if display:
            cv2.imshow('Safety Monitoring - Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Write frame
        if writer:
            writer.write(frame)
    
    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage:
    # Process a video file
    video_path = "path/to/your/video.mp4"
    output_path = "output_video.mp4"
    
    # Uncomment to process a video file
    # stats = process_video(video_path, output_path, display=True)
    
    # Uncomment to process webcam
    # process_webcam(display=True, save_output=False)



