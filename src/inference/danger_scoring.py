danger_to_msg={
    'bending_over': 'Person bending over - potential back injury risk',
    'overhead_work': 'Overhead work detected - fall/strain risk',
    'squatting': 'Deep squat detected - prolonged strain risk',
    'imbalanced_stance': 'Imbalanced stance - fall risk',
    'vehicle_nearby': 'Vehicle or machinery detected near person - risk of accidents',

}
msg_to_danger={v:k for k,v in danger_to_msg.items()}





def score_safety(detection_pipeline_output):
    """
    Score safety based on detection pipeline output
    Uses a 0-100 scale where 100 is safest
    
    Returns:
        tuple: (safety_score, list of detected dangers, risk_level)
    """
    dangers = detection_pipeline_output['dangers']
    ppe = detection_pipeline_output['ppe']
    
    score = 100  # Start with maximum safety score
    
    # Define severity levels for each danger (points to deduct)
    danger_penalties = {
        danger_to_msg['vehicle_nearby']: 25,        # High risk - machinery/vehicle
        danger_to_msg['imbalanced_stance']: 15,     # Medium-high risk - fall risk
        danger_to_msg['overhead_work']: 18,         # Medium-high risk - fall/strain
        danger_to_msg['bending_over']: 12,          # Medium risk - back injury
        danger_to_msg['squatting']: 10,             # Medium risk - strain
    }
    
    # PPE penalties (points to deduct if missing)
    ppe_penalties = {
        'NO-Hardhat': 20,                           # Critical PPE
        'NO-Safety Vest': 15,                       # Important PPE
        'NO-Mask': 12,                              # Important PPE
    }
    
    # Apply danger penalties
    for danger in dangers:
        if danger in danger_penalties:
            score -= danger_penalties[danger]
    
    # Apply PPE penalties
    for missing_ppe in ppe:
        if missing_ppe in ppe_penalties:
            score -= ppe_penalties[missing_ppe]
    
    # Apply multiplier for high-risk combinations
    high_risk_dangers = [danger_to_msg['vehicle_nearby'], danger_to_msg['imbalanced_stance']]
    critical_ppe = ['NO-Hardhat', 'NO-Safety Vest']
    
    has_high_risk_danger = any(d in dangers for d in high_risk_dangers)
    has_critical_ppe_missing = any(p in ppe for p in critical_ppe)
    
    if has_high_risk_danger and has_critical_ppe_missing:
        score -= 20  # Additional penalty for dangerous situation with missing PPE
    
    # Determine risk level
    if score >= 80:
        risk_level = "LOW"
    elif score >= 60:
        risk_level = "MODERATE"
    elif score >= 40:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"
    
    # Clamp score between 0 and 100
    score = max(0, min(100, score))
    
    return score, dangers, risk_level