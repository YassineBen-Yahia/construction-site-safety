"""
Test script for Construction Site Safety API
"""

import requests
import json
import time
from pathlib import Path


def test_health_check(base_url="http://localhost:8000"):
    """Test health check endpoint"""
    print("\n=== Testing Health Check ===")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_upload_video(video_path, base_url="http://localhost:8000"):
    """Test video upload and processing"""
    print("\n=== Testing Video Upload ===")
    
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        return None
    
    try:
        with open(video_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{base_url}/api/v1/process-video",
                files=files,
                timeout=30
            )
        
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        return data.get("job_id")
    
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_job_status(job_id, base_url="http://localhost:8000"):
    """Test job status endpoint"""
    print(f"\n=== Testing Job Status (ID: {job_id}) ===")
    
    try:
        response = requests.get(
            f"{base_url}/api/v1/job/{job_id}",
            timeout=5
        )
        
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Job Status: {data.get('status')}")
        print(f"Progress: {data.get('progress')}%")
        
        if data.get('stats'):
            print(f"Average Score: {data['stats'].get('avg_score')}")
            print(f"Risk Distribution: {data['stats'].get('risk_distribution')}")
        
        return data
    
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_list_jobs(base_url="http://localhost:8000"):
    """Test job listing endpoint"""
    print("\n=== Testing List Jobs ===")
    
    try:
        response = requests.get(
            f"{base_url}/api/v1/jobs",
            timeout=5
        )
        
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Total Jobs: {data.get('total')}")
        
        for job in data.get('jobs', [])[:3]:  # Show first 3 jobs
            print(f"  - {job['job_id']}: {job['status']} ({job['filename']})")
        
        return data
    
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_statistics(base_url="http://localhost:8000"):
    """Test statistics endpoint"""
    print("\n=== Testing Statistics ===")
    
    try:
        response = requests.get(
            f"{base_url}/api/v1/stats",
            timeout=5
        )
        
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Total Jobs: {data.get('total_jobs')}")
        print(f"Completed: {data.get('completed')}")
        print(f"Processing: {data.get('processing')}")
        print(f"Failed: {data.get('failed')}")
        print(f"Average Safety Score: {data.get('average_safety_score')}")
        
        return data
    
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_frame_analysis(image_path, base_url="http://localhost:8000"):
    """Test frame analysis endpoint"""
    print(f"\n=== Testing Frame Analysis ===")
    
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return None
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{base_url}/api/v1/analyze-frame",
                files=files,
                timeout=30
            )
        
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"People Detected: {data.get('people_detected')}")
        
        for person in data.get('analysis', []):
            print(f"  - Person {person['person_id']}: Score {person['safety_score']}, Risk: {person['risk_level']}")
        
        return data
    
    except Exception as e:
        print(f"Error: {e}")
        return None


def monitor_job(job_id, base_url="http://localhost:8000", check_interval=5, max_checks=120):
    """Monitor a job until completion"""
    print(f"\n=== Monitoring Job (ID: {job_id}) ===")
    
    checks = 0
    while checks < max_checks:
        data = test_job_status(job_id, base_url)
        
        if not data:
            return False
        
        status = data.get('status')
        progress = data.get('progress', 0)
        
        print(f"[{checks * check_interval}s] Status: {status}, Progress: {progress}%")
        
        if status == "completed":
            print("✓ Job completed!")
            return True
        elif status == "failed":
            print(f"✗ Job failed: {data.get('error')}")
            return False
        
        checks += 1
        time.sleep(check_interval)
    
    print("⚠ Job monitoring timeout")
    return False


def run_all_tests(base_url="http://localhost:8000"):
    """Run all tests"""
    print("=" * 60)
    print("Construction Site Safety API - Test Suite")
    print("=" * 60)
    
    # Test 1: Health check
    if not test_health_check(base_url):
        print("\n API is not running! Start it with:")
        print("   python -m uvicorn api.main:app --reload")
        return
    
    print("\n✓ API is running!")
    
    # Test 2: List existing jobs
    test_list_jobs(base_url)
    
    # Test 3: Statistics
    test_statistics(base_url)
    
    # Test 4: Frame analysis (if image exists)
    test_frame_path = "data/f.webp"
    if Path(test_frame_path).exists():
        test_frame_analysis(test_frame_path, base_url)
    else:
        print(f"\n⚠ Skipping frame analysis test (no image at {test_frame_path})")
    
    # Test 5: Video upload (if video exists)
    test_video_path = "data/vid/indianworkers.mp4"
    if Path(test_video_path).exists():
        print("\n⚠ Note: Video processing test will start a background job")
        response = input("Continue with video processing test? (y/n): ")
        if response.lower() == 'y':
            job_id = test_upload_video(test_video_path, base_url)
            if job_id:
                monitor_job(job_id, base_url, check_interval=10, max_checks=10)
    else:
        print(f"\n⚠ Skipping video processing test (no video at {test_video_path})")
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print(f"Using API base URL: {base_url}")
    
    run_all_tests(base_url)
