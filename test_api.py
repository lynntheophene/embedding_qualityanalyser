#!/usr/bin/env python3
"""
API Endpoint Tests for Neural Embedding Quality Analyzer
Tests all backend endpoints without requiring actual Gemini API calls
"""

import sys
import json
import time
import requests
from multiprocessing import Process
import os
import signal

def start_server():
    """Start the Flask server in a subprocess"""
    # Suppress Flask output
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    from api_server import app
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def wait_for_server(url='http://localhost:5000/api/health', timeout=10):
    """Wait for server to be ready"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False

def test_health_endpoint():
    """Test /api/health endpoint"""
    print("Testing /api/health...", end=" ")
    response = requests.get('http://localhost:5000/api/health')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert data['backend'] == 'gemini-analyzer'
    print("✓")
    return True

def test_generate_sample():
    """Test /api/generate-sample endpoint"""
    print("Testing /api/generate-sample...", end=" ")
    response = requests.post(
        'http://localhost:5000/api/generate-sample',
        json={'n_samples': 50, 'n_features': 32}
    )
    assert response.status_code == 200
    data = response.json()
    assert data['success'] == True
    assert 'data' in data
    assert 'embeddings' in data['data']
    assert 'metrics' in data['data']
    assert len(data['data']['embeddings']) > 0
    print("✓")
    return data['data']

def test_analyze_quality(embeddings):
    """Test /api/analyze-quality endpoint"""
    print("Testing /api/analyze-quality...", end=" ")
    response = requests.post(
        'http://localhost:5000/api/analyze-quality',
        json={'embeddings': embeddings}
    )
    assert response.status_code == 200
    data = response.json()
    assert data['success'] == True
    assert 'analysis' in data
    assert 'metrics' in data
    print("✓")
    return data

def test_denoise_embeddings(embeddings):
    """Test /api/denoise-embeddings endpoint"""
    print("Testing /api/denoise-embeddings...", end=" ")
    response = requests.post(
        'http://localhost:5000/api/denoise-embeddings',
        json={'embeddings': embeddings}
    )
    assert response.status_code == 200
    data = response.json()
    assert data['success'] == True
    assert 'data' in data
    assert 'cleaned_embeddings' in data['data']
    assert 'cleaned_metrics' in data['data']
    print("✓")
    return data

def main():
    """Run all endpoint tests"""
    print("=" * 70)
    print("Neural Embedding Quality Analyzer - API Endpoint Tests")
    print("=" * 70)
    print()
    
    # Start server
    print("Starting Flask server...")
    server_process = Process(target=start_server)
    server_process.start()
    
    try:
        # Wait for server to be ready
        if not wait_for_server():
            print("❌ Server failed to start within 10 seconds")
            return 1
        
        print("✓ Server is ready\n")
        
        # Run tests
        print("Running endpoint tests:")
        print("-" * 70)
        
        # Test 1: Health check
        test_health_endpoint()
        
        # Test 2: Generate sample data
        sample_data = test_generate_sample()
        embeddings = sample_data['embeddings']
        
        # Test 3: Analyze quality
        analysis_result = test_analyze_quality(embeddings)
        
        # Test 4: Denoise embeddings
        denoise_result = test_denoise_embeddings(embeddings)
        
        print("-" * 70)
        print()
        print("=" * 70)
        print("✅ All API endpoint tests passed!")
        print("=" * 70)
        print()
        print("Summary:")
        print(f"  • Generated {len(embeddings)} sample embeddings")
        print(f"  • Quality score: {analysis_result['analysis']['quality_score']}/100")
        print(f"  • Overall quality: {analysis_result['analysis']['overall_quality']}")
        print(f"  • Cleaned {len(denoise_result['data']['cleaned_embeddings'])} embeddings")
        print()
        print("The backend API is fully functional! ✨")
        print()
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Stop server
        print("Stopping server...")
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()
        print("✓ Server stopped\n")

if __name__ == "__main__":
    sys.exit(main())
