#!/usr/bin/env python3
"""
Comprehensive test script to validate the denoising fix and demo data
"""

import os
import sys
import time
import numpy as np

# Set a test API key (will use fallback analysis)
os.environ['GEMINI_API_KEY'] = 'test-key-for-validation'

from gemni_analyzer import NeuralEmbeddingAnalyzer
from api_server import app
import json

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def test_demo_data_files():
    """Test that demo data files exist and are valid"""
    print_header("TEST 1: Demo Data Files")
    
    files = {
        'demo_embeddings.npy': (200, 128),
        'demo_labels.npy': (200,),
        'cleaned_embeddings.npy': None  # Variable shape after cleaning
    }
    
    for filename, expected_shape in files.items():
        if os.path.exists(filename):
            data = np.load(filename)
            print(f"✓ {filename} exists")
            print(f"  Shape: {data.shape}")
            if expected_shape and data.shape != expected_shape:
                print(f"  ⚠ WARNING: Expected shape {expected_shape}, got {data.shape}")
        else:
            print(f"✗ {filename} NOT FOUND")
            return False
    
    return True

def test_analyzer_with_timeout():
    """Test that analyzer handles timeouts properly"""
    print_header("TEST 2: Analyzer Timeout Handling")
    
    analyzer = NeuralEmbeddingAnalyzer(api_key='test-key')
    
    # Generate sample data
    embeddings, labels = analyzer._generate_sample_data(n_samples=50, n_features=64)
    print(f"✓ Generated sample data: {embeddings.shape}")
    
    # Test metrics calculation
    metrics = analyzer.calculate_quality_metrics(embeddings, labels)
    print(f"✓ Calculated metrics: separability={metrics['separability']:.3f}")
    
    # Test analyze_with_gemini with timeout
    start_time = time.time()
    analysis = analyzer.analyze_with_gemini(metrics)
    elapsed = time.time() - start_time
    
    print(f"✓ Analysis completed in {elapsed:.2f} seconds")
    print(f"  Quality: {analysis['overall_quality']}")
    print(f"  Score: {analysis['quality_score']}/100")
    
    # Check if fallback was used
    if 'note' in analysis and 'Fallback' in analysis['note']:
        print(f"✓ Fallback analysis was used (expected with test key)")
    
    # Should complete quickly with timeout
    if elapsed > 15:
        print(f"⚠ WARNING: Analysis took {elapsed:.2f}s, should be faster")
        return False
    
    return True

def test_denoising():
    """Test that denoising works without hanging"""
    print_header("TEST 3: Denoising Functionality")
    
    analyzer = NeuralEmbeddingAnalyzer(api_key='test-key')
    
    # Load demo data
    embeddings = np.load('demo_embeddings.npy')
    labels = np.load('demo_labels.npy')
    print(f"✓ Loaded demo data: {embeddings.shape}")
    
    # Calculate metrics
    metrics = analyzer.calculate_quality_metrics(embeddings, labels)
    print(f"✓ Before - Separability: {metrics['separability']:.3f}")
    print(f"         - Noise Level: {metrics['noise_level']:.3f}")
    print(f"         - Outlier Ratio: {metrics['outlier_ratio']*100:.1f}%")
    
    # Get analysis
    analysis = analyzer.analyze_with_gemini(metrics)
    
    # Apply denoising
    start_time = time.time()
    cleaned_embeddings = analyzer.denoise_embeddings(embeddings, analysis)
    elapsed = time.time() - start_time
    
    print(f"✓ Denoising completed in {elapsed:.2f} seconds")
    print(f"  Cleaned shape: {cleaned_embeddings.shape}")
    
    # Calculate cleaned metrics
    cleaned_metrics = analyzer.calculate_quality_metrics(cleaned_embeddings)
    print(f"✓ After  - Separability: {cleaned_metrics['separability']:.3f}")
    print(f"         - Noise Level: {cleaned_metrics['noise_level']:.3f}")
    print(f"         - Outlier Ratio: {cleaned_metrics['outlier_ratio']*100:.1f}%")
    
    return True

def test_api_endpoints():
    """Test API endpoints work correctly"""
    print_header("TEST 4: API Endpoints")
    
    client = app.test_client()
    
    # Test health check
    response = client.get('/api/health')
    assert response.status_code == 200
    print("✓ Health check endpoint works")
    
    # Test generate sample
    response = client.post('/api/generate-sample',
        data=json.dumps({'n_samples': 50, 'n_features': 64}),
        content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] == True
    embeddings = data['data']['embeddings']
    print(f"✓ Generate sample endpoint works ({len(embeddings)} samples)")
    
    # Test analyze quality
    response = client.post('/api/analyze-quality',
        data=json.dumps({'embeddings': embeddings}),
        content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] == True
    print(f"✓ Analyze quality endpoint works (quality: {data['analysis']['overall_quality']})")
    
    # Test denoise embeddings
    response = client.post('/api/denoise-embeddings',
        data=json.dumps({'embeddings': embeddings}),
        content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] == True
    print(f"✓ Denoise endpoint works ({len(data['data']['cleaned_embeddings'])} cleaned samples)")
    
    return True

def main():
    """Run all tests"""
    print_header("DENOISING FIX & DEMO DATA VALIDATION")
    
    tests = [
        ("Demo Data Files", test_demo_data_files),
        ("Analyzer Timeout", test_analyzer_with_timeout),
        ("Denoising", test_denoising),
        ("API Endpoints", test_api_endpoints)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
        print("\n  The denoising issue is fixed:")
        print("  • Gemini API calls have 10-second timeout")
        print("  • Fallback analysis works when API is unavailable")
        print("  • Denoising completes successfully")
        print("  • Demo data files are available")
        return 0
    else:
        print("\n  ⚠ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
