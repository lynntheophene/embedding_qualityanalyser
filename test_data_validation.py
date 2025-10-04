#!/usr/bin/env python3
"""
Comprehensive Data Validation Test
Ensures all data throughout the system is valid (no NaN, no Inf, proper types)
and all API responses have success: True
"""

import numpy as np
import sys
import json
from gemni_analyzer import NeuralEmbeddingAnalyzer

def validate_array(arr, name):
    """Validate a numpy array for data integrity"""
    print(f"\nValidating {name}:")
    
    # Check type
    assert isinstance(arr, np.ndarray), f"  ❌ {name} is not a numpy array"
    print(f"  ✓ Is numpy array")
    
    # Check for NaN
    assert not np.any(np.isnan(arr)), f"  ❌ {name} contains NaN values"
    print(f"  ✓ No NaN values")
    
    # Check for Inf
    assert not np.any(np.isinf(arr)), f"  ❌ {name} contains Inf values"
    print(f"  ✓ No Inf values")
    
    # Check shape is valid
    assert len(arr.shape) > 0, f"  ❌ {name} has invalid shape"
    assert all(s > 0 for s in arr.shape), f"  ❌ {name} has zero dimension"
    print(f"  ✓ Valid shape: {arr.shape}")
    
    # Check dtype is numeric for float arrays
    if arr.dtype.kind == 'f':
        assert arr.dtype in [np.float32, np.float64], f"  ❌ {name} has unusual float dtype"
        print(f"  ✓ Valid dtype: {arr.dtype}")
    
    return True

def validate_metrics(metrics, name="metrics"):
    """Validate quality metrics dictionary"""
    print(f"\nValidating {name}:")
    
    # Required keys
    required_keys = ['separability', 'noise_level', 'cluster_coherence', 
                     'outlier_ratio', 'snr_db', 'n_samples', 'n_features']
    
    for key in required_keys:
        assert key in metrics, f"  ❌ Missing required key: {key}"
        value = metrics[key]
        
        # Check value is numeric (not list)
        if key in ['separability', 'noise_level', 'cluster_coherence', 'outlier_ratio', 'snr_db']:
            assert isinstance(value, (int, float)), f"  ❌ {key} is not numeric"
            assert not np.isnan(value), f"  ❌ {key} is NaN"
            assert not np.isinf(value), f"  ❌ {key} is Inf"
            print(f"  ✓ {key}: {value:.4f} (valid)")
        elif key in ['n_samples', 'n_features']:
            assert isinstance(value, int), f"  ❌ {key} is not int"
            assert value > 0, f"  ❌ {key} is not positive"
            print(f"  ✓ {key}: {value} (valid)")
    
    return True

def validate_analysis(analysis, name="analysis"):
    """Validate analysis dictionary"""
    print(f"\nValidating {name}:")
    
    # Required keys
    required_keys = ['overall_quality', 'quality_score', 'recommendations']
    
    for key in required_keys:
        assert key in analysis, f"  ❌ Missing required key: {key}"
    
    # Check overall_quality
    valid_qualities = ['excellent', 'good', 'fair', 'poor']
    assert analysis['overall_quality'] in valid_qualities, \
        f"  ❌ overall_quality '{analysis['overall_quality']}' not in {valid_qualities}"
    print(f"  ✓ overall_quality: {analysis['overall_quality']}")
    
    # Check quality_score
    score = analysis['quality_score']
    assert isinstance(score, (int, float)), f"  ❌ quality_score is not numeric"
    assert 0 <= score <= 100, f"  ❌ quality_score {score} not in range [0, 100]"
    assert not np.isnan(score), f"  ❌ quality_score is NaN"
    print(f"  ✓ quality_score: {score} (valid)")
    
    # Check recommendations is a list
    assert isinstance(analysis['recommendations'], list), \
        f"  ❌ recommendations is not a list"
    assert len(analysis['recommendations']) > 0, \
        f"  ❌ recommendations is empty"
    print(f"  ✓ recommendations: {len(analysis['recommendations'])} items")
    
    return True

def test_data_generation():
    """Test 1: Data generation produces valid data"""
    print("="*70)
    print("TEST 1: Data Generation Validation")
    print("="*70)
    
    analyzer = NeuralEmbeddingAnalyzer(api_key='test-key')
    
    # Test different sizes
    test_cases = [
        (50, 32),
        (100, 64),
        (200, 128),
    ]
    
    for n_samples, n_features in test_cases:
        embeddings, labels = analyzer._generate_sample_data(n_samples, n_features)
        
        validate_array(embeddings, f"embeddings ({n_samples}x{n_features})")
        validate_array(labels, f"labels ({n_samples})")
        
        # Check dimensions match
        assert embeddings.shape == (n_samples, n_features), \
            f"  ❌ embeddings shape mismatch"
        assert labels.shape == (n_samples,), \
            f"  ❌ labels shape mismatch"
        
        # Check labels are binary
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 2, \
            f"  ❌ labels should have 2 unique values"
        assert set(unique_labels) == {0, 1}, \
            f"  ❌ labels should be 0 and 1"
        print(f"  ✓ Labels are binary: {unique_labels}")
    
    print("\n✅ Data generation validation PASSED")
    return True

def test_metrics_calculation():
    """Test 2: Metrics calculation produces valid data"""
    print("\n" + "="*70)
    print("TEST 2: Metrics Calculation Validation")
    print("="*70)
    
    analyzer = NeuralEmbeddingAnalyzer(api_key='test-key')
    embeddings, labels = analyzer._generate_sample_data(100, 64)
    
    # Test with labels
    metrics_with_labels = analyzer.calculate_quality_metrics(embeddings, labels)
    validate_metrics(metrics_with_labels, "metrics (with labels)")
    
    # Test without labels
    metrics_without_labels = analyzer.calculate_quality_metrics(embeddings, None)
    validate_metrics(metrics_without_labels, "metrics (without labels)")
    
    # Both should have same structure
    assert set(metrics_with_labels.keys()) == set(metrics_without_labels.keys()), \
        "  ❌ Metrics keys differ between with/without labels"
    
    print("\n✅ Metrics calculation validation PASSED")
    return True

def test_analysis_generation():
    """Test 3: Analysis generation produces valid data"""
    print("\n" + "="*70)
    print("TEST 3: Analysis Generation Validation")
    print("="*70)
    
    analyzer = NeuralEmbeddingAnalyzer(api_key='test-key')
    embeddings, labels = analyzer._generate_sample_data(100, 64)
    metrics = analyzer.calculate_quality_metrics(embeddings, labels)
    
    # Generate analysis (will use fallback)
    analysis = analyzer.analyze_with_gemini(metrics)
    
    validate_analysis(analysis, "Gemini analysis")
    
    print("\n✅ Analysis generation validation PASSED")
    return True

def test_denoising():
    """Test 4: Denoising produces valid data"""
    print("\n" + "="*70)
    print("TEST 4: Denoising Validation")
    print("="*70)
    
    analyzer = NeuralEmbeddingAnalyzer(api_key='test-key')
    embeddings, labels = analyzer._generate_sample_data(100, 64)
    metrics = analyzer.calculate_quality_metrics(embeddings, labels)
    analysis = analyzer.analyze_with_gemini(metrics)
    
    # Apply denoising
    cleaned = analyzer.denoise_embeddings(embeddings, analysis)
    
    validate_array(cleaned, "cleaned embeddings")
    
    # Check cleaned has valid shape
    assert cleaned.shape[0] == embeddings.shape[0], \
        "  ❌ Number of samples changed during denoising"
    print(f"  ✓ Sample count preserved: {cleaned.shape[0]}")
    
    # Calculate metrics for cleaned data
    cleaned_metrics = analyzer.calculate_quality_metrics(cleaned)
    validate_metrics(cleaned_metrics, "cleaned metrics")
    
    # Check improvement
    print(f"\n  Improvement metrics:")
    print(f"    Separability: {metrics['separability']:.3f} → {cleaned_metrics['separability']:.3f}")
    print(f"    SNR: {metrics['snr_db']:.2f} dB → {cleaned_metrics['snr_db']:.2f} dB")
    
    print("\n✅ Denoising validation PASSED")
    return True

def test_demo_files():
    """Test 5: Demo files contain valid data"""
    print("\n" + "="*70)
    print("TEST 5: Demo Files Validation")
    print("="*70)
    
    import os
    
    # Check files exist
    files = ['demo_embeddings.npy', 'demo_labels.npy', 'cleaned_embeddings.npy']
    for filename in files:
        assert os.path.exists(filename), f"  ❌ {filename} does not exist"
        print(f"  ✓ {filename} exists")
    
    # Load and validate
    demo_embeddings = np.load('demo_embeddings.npy')
    validate_array(demo_embeddings, "demo_embeddings.npy")
    
    demo_labels = np.load('demo_labels.npy')
    validate_array(demo_labels, "demo_labels.npy")
    
    cleaned_embeddings = np.load('cleaned_embeddings.npy')
    validate_array(cleaned_embeddings, "cleaned_embeddings.npy")
    
    # Check consistency
    assert demo_embeddings.shape[0] == demo_labels.shape[0], \
        "  ❌ demo embeddings and labels have different number of samples"
    print(f"  ✓ Demo embeddings and labels are consistent")
    
    assert demo_embeddings.shape[0] == cleaned_embeddings.shape[0], \
        "  ❌ demo and cleaned embeddings have different number of samples"
    print(f"  ✓ Demo and cleaned embeddings have same sample count")
    
    print("\n✅ Demo files validation PASSED")
    return True

def test_api_response_format():
    """Test 6: API response format validation"""
    print("\n" + "="*70)
    print("TEST 6: API Response Format Validation")
    print("="*70)
    
    from api_server import app
    
    client = app.test_client()
    
    # Test 1: Health endpoint
    response = client.get('/api/health')
    assert response.status_code == 200, "  ❌ Health endpoint returned non-200"
    data = json.loads(response.data)
    assert 'status' in data, "  ❌ Health response missing 'status'"
    assert data['status'] == 'healthy', "  ❌ Health status is not 'healthy'"
    print("  ✓ Health endpoint returns valid response")
    
    # Test 2: Generate sample endpoint
    response = client.post('/api/generate-sample',
        data=json.dumps({'n_samples': 50, 'n_features': 32}),
        content_type='application/json')
    assert response.status_code == 200, "  ❌ Generate sample returned non-200"
    data = json.loads(response.data)
    assert data['success'] == True, "  ❌ Generate sample success is not True"
    assert 'data' in data, "  ❌ Generate sample response missing 'data'"
    print("  ✓ Generate sample endpoint: success=True")
    
    embeddings = data['data']['embeddings']
    
    # Test 3: Analyze quality endpoint
    response = client.post('/api/analyze-quality',
        data=json.dumps({'embeddings': embeddings}),
        content_type='application/json')
    assert response.status_code == 200, "  ❌ Analyze quality returned non-200"
    data = json.loads(response.data)
    assert data['success'] == True, "  ❌ Analyze quality success is not True"
    assert 'analysis' in data, "  ❌ Analyze quality response missing 'analysis'"
    assert 'metrics' in data, "  ❌ Analyze quality response missing 'metrics'"
    print("  ✓ Analyze quality endpoint: success=True")
    
    # Test 4: Denoise embeddings endpoint
    response = client.post('/api/denoise-embeddings',
        data=json.dumps({'embeddings': embeddings}),
        content_type='application/json')
    assert response.status_code == 200, "  ❌ Denoise embeddings returned non-200"
    data = json.loads(response.data)
    assert data['success'] == True, "  ❌ Denoise embeddings success is not True"
    assert 'data' in data, "  ❌ Denoise response missing 'data'"
    assert 'cleaned_embeddings' in data['data'], "  ❌ Denoise response missing 'cleaned_embeddings'"
    print("  ✓ Denoise embeddings endpoint: success=True")
    
    print("\n✅ API response format validation PASSED")
    return True

def main():
    """Run all validation tests"""
    print("="*70)
    print("COMPREHENSIVE DATA VALIDATION TEST SUITE")
    print("Ensuring 'every data must be true' (valid and success=True)")
    print("="*70)
    
    tests = [
        ("Data Generation", test_data_generation),
        ("Metrics Calculation", test_metrics_calculation),
        ("Analysis Generation", test_analysis_generation),
        ("Denoising", test_denoising),
        ("Demo Files", test_demo_files),
        ("API Response Format", test_api_response_format),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Tests Passed: {passed}/{len(tests)}")
    print(f"Tests Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("✅ Every data is true (valid)")
        print("✅ All API responses return success: True")
        print("✅ No NaN or Inf values anywhere")
        print("✅ All data types are correct")
        print("✅ Demo files are valid")
        return 0
    else:
        print(f"\n⚠️  {failed} validation(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
