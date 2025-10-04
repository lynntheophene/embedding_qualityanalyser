#!/usr/bin/env python3
"""
Test script for Neural Embedding Quality Analyzer
Tests core functionality without requiring Gemini API key
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from gemni_analyzer import NeuralEmbeddingAnalyzer

def test_sample_data_generation():
    """Test sample data generation"""
    print("Testing sample data generation...")
    
    # Create a mock analyzer with fake API key
    class MockAnalyzer:
        def __init__(self):
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
    
    analyzer = MockAnalyzer()
    
    # Import and test the _generate_sample_data method directly
    from gemni_analyzer import NeuralEmbeddingAnalyzer
    embeddings, labels = NeuralEmbeddingAnalyzer._generate_sample_data(None, n_samples=100, n_features=64)
    
    assert embeddings.shape == (100, 64), f"Expected shape (100, 64), got {embeddings.shape}"
    assert labels.shape == (100,), f"Expected labels shape (100,), got {labels.shape}"
    assert len(np.unique(labels)) == 2, f"Expected 2 unique labels, got {len(np.unique(labels))}"
    
    print("✓ Sample data generation works")
    return embeddings, labels

def test_quality_metrics():
    """Test quality metrics calculation"""
    print("\nTesting quality metrics calculation...")
    
    from gemni_analyzer import NeuralEmbeddingAnalyzer
    
    # Generate sample data
    embeddings, labels = NeuralEmbeddingAnalyzer._generate_sample_data(None, n_samples=100, n_features=64)
    
    # Create a mock analyzer
    class MockAnalyzer:
        def __init__(self):
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
    
    analyzer = MockAnalyzer()
    
    # Test calculate_quality_metrics
    metrics = NeuralEmbeddingAnalyzer.calculate_quality_metrics(analyzer, embeddings, labels)
    
    assert 'separability' in metrics
    assert 'noise_level' in metrics
    assert 'cluster_coherence' in metrics
    assert 'outlier_ratio' in metrics
    assert 'snr_db' in metrics
    assert 'n_samples' in metrics
    assert 'n_features' in metrics
    
    assert 0 <= metrics['separability'] <= 1
    assert 0 <= metrics['noise_level'] <= 1
    assert 0 <= metrics['cluster_coherence'] <= 1
    assert 0 <= metrics['outlier_ratio'] <= 1
    assert metrics['n_samples'] == 100
    assert metrics['n_features'] == 64
    
    print("✓ Quality metrics calculation works")
    print(f"  - Separability: {metrics['separability']:.3f}")
    print(f"  - Noise Level: {metrics['noise_level']:.3f}")
    print(f"  - SNR: {metrics['snr_db']:.2f} dB")
    
    return metrics

def test_outlier_removal():
    """Test outlier removal"""
    print("\nTesting outlier removal...")
    
    from gemni_analyzer import NeuralEmbeddingAnalyzer
    
    # Generate sample data
    embeddings, _ = NeuralEmbeddingAnalyzer._generate_sample_data(None, n_samples=100, n_features=64)
    
    # Create a mock analyzer
    class MockAnalyzer:
        def __init__(self):
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
    
    analyzer = MockAnalyzer()
    
    # Test remove_outliers
    cleaned, outlier_mask = NeuralEmbeddingAnalyzer.remove_outliers(analyzer, embeddings)
    
    assert cleaned.shape == embeddings.shape
    assert outlier_mask.shape[0] == embeddings.shape[0]
    
    n_outliers = np.sum(outlier_mask)
    print(f"✓ Outlier removal works")
    print(f"  - Removed {n_outliers} outliers ({n_outliers/len(embeddings)*100:.1f}%)")
    
    return cleaned

def main():
    """Run all tests"""
    print("=" * 70)
    print("Neural Embedding Quality Analyzer - Test Suite")
    print("=" * 70)
    
    try:
        # Test 1: Sample data generation
        embeddings, labels = test_sample_data_generation()
        
        # Test 2: Quality metrics
        metrics = test_quality_metrics()
        
        # Test 3: Outlier removal
        cleaned = test_outlier_removal()
        
        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
        print("\nThe analyzer is working correctly!")
        print("To use with Gemini AI:")
        print("1. Get your API key from: https://makersuite.google.com/app/apikey")
        print("2. Add it to .env: GEMINI_API_KEY=your_key_here")
        print("3. Run: python api_server.py")
        print("4. Open: http://localhost:3000")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
