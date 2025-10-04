#!/usr/bin/env python3
"""
Simple test script to verify the Gemini analyzer works independently
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemni_analyzer import NeuralEmbeddingAnalyzer
from dotenv import load_dotenv

def test_analyzer():
    print("🧪 Testing Gemini Neural Embedding Analyzer...")
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        return False
    
    try:
        # Initialize analyzer
        print("📡 Initializing Gemini API connection...")
        analyzer = NeuralEmbeddingAnalyzer(api_key=api_key)
        
        # Generate sample data
        print("📊 Generating sample neural embeddings...")
        embeddings, labels = analyzer._generate_sample_data(n_samples=50, n_features=64)
        print(f"✓ Generated embeddings shape: {embeddings.shape}")
        
        # Calculate metrics
        print("📈 Calculating quality metrics...")
        metrics = analyzer.calculate_quality_metrics(embeddings, labels)
        print(f"✓ Metrics calculated:")
        print(f"  - Separability: {metrics['separability']:.3f}")
        print(f"  - Noise Level: {metrics['noise_level']:.3f}")
        print(f"  - Outlier Ratio: {metrics['outlier_ratio']:.1%}")
        
        # Test Gemini analysis
        print("🤖 Testing Gemini AI analysis...")
        analysis = analyzer.analyze_with_gemini(metrics)
        print(f"✓ Gemini analysis completed:")
        print(f"  - Quality Score: {analysis.get('quality_score', 'N/A')}")
        print(f"  - Overall Quality: {analysis.get('overall_quality', 'N/A')}")
        
        print("\n🎉 All tests passed! Your Gemini analyzer is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_analyzer()
    sys.exit(0 if success else 1)
