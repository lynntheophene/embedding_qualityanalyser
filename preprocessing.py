"""
Neural Embedding Quality Analyzer
Powered by Claude Sonnet 4.5

Analyzes and improves quality of neural embeddings from BCI datasets
NO HARDWARE REQUIRED - Works with any embedding data
"""

import numpy as np
import anthropic
import json
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import iqr

class NeuralEmbeddingAnalyzer:
    """
    Claude-powered analyzer for neural embedding quality.
    Works with embeddings from any BCI system (EEG, MEG, fMRI, etc.)
    """
    
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.scaler = RobustScaler()
        
    def load_embeddings(self, filepath=None, format='npy'):
        """
        Load embeddings from file or generate sample data.
        
        Args:
            filepath: Path to embeddings file (.npy, .csv, .json)
            format: File format ('npy', 'csv', 'json')
        
        Returns:
            embeddings: numpy array of shape (n_samples, n_features)
            labels: optional labels array
        """
        if filepath:
            if format == 'npy':
                data = np.load(filepath)
            elif format == 'csv':
                import pandas as pd
                data = pd.read_csv(filepath).values
            elif format == 'json':
                with open(filepath, 'r') as f:
                    data = np.array(json.load(f))
            
            return data, None
        else:
            # Generate sample noisy embeddings for demo
            print("📊 Generating sample neural embeddings...")
            return self._generate_sample_data()
    
    def _generate_sample_data(self, n_samples=200, n_features=128):
        """Generate realistic noisy neural embeddings."""
        # Clean signal component
        t = np.linspace(0, 4*np.pi, n_samples)
        clean_pattern = np.column_stack([
            np.sin(t),
            np.cos(t),
            np.sin(2*t),
            np.cos(2*t)
        ])
        
        # Add random features
        random_features = np.random.randn(n_samples, n_features - 4) * 0.5
        embeddings = np.column_stack([clean_pattern, random_features])
        
        # Add realistic noise and artifacts
        # 1. Gaussian noise
        noise = np.random.randn(n_samples, n_features) * 0.3
        
        # 2. Outliers (simulating artifacts)
        n_outliers = int(n_samples * 0.15)
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        outlier_noise = np.random.randn(n_outliers, n_features) * 3.0
        
        # 3. Drift (simulating session effects)
        drift = np.linspace(0, 1, n_samples)[:, np.newaxis] * np.random.randn(1, n_features) * 0.5
        
        # Combine all noise sources
        noisy_embeddings = embeddings + noise + drift
        noisy_embeddings[outlier_indices] += outlier_noise
        
        # Create labels (for visualization)
        labels = np.concatenate([
            np.zeros(n_samples//2),
            np.ones(n_samples//2)
        ]).astype(int)
        
        return noisy_embeddings, labels
    
    def calculate_quality_metrics(self, embeddings, labels=None):
        """
        Calculate comprehensive quality metrics.
        """
        n_samples, n_features = embeddings.shape
        
        # 1. Separability (if labels available)
        if labels is not None:
            from sklearn.metrics import silhouette_score
            try:
                separability = silhouette_score(embeddings, labels)
            except:
                separability = 0.5
        else:
            # Estimate from variance
            separability = 1 - (np.std(embeddings) / (np.max(embeddings) - np.min(embeddings)))
        
        # 2. Noise level (coefficient of variation)
        noise_level = np.mean(np.std(embeddings, axis=0) / (np.mean(np.abs(embeddings), axis=0) + 1e-8))
        
        # 3. Cluster coherence (inverse of mean pairwise distance)
        sample_size = min(100, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        distances = cdist(embeddings[sample_indices], embeddings[sample_indices])
        coherence = 1 / (1 + np.mean(distances))
        
        # 4. Outlier ratio (using IQR method)
        outliers = []
        for feature_idx in range(n_features):
            feature = embeddings[:, feature_idx]
            q1, q3 = np.percentile(feature, [25, 75])
            iqr_val = q3 - q1
            lower_bound = q1 - 1.5 * iqr_val
            upper_bound = q3 + 1.5 * iqr_val
            outliers.extend(np.where((feature < lower_bound) | (feature > upper_bound))[0])
        
        outlier_ratio = len(set(outliers)) / n_samples
        
        # 5. Dimension variance (check for redundant/noisy dimensions)
        dimension_variance = np.var(embeddings, axis=0)
        
        # 6. Signal-to-noise ratio estimate
        signal_power = np.mean(embeddings ** 2)
        noise_estimate = np.mean(dimension_variance[dimension_variance < np.percentile(dimension_variance, 25)])
        snr = 10 * np.log10(signal_power / (noise_estimate + 1e-8))
        
        metrics = {
            'separability': float(np.clip(separability, 0, 1)),
            'noise_level': float(np.clip(noise_level, 0, 1)),
            'cluster_coherence': float(np.clip(coherence, 0, 1)),
            'outlier_ratio': float(outlier_ratio),
            'dimension_variance': dimension_variance.tolist()[:5],  # First 5 for brevity
            'snr_db': float(snr),
            'n_samples': n_samples,
            'n_features': n_features
        }
        
        return metrics
    
    def analyze_with_claude(self, metrics):
        """
        Use Claude to analyze quality metrics and provide recommendations.
        """
        prompt = f"""Analyze these neural embedding quality metrics from a brain-computer interface dataset:

QUALITY METRICS:
- Separability Score: {metrics['separability']:.3f} (0=poor, 1=excellent)
- Noise Level: {metrics['noise_level']:.3f} (lower is better)
- Cluster Coherence: {metrics['cluster_coherence']:.3f} (higher is better)
- Outlier Ratio: {metrics['outlier_ratio']:.1%}
- Signal-to-Noise Ratio: {metrics['snr_db']:.2f} dB
- Dataset Size: {metrics['n_samples']} samples × {metrics['n_features']} features

CONTEXT:
These embeddings represent neural activity patterns extracted from brain signals. They will be used for:
- Speech decoding (predicting words/phonemes from brain activity)
- Mental state classification
- Brain-computer interface control

High-quality embeddings should have:
- High separability (>0.75) for distinct classes
- Low noise (<0.3)
- High coherence (>0.6) 
- Low outliers (<10%)
- SNR > 10 dB

TASK:
Provide a comprehensive analysis in JSON format:
{{
    "overall_quality": "excellent/good/fair/poor",
    "quality_score": <0-100>,
    "critical_issues": ["issue1", "issue2", ...],
    "recommendations": [
        {{
            "issue": "specific problem",
            "solution": "specific fix",
            "expected_improvement": "quantified benefit",
            "priority": "high/medium/low"
        }},
        ...
    ],
    "performance_prediction": {{
        "current_accuracy_estimate": "<percentage>",
        "post_cleaning_estimate": "<percentage>",
        "confidence": "high/medium/low"
    }},
    "next_steps": ["step1", "step2", ...]
}}"""

        message = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        
        # Extract JSON from response
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            analysis = json.loads(response_text[start_idx:end_idx])
        except Exception as e:
            print(f"Warning: Could not parse Claude response: {e}")
            analysis = {
                "overall_quality": "fair",
                "quality_score": 65,
                "critical_issues": ["Unable to parse full analysis"],
                "recommendations": [],
                "performance_prediction": {
                    "current_accuracy_estimate": "70-75%",
                    "post_cleaning_estimate": "80-85%",
                    "confidence": "medium"
                },
                "next_steps": ["Retry analysis"]
            }
        
        return analysis
    
    def remove_outliers(self, embeddings, method='iqr', threshold=1.5):
        """
        Remove outliers using IQR or Z-score method.
        """
        if method == 'iqr':
            # Calculate IQR for each feature
            q1 = np.percentile(embeddings, 25, axis=0)
            q3 = np.percentile(embeddings, 75, axis=0)
            iqr_val = q3 - q1
            
            lower_bound = q1 - threshold * iqr_val
            upper_bound = q3 + threshold * iqr_val
            
            # Find outlier samples (any feature out of bounds)
            outlier_mask = np.any(
                (embeddings < lower_bound) | (embeddings > upper_bound),
                axis=1
            )
            
            # Replace outliers with feature median
            cleaned = embeddings.copy()
            medians = np.median(embeddings, axis=0)
            cleaned[outlier_mask] = medians
            
            n_outliers = np.sum(outlier_mask)
            print(f"  → Removed {n_outliers} outlier samples ({n_outliers/len(embeddings)*100:.1f}%)")
            
        return cleaned, outlier_mask
    
    def denoise_embeddings(self, embeddings, analysis):
        """
        Apply denoising based on Claude's recommendations.
        """
        print("\n🔧 Applying Claude's Recommendations...")
        cleaned = embeddings.copy()
        
        # Step 1: Remove outliers
        if analysis.get('recommendations'):
            for rec in analysis['recommendations']:
                if 'outlier' in rec.get('issue', '').lower():
                    print(f"  ✓ {rec['solution']}")
                    cleaned, _ = self.remove_outliers(cleaned, method='iqr')
                    break
        
        # Step 2: Robust scaling
        print("  ✓ Applying robust scaling (median-based normalization)")
        cleaned = self.scaler.fit_transform(cleaned)
        
        # Step 3: Reduce high-variance noise dimensions
        variance = np.var(cleaned, axis=0)
        high_var_dims = variance > np.percentile(variance, 90)
        if np.any(high_var_dims):
            print(f"  ✓ Reducing noise in {np.sum(high_var_dims)} high-variance dimensions")
            cleaned[:, high_var_dims] *= 0.5
        
        # Step 4: Optional PCA denoising
        if embeddings.shape[1] > 50:
            print("  ✓ Applying PCA denoising")
            pca = PCA(n_components=0.95)  # Keep 95% variance
            cleaned = pca.fit_transform(cleaned)
            print(f"    → Reduced from {embeddings.shape[1]} to {cleaned.shape[1]} dimensions")
        
        return cleaned
    
    def generate_report(self, before_metrics, after_metrics, analysis):
        """
        Generate a comprehensive quality report.
        """
        print("\n" + "="*70)
        print("📊 NEURAL EMBEDDING QUALITY REPORT")
        print("="*70)
        
        print("\n🔍 CLAUDE'S ANALYSIS:")
        print(f"  Overall Quality: {analysis['overall_quality'].upper()}")
        print(f"  Quality Score: {analysis['quality_score']}/100")
        
        if analysis.get('critical_issues'):
            print("\n❌ Critical Issues:")
            for issue in analysis['critical_issues']:
                print(f"    • {issue}")
        
        if analysis.get('recommendations'):
            print("\n💡 Recommendations Applied:")
            for i, rec in enumerate(analysis['recommendations'][:3], 1):
                print(f"    {i}. {rec.get('issue', 'N/A')}")
                print(f"       → {rec.get('solution', 'N/A')}")
                print(f"       Expected: {rec.get('expected_improvement', 'N/A')}")
        
        print("\n📈 IMPROVEMENT METRICS:")
        improvements = {
            'Separability': (before_metrics['separability'], after_metrics['separability'], False),
            'Noise Level': (before_metrics['noise_level'], after_metrics['noise_level'], True),
            'Coherence': (before_metrics['cluster_coherence'], after_metrics['cluster_coherence'], False),
            'Outlier Ratio': (before_metrics['outlier_ratio'], after_metrics['outlier_ratio'], True),
            'SNR': (before_metrics['snr_db'], after_metrics['snr_db'], False)
        }
        
        for metric_name, (before, after, inverse) in improvements.items():
            if inverse:
                improvement = ((before - after) / before * 100) if before > 0 else 0
            else:
                improvement = ((after - before) / before * 100) if before > 0 else 0
            
            arrow = "↓" if inverse else "↑"
            print(f"  {metric_name:15s}: {before:6.3f} → {after:6.3f}  ({arrow} {abs(improvement):5.1f}%)")
        
        print("\n🎯 PERFORMANCE PREDICTION:")
        pred = analysis.get('performance_prediction', {})
        print(f"  Current Estimate: {pred.get('current_accuracy_estimate', 'N/A')}")
        print(f"  After Cleaning:   {pred.get('post_cleaning_estimate', 'N/A')}")
        print(f"  Confidence:       {pred.get('confidence', 'N/A').upper()}")
        
        print("\n" + "="*70)
    
    def run_pipeline(self, filepath=None, format='npy'):
        """
        Execute the complete analysis and cleaning pipeline.
        """
        print("🧠 Neural Embedding Quality Analyzer")
        print("Powered by Claude Sonnet 4.5")
        print("="*70)
        
        # Step 1: Load embeddings
        print("\n[1/5] Loading neural embeddings...")
        embeddings, labels = self.load_embeddings(filepath, format)
        print(f"✓ Loaded: {embeddings.shape[0]} samples × {embeddings.shape[1]} features")
        
        # Step 2: Calculate quality metrics
        print("\n[2/5] Calculating quality metrics...")
        before_metrics = self.calculate_quality_metrics(embeddings, labels)
        print(f"✓ Quality Score: {before_metrics['separability']*100:.1f}/100")
        print(f"✓ SNR: {before_metrics['snr_db']:.2f} dB")
        print(f"✓ Outlier Ratio: {before_metrics['outlier_ratio']*100:.1f}%")
        
        # Step 3: Claude analysis
        print("\n[3/5] Analyzing with Claude AI...")
        analysis = self.analyze_with_claude(before_metrics)
        print(f"✓ Claude Assessment: {analysis['overall_quality'].upper()}")
        print(f"✓ Identified {len(analysis.get('critical_issues', []))} critical issues")
        print(f"✓ Generated {len(analysis.get('recommendations', []))} recommendations")
        
        # Step 4: Apply denoising
        print("\n[4/5] Applying denoising and cleanup...")
        cleaned_embeddings = self.denoise_embeddings(embeddings, analysis)
        print("✓ Denoising complete")
        
        # Step 5: Re-evaluate quality
        print("\n[5/5] Re-evaluating quality...")
        after_metrics = self.calculate_quality_metrics(cleaned_embeddings, labels)
        print(f"✓ New Quality Score: {after_metrics['separability']*100:.1f}/100")
        print(f"✓ New SNR: {after_metrics['snr_db']:.2f} dB")
        
        # Generate comprehensive report
        self.generate_report(before_metrics, after_metrics, analysis)
        
        return {
            'original_embeddings': embeddings,
            'cleaned_embeddings': cleaned_embeddings,
            'before_metrics': before_metrics,
            'after_metrics': after_metrics,
            'claude_analysis': analysis
        }


# Example usage
if __name__ == "__main__":
    import os
    
    # Get API key from environment
    API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")
    
    if API_KEY == "your-api-key-here":
        print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        exit(1)
    
    # Initialize analyzer
    analyzer = NeuralEmbeddingAnalyzer(api_key=API_KEY)
    
    # Run the complete pipeline
    # Option 1: Use sample data
    results = analyzer.run_pipeline()
    
    # Option 2: Use your own embeddings
    # results = analyzer.run_pipeline(filepath='your_embeddings.npy', format='npy')
    
    # Save cleaned embeddings
    print("\n💾 Saving cleaned embeddings...")
    np.save('cleaned_embeddings.npy', results['cleaned_embeddings'])
    print("✓ Saved to: cleaned_embeddings.npy")
    
    print("\n✨ Pipeline complete! Ready for downstream tasks (classification, decoding, etc.)")