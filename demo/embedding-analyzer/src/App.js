import React, { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Brain, Upload, Wand2, Download, Activity, TrendingUp, AlertCircle, CheckCircle } from 'lucide-react';
import './App.css';

function App() {
  const [currentStep, setCurrentStep] = useState(0);
  const [embeddings, setEmbeddings] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [cleanedEmbeddings, setCleanedEmbeddings] = useState(null);
  const [cleanedMetrics, setCleanedMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const steps = [
    { id: 0, name: 'Upload', icon: Upload },
    { id: 1, name: 'Visualize', icon: Activity },
    { id: 2, name: 'Analyze', icon: Brain },
    { id: 3, name: 'Clean', icon: Wand2 }
  ];

  // Generate sample data
  const generateSampleData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/generate-sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_samples: 200, n_features: 128 })
      });
      
      const result = await response.json();
      if (result.success) {
        setEmbeddings(result.data.embeddings);
        setMetrics(result.data.metrics);
        setCurrentStep(1);
      } else {
        setError(result.error || 'Failed to generate sample data');
      }
    } catch (err) {
      setError('Backend connection failed. Make sure the Flask server is running on port 5000.');
    }
    setLoading(false);
  };

  // Analyze quality with Gemini
  const analyzeQuality = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/analyze-quality', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ embeddings })
      });
      
      const result = await response.json();
      if (result.success) {
        setAnalysis(result.analysis);
        setMetrics(result.metrics);
        setCurrentStep(2);
      } else {
        setError(result.error || 'Analysis failed');
      }
    } catch (err) {
      setError('Analysis failed: ' + err.message);
    }
    setLoading(false);
  };

  // Denoise embeddings
  const denoiseEmbeddings = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/denoise-embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ embeddings })
      });
      
      const result = await response.json();
      if (result.success) {
        setCleanedEmbeddings(result.data.cleaned_embeddings);
        setCleanedMetrics(result.data.cleaned_metrics);
        setCurrentStep(3);
      } else {
        setError(result.error || 'Denoising failed');
      }
    } catch (err) {
      setError('Denoising failed: ' + err.message);
    }
    setLoading(false);
  };

  // Render metric card
  const MetricCard = ({ title, value, unit, good, description }) => {
    const isGood = good === undefined ? true : good;
    return (
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">{title}</span>
          {isGood ? <CheckCircle size={16} className="icon-good" /> : <AlertCircle size={16} className="icon-warning" />}
        </div>
        <div className="metric-value">
          {typeof value === 'number' ? value.toFixed(3) : value}
          {unit && <span className="metric-unit">{unit}</span>}
        </div>
        {description && <div className="metric-description">{description}</div>}
      </div>
    );
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <Brain size={32} className="logo" />
          <h1>Neural Embedding Quality Analyzer</h1>
          <p>AI-Powered Analysis with Google Gemini</p>
        </div>
      </header>

      {/* Progress Steps */}
      <div className="steps-container">
        {steps.map((step, idx) => (
          <div key={step.id} className={`step ${currentStep >= idx ? 'active' : ''} ${currentStep === idx ? 'current' : ''}`}>
            <div className="step-icon">
              <step.icon size={20} />
            </div>
            <div className="step-name">{step.name}</div>
            {idx < steps.length - 1 && <div className="step-connector" />}
          </div>
        ))}
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-banner">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {/* Main Content */}
      <div className="content">
        
        {/* Step 0: Upload */}
        {currentStep === 0 && (
          <div className="card">
            <h2>📊 Load Neural Embeddings</h2>
            <p>Upload your neural embeddings or generate sample data for demo</p>
            <div className="button-group">
              <button 
                onClick={generateSampleData} 
                disabled={loading}
                className="button button-primary"
              >
                {loading ? 'Generating...' : 'Generate Sample Data'}
              </button>
              <button className="button button-secondary" disabled>
                Upload File (Coming Soon)
              </button>
            </div>
          </div>
        )}

        {/* Step 1: Visualize */}
        {currentStep >= 1 && embeddings && (
          <div className="card">
            <h2>📈 Embedding Visualization</h2>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="dim1" name="Dimension 1" />
                <YAxis type="number" dataKey="dim2" name="Dimension 2" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                <Scatter 
                  name="Original Embeddings" 
                  data={embeddings} 
                  fill="#8884d8" 
                />
                {cleanedEmbeddings && (
                  <Scatter 
                    name="Cleaned Embeddings" 
                    data={cleanedEmbeddings} 
                    fill="#82ca9d" 
                  />
                )}
              </ScatterChart>
            </ResponsiveContainer>

            {metrics && (
              <div className="metrics-grid">
                <MetricCard 
                  title="Separability" 
                  value={metrics.separability} 
                  good={metrics.separability > 0.75}
                  description="Class distinction quality"
                />
                <MetricCard 
                  title="Noise Level" 
                  value={metrics.noise_level} 
                  good={metrics.noise_level < 0.3}
                  description="Signal noise ratio"
                />
                <MetricCard 
                  title="Coherence" 
                  value={metrics.cluster_coherence} 
                  good={metrics.cluster_coherence > 0.6}
                  description="Data consistency"
                />
                <MetricCard 
                  title="Outlier Ratio" 
                  value={(metrics.outlier_ratio * 100).toFixed(1)} 
                  unit="%"
                  good={metrics.outlier_ratio < 0.1}
                  description="Artifact detection"
                />
                <MetricCard 
                  title="SNR" 
                  value={metrics.snr_db} 
                  unit=" dB"
                  good={metrics.snr_db > 10}
                  description="Signal-to-noise ratio"
                />
                <MetricCard 
                  title="Samples" 
                  value={metrics.n_samples} 
                  description="Total data points"
                />
              </div>
            )}

            {currentStep === 1 && (
              <div className="button-group">
                <button 
                  onClick={analyzeQuality} 
                  disabled={loading}
                  className="button button-primary"
                >
                  {loading ? 'Analyzing...' : '🤖 Analyze with Gemini AI'}
                </button>
              </div>
            )}
          </div>
        )}

        {/* Step 2: Analysis */}
        {currentStep >= 2 && analysis && (
          <div className="card">
            <h2>🤖 Gemini AI Analysis</h2>
            
            <div className="analysis-header">
              <div className="quality-badge" data-quality={analysis.overall_quality}>
                {analysis.overall_quality?.toUpperCase()}
              </div>
              <div className="quality-score">
                Score: {analysis.quality_score}/100
              </div>
            </div>

            {analysis.critical_issues && analysis.critical_issues.length > 0 && (
              <div className="issues-section">
                <h3>❌ Critical Issues</h3>
                <ul>
                  {analysis.critical_issues.map((issue, idx) => (
                    <li key={idx}>{issue}</li>
                  ))}
                </ul>
              </div>
            )}

            {analysis.recommendations && analysis.recommendations.length > 0 && (
              <div className="recommendations-section">
                <h3>💡 Recommendations</h3>
                {analysis.recommendations.map((rec, idx) => (
                  <div key={idx} className="recommendation-card">
                    <div className="rec-priority" data-priority={rec.priority}>
                      {rec.priority}
                    </div>
                    <div className="rec-content">
                      <div className="rec-issue"><strong>Issue:</strong> {rec.issue}</div>
                      <div className="rec-solution"><strong>Solution:</strong> {rec.solution}</div>
                      <div className="rec-improvement"><strong>Expected:</strong> {rec.expected_improvement}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {analysis.performance_prediction && (
              <div className="prediction-section">
                <h3>🎯 Performance Prediction</h3>
                <div className="prediction-grid">
                  <div className="prediction-item">
                    <span className="prediction-label">Current Estimate:</span>
                    <span className="prediction-value">{analysis.performance_prediction.current_accuracy_estimate}</span>
                  </div>
                  <div className="prediction-item">
                    <span className="prediction-label">After Cleaning:</span>
                    <span className="prediction-value highlight">{analysis.performance_prediction.post_cleaning_estimate}</span>
                  </div>
                  <div className="prediction-item">
                    <span className="prediction-label">Confidence:</span>
                    <span className="prediction-value">{analysis.performance_prediction.confidence}</span>
                  </div>
                </div>
              </div>
            )}

            {currentStep === 2 && (
              <div className="button-group">
                <button 
                  onClick={denoiseEmbeddings} 
                  disabled={loading}
                  className="button button-primary"
                >
                  {loading ? 'Cleaning...' : '🧹 Apply Denoising'}
                </button>
              </div>
            )}
          </div>
        )}

        {/* Step 3: Cleaned Results */}
        {currentStep >= 3 && cleanedMetrics && (
          <div className="card">
            <h2>✨ Cleaned Results</h2>
            <p>Embeddings have been cleaned based on Gemini's recommendations</p>
            
            <div className="metrics-grid">
              <MetricCard 
                title="Separability" 
                value={cleanedMetrics.separability} 
                good={cleanedMetrics.separability > 0.75}
                description={`Improved from ${metrics.separability.toFixed(3)}`}
              />
              <MetricCard 
                title="Noise Level" 
                value={cleanedMetrics.noise_level} 
                good={cleanedMetrics.noise_level < 0.3}
                description={`Reduced from ${metrics.noise_level.toFixed(3)}`}
              />
              <MetricCard 
                title="Coherence" 
                value={cleanedMetrics.cluster_coherence} 
                good={cleanedMetrics.cluster_coherence > 0.6}
                description={`Improved from ${metrics.cluster_coherence.toFixed(3)}`}
              />
              <MetricCard 
                title="Outlier Ratio" 
                value={(cleanedMetrics.outlier_ratio * 100).toFixed(1)} 
                unit="%"
                good={cleanedMetrics.outlier_ratio < 0.1}
                description={`Reduced from ${(metrics.outlier_ratio * 100).toFixed(1)}%`}
              />
            </div>

            <div className="button-group">
              <button 
                onClick={() => {
                  setCurrentStep(0);
                  setEmbeddings(null);
                  setMetrics(null);
                  setAnalysis(null);
                  setCleanedEmbeddings(null);
                  setCleanedMetrics(null);
                }}
                className="button button-secondary"
              >
                Start Over
              </button>
              <button className="button button-primary" disabled>
                <Download size={20} />
                Download Cleaned Data
              </button>
            </div>
          </div>
        )}
      </div>

      <footer className="footer">
        <p>Powered by Google Gemini AI • Neural Embedding Quality Analyzer</p>
      </footer>
    </div>
  );
}

export default App;
