# 🧠 Neural Embedding Quality Analyzer

A complete AI-powered system for analyzing and improving neural embedding quality from brain-computer interface (BCI) data, featuring **Google Gemini AI** integration and a **React-based web interface**.

![Demo](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![React](https://img.shields.io/badge/React-19.0-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Features

- 🤖 **AI-Powered Analysis**: Google Gemini integration for intelligent embedding quality assessment
- 📊 **Interactive Visualization**: Real-time scatter plots and quality metrics dashboard
- 🧹 **Smart Denoising**: Automated embedding cleanup based on AI recommendations
- 📈 **Quality Metrics**: Comprehensive analysis including separability, SNR, coherence, and outlier detection
- 🎨 **Modern UI**: Clean, responsive React interface with multi-step workflow
- ⚡ **Fast & Free**: Uses Google Gemini's free API tier

## 🏗️ System Architecture

### Backend (Python + Flask)
- **Location**: `api_server.py`
- **Port**: `http://localhost:5000`
- RESTful API with CORS enabled
- Gemini AI integration for analysis
- Neural embedding processing and quality metrics

### Frontend (React + TypeScript)
- **Location**: `demo/embedding-analyzer`
- **Port**: `http://localhost:3000`
- Interactive visualization with Recharts
- Multi-step workflow interface
- Real-time quality metrics display

### Core AI Engine
- **Location**: `gemni_analyzer.py`
- Google Gemini API integration
- Quality metrics calculation
- Advanced denoising algorithms
- Performance prediction

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ and Node.js 14+
- Google Gemini API key ([Get it FREE here](https://makersuite.google.com/app/apikey))

### Installation (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/lynntheophene/embedding_qualityanalyser.git
cd embedding_qualityanalyser

# 2. Install backend dependencies
pip install -r req.txt

# 3. Configure API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Start backend (Terminal 1)
python api_server.py

# 5. Start frontend (Terminal 2)
cd demo/embedding-analyzer
npm install
npm start
```

**🎉 Done!** Open `http://localhost:3000` in your browser.

**📖 Detailed Instructions**: See [SETUP.md](SETUP.md) for step-by-step setup guide with troubleshooting.

## 📖 Usage

### Web Interface Workflow

1. **Upload**: Click "Generate Sample Data" to create demo embeddings or upload your own
2. **Visualize**: View interactive scatter plots and initial quality metrics
3. **Analyze**: Click "Analyze with Gemini AI" to get AI-powered quality assessment
4. **Clean**: Apply denoising based on Gemini's recommendations

### Command Line Usage

```python
from gemni_analyzer import NeuralEmbeddingAnalyzer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize analyzer
analyzer = NeuralEmbeddingAnalyzer(api_key=os.getenv("GEMINI_API_KEY"))

# Run the complete pipeline
results = analyzer.run_pipeline()

# Save cleaned embeddings
import numpy as np
np.save('cleaned_embeddings.npy', results['cleaned_embeddings'])
```

### Using Your Own Data

```python
# Load your embeddings (.npy, .csv, or .json)
results = analyzer.run_pipeline(filepath='your_embeddings.npy', format='npy')

# Access results
print(f"Quality improved from {results['before_metrics']['separability']:.2f} to {results['after_metrics']['separability']:.2f}")
```

## 📊 Quality Metrics

The analyzer calculates and monitors:

- **Separability Score**: Measures class distinction (target: > 0.75)
- **Noise Level**: Signal quality assessment (target: < 0.3)
- **Cluster Coherence**: Data consistency evaluation (target: > 0.6)
- **Outlier Ratio**: Artifact detection (target: < 10%)
- **SNR**: Signal-to-noise ratio (target: > 10 dB)

## 🔧 API Endpoints

### Backend API

- `GET /api/health` - Health check
- `POST /api/generate-sample` - Generate sample embeddings
- `POST /api/analyze-quality` - Analyze embedding quality with Gemini
- `POST /api/denoise-embeddings` - Apply denoising
- `POST /api/upload-embeddings` - Upload embedding files
- `POST /api/download-cleaned` - Download cleaned embeddings

## 🐛 Troubleshooting

### Backend won't start
- Make sure you've set `GEMINI_API_KEY` in `.env`
- Check that all Python dependencies are installed: `pip install -r req.txt`
- Verify Python version: `python --version` (needs 3.8+)

### Frontend won't connect to backend
- Ensure backend is running on port 5000
- Check browser console for CORS errors
- Verify `proxy` setting in `package.json` points to `http://localhost:5000`

### Gemini API errors
- Verify your API key is valid
- Check you haven't exceeded rate limits
- Ensure you have internet connectivity

## 📁 Project Structure

```
embedding_qualityanalyser/
├── api_server.py              # Flask backend server
├── gemni_analyzer.py          # Core AI analyzer
├── req.txt                    # Python dependencies
├── .env.example               # Environment template
├── demo/
│   └── embedding-analyzer/    # React frontend
│       ├── src/
│       │   ├── App.js        # Main React component
│       │   ├── App.css       # Styles
│       │   └── index.js      # Entry point
│       ├── public/
│       └── package.json
└── README.md                  # This file
```

## 🎓 Use Cases

- **BCI Research**: Improve neural signal quality for brain-computer interfaces
- **EEG Analysis**: Clean and validate EEG embeddings
- **Neural Decoding**: Enhance embeddings for speech/motor decoding tasks
- **Quality Control**: Automated assessment of neural data quality

## 🔑 Getting a Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it in your `.env` file

The free tier includes:
- 60 requests per minute
- 1,500 requests per day
- Perfect for development and testing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Google Gemini AI for intelligent analysis
- React and Recharts for visualization
- Flask for backend API
- scikit-learn for ML utilities

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for the BCI community**
