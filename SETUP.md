# 🚀 Setup Guide - Neural Embedding Quality Analyzer

Complete step-by-step guide to get the Neural Embedding Quality Analyzer running on your system.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **Node.js 14+** ([Download](https://nodejs.org/))
- **npm** (comes with Node.js)
- **Git** ([Download](https://git-scm.com/))

### Check Your Versions

```bash
python3 --version  # Should be 3.8 or higher
node --version     # Should be 14 or higher
npm --version      # Should be 6 or higher
```

## 🔑 Step 1: Get Your Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the API key (it starts with something like `AIza...`)
5. Keep it safe - you'll need it in Step 3

**Important**: The Gemini API is **FREE** with generous limits:
- 60 requests per minute
- 1,500 requests per day
- Perfect for development and testing

## 📦 Step 2: Clone and Install

### Clone the Repository

```bash
git clone https://github.com/lynntheophene/embedding_qualityanalyser.git
cd embedding_qualityanalyser
```

### Install Backend Dependencies

```bash
# Install Python packages
pip install -r req.txt
```

This will install:
- Flask (web server)
- Google Generative AI (Gemini API)
- NumPy, SciPy, scikit-learn (data processing)
- And other required packages

### Install Frontend Dependencies

```bash
# Navigate to frontend directory
cd demo/embedding-analyzer

# Install Node packages
npm install

# Return to main directory
cd ../..
```

## ⚙️ Step 3: Configure Environment

### Create .env File

```bash
# Copy the example file
cp .env.example .env
```

### Add Your API Key

Open `.env` in your favorite text editor and replace the placeholder:

```env
# Before:
GEMINI_API_KEY=your_gemini_api_key_here

# After:
GEMINI_API_KEY=AIzaSyB...your-actual-key-here
```

**Using nano (Linux/Mac)**:
```bash
nano .env
# Edit, then press Ctrl+X, Y, Enter to save
```

**Using vim**:
```bash
vim .env
# Press i to insert, edit, press Esc, type :wq, press Enter
```

**Using VS Code**:
```bash
code .env
```

## 🏃 Step 4: Start the Application

You have two options:

### Option A: Use the Startup Script (Recommended)

**Linux/Mac**:
```bash
./start.sh
```

**Windows**:
```bash
# Start backend (in one terminal)
python api_server.py

# Start frontend (in another terminal)
cd demo/embedding-analyzer
npm start
```

### Option B: Manual Startup

**Terminal 1 - Backend**:
```bash
python api_server.py
```

You should see:
```
🚀 Starting Neural Embedding Analyzer API Server...
🔗 React frontend can connect to: http://localhost:5000
💡 Make sure your GEMINI_API_KEY is set!
* Running on http://127.0.0.1:5000
```

**Terminal 2 - Frontend**:
```bash
cd demo/embedding-analyzer
npm start
```

The browser should automatically open to `http://localhost:3000`

## ✅ Step 5: Verify It's Working

### Test the Backend

Open a new terminal and run:

```bash
curl http://localhost:5000/api/health
```

You should see:
```json
{"backend":"gemini-analyzer","status":"healthy"}
```

### Test the Frontend

1. Open your browser to `http://localhost:3000`
2. You should see the Neural Embedding Quality Analyzer interface
3. Click **"Generate Sample Data"**
4. You should see a scatter plot with embeddings

## 🎯 Usage Workflow

Once everything is running:

1. **Upload** - Click "Generate Sample Data" to create demo embeddings
2. **Visualize** - View the interactive scatter plot and quality metrics
3. **Analyze** - Click "🤖 Analyze with Gemini AI" for AI-powered analysis
4. **Clean** - Click "🧹 Apply Denoising" to clean the embeddings

## 🐛 Troubleshooting

### Backend Won't Start

**Problem**: `ModuleNotFoundError: No module named 'flask'`

**Solution**:
```bash
pip install -r req.txt
```

---

**Problem**: `⚠️ Please set GEMINI_API_KEY environment variable`

**Solution**: Make sure you:
1. Created `.env` file: `cp .env.example .env`
2. Added your API key to `.env`
3. API key is not in quotes

---

**Problem**: Port 5000 already in use

**Solution**:
```bash
# Find and kill the process using port 5000
# Linux/Mac:
lsof -ti:5000 | xargs kill -9

# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Frontend Won't Start

**Problem**: `npm: command not found`

**Solution**: Install Node.js from [nodejs.org](https://nodejs.org/)

---

**Problem**: `Module not found: Error: Can't resolve 'recharts'`

**Solution**:
```bash
cd demo/embedding-analyzer
npm install
```

---

**Problem**: Frontend can't connect to backend

**Solution**: 
1. Make sure backend is running on port 5000
2. Check `package.json` has `"proxy": "http://localhost:5000"`
3. Try clearing browser cache (Ctrl+Shift+R)

### Gemini API Errors

**Problem**: `API key not valid`

**Solution**: 
1. Verify your API key is correct in `.env`
2. Make sure there are no extra spaces or quotes
3. Generate a new key if needed

---

**Problem**: `Rate limit exceeded`

**Solution**: You've hit the API limit. Wait a few minutes and try again.

## 📊 Testing Without Gemini API

To test the core functionality without using the Gemini API:

```bash
python test_core.py
```

This will test:
- Sample data generation
- Quality metrics calculation
- Outlier removal

## 🔄 Updating

To get the latest version:

```bash
git pull origin main
pip install -r req.txt
cd demo/embedding-analyzer
npm install
cd ../..
```

## 📝 Using Your Own Data

To analyze your own neural embeddings:

```python
from gemni_analyzer import NeuralEmbeddingAnalyzer
import os
from dotenv import load_dotenv

load_dotenv()
analyzer = NeuralEmbeddingAnalyzer(api_key=os.getenv("GEMINI_API_KEY"))

# Load your embeddings
results = analyzer.run_pipeline(filepath='your_embeddings.npy', format='npy')

# Save cleaned version
import numpy as np
np.save('cleaned_embeddings.npy', results['cleaned_embeddings'])
```

Supported formats:
- `.npy` - NumPy arrays
- `.csv` - CSV files
- `.json` - JSON arrays

## 🎓 Next Steps

- Read the [README.md](README.md) for more details
- Check out the code in `gemni_analyzer.py` to understand the algorithms
- Experiment with your own neural data
- Modify parameters in the code to suit your needs

## 💬 Need Help?

- Check existing [issues](https://github.com/lynntheophene/embedding_qualityanalyser/issues)
- Open a new issue with your problem
- Include error messages and steps to reproduce

## 🎉 Success!

If you see the React interface and can generate sample data, you're all set! 

Enjoy analyzing your neural embeddings with AI! 🧠✨
