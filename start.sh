#!/bin/bash
# Startup script for Neural Embedding Quality Analyzer

echo "🧠 Neural Embedding Quality Analyzer - Startup Script"
echo "======================================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your GEMINI_API_KEY"
    echo "Get your FREE API key from: https://makersuite.google.com/app/apikey"
    echo ""
    echo "After adding your API key, run this script again."
    exit 1
fi

# Check if GEMINI_API_KEY is set
if ! grep -q "^GEMINI_API_KEY=..*" .env; then
    echo "⚠️  GEMINI_API_KEY not set in .env file!"
    echo "Please edit .env and add your API key."
    echo "Get your FREE API key from: https://makersuite.google.com/app/apikey"
    exit 1
fi

echo "✓ Environment configuration found"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
if ! python3 -c "import flask, flask_cors, google.generativeai" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip install -r req.txt
fi
echo "✓ Python dependencies ready"
echo ""

# Start backend
echo "🚀 Starting Flask backend on http://localhost:5000..."
python3 api_server.py &
BACKEND_PID=$!
echo "✓ Backend started (PID: $BACKEND_PID)"
echo ""

# Wait a bit for backend to start
sleep 2

# Check if frontend dependencies are installed
if [ ! -d "demo/embedding-analyzer/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd demo/embedding-analyzer
    npm install
    cd ../..
fi

# Start frontend
echo "🎨 Starting React frontend on http://localhost:3000..."
echo ""
cd demo/embedding-analyzer
npm start &
FRONTEND_PID=$!

echo ""
echo "======================================================"
echo "✅ Application Started!"
echo "======================================================"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔌 Backend:  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for user to stop
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT

# Keep script running
wait
