#!/usr/bin/env python3
"""
Quick validation test - verifies the system is properly set up
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    print("Checking Python version...", end=" ")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} (need 3.8+)")
        return False

def check_dependencies():
    """Check if all dependencies are installed"""
    print("\nChecking Python dependencies:")
    required = [
        'flask',
        'flask_cors',
        'numpy',
        'scipy',
        'sklearn',
        'google.generativeai',
        'dotenv'
    ]
    
    all_ok = True
    for module in required:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} (missing)")
            all_ok = False
    
    return all_ok

def check_env_file():
    """Check if .env file exists"""
    print("\nChecking environment configuration:")
    
    if not os.path.exists('.env'):
        print("  ✗ .env file not found")
        print("     Run: cp .env.example .env")
        return False
    
    print("  ✓ .env file exists")
    
    # Check if GEMINI_API_KEY is set
    with open('.env', 'r') as f:
        content = f.read()
        if 'GEMINI_API_KEY=' in content and not 'your_gemini_api_key_here' in content:
            # Check if it's not empty
            for line in content.split('\n'):
                if line.startswith('GEMINI_API_KEY=') and len(line.split('=')[1].strip()) > 10:
                    print("  ✓ GEMINI_API_KEY is set")
                    return True
    
    print("  ⚠️  GEMINI_API_KEY not configured")
    print("     Edit .env and add your API key from:")
    print("     https://makersuite.google.com/app/apikey")
    return False

def check_imports():
    """Check if main modules import correctly"""
    print("\nChecking module imports:")
    
    try:
        from gemni_analyzer import NeuralEmbeddingAnalyzer
        print("  ✓ gemni_analyzer.py")
    except Exception as e:
        print(f"  ✗ gemni_analyzer.py ({e})")
        return False
    
    try:
        from api_server import app
        print("  ✓ api_server.py")
    except Exception as e:
        print(f"  ✗ api_server.py ({e})")
        return False
    
    return True

def check_frontend():
    """Check if frontend is set up"""
    print("\nChecking frontend setup:")
    
    frontend_path = 'demo/embedding-analyzer'
    
    if not os.path.exists(f"{frontend_path}/package.json"):
        print("  ✗ package.json not found")
        return False
    
    print("  ✓ package.json exists")
    
    if os.path.exists(f"{frontend_path}/src/App.js"):
        print("  ✓ React app files exist")
    else:
        print("  ✗ React app files missing")
        return False
    
    if os.path.exists(f"{frontend_path}/node_modules"):
        print("  ✓ Node modules installed")
    else:
        print("  ⚠️  Node modules not installed")
        print("     Run: cd demo/embedding-analyzer && npm install")
    
    return True

def main():
    """Run all validation checks"""
    print("=" * 70)
    print("Neural Embedding Quality Analyzer - System Validation")
    print("=" * 70)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment", check_env_file),
        ("Module Imports", check_imports),
        ("Frontend", check_frontend)
    ]
    
    results = {}
    for name, check in checks:
        try:
            results[name] = check()
        except Exception as e:
            print(f"\n✗ Error checking {name}: {e}")
            results[name] = False
    
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 All checks passed! Your system is ready to use.")
        print("\nTo start the application:")
        print("  1. Backend:  python api_server.py")
        print("  2. Frontend: cd demo/embedding-analyzer && npm start")
        print("  3. Open:     http://localhost:3000")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nFor help, see: SETUP.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
