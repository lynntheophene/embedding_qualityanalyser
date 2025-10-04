#!/usr/bin/env python3
"""
Master Test Runner
Runs all test suites and reports results
"""

import subprocess
import sys
import time

def run_test(name, command, timeout=90):
    """Run a test and return result"""
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print('='*70)
    
    start_time = time.time()
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd='.'
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {name} PASSED ({elapsed:.1f}s)")
            return True
        else:
            print(f"❌ {name} FAILED ({elapsed:.1f}s)")
            print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏱️  {name} TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"💥 {name} ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("="*70)
    print("NEURAL EMBEDDING QUALITY ANALYZER - MASTER TEST RUNNER")
    print("="*70)
    
    tests = [
        ("Core Functionality Tests", "python test_core.py", 60),
        ("Fix Validation Tests", "python test_fixes.py", 90),
        ("Data Validation Tests", "python test_data_validation.py", 90),
        ("API Endpoint Tests", "python test_api.py", 120),
    ]
    
    results = []
    for name, command, timeout in tests:
        result = run_test(name, command, timeout)
        results.append((name, result))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{len(results)} passed")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Everything is working correctly")
        print("✅ All data is valid (no NaN/Inf)")
        print("✅ All API responses return success: True")
        return 0
    else:
        print(f"\n⚠️  {failed} test suite(s) failed")
        print("Please check the output above for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())
