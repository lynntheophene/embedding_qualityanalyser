# Quick Reference: Data Validation

## One-Line Test Command

```bash
python run_all_tests.py
```

This runs all tests and verifies:
- ✅ Everything works
- ✅ Every data is valid (no NaN/Inf)
- ✅ All API responses return `success: True`

## Expected Output

```
🎉 ALL TESTS PASSED!
✅ Everything is working correctly
✅ All data is valid (no NaN/Inf)
✅ All API responses return success: True
```

## What Gets Tested

1. **Core Functionality** (`test_core.py`)
   - Data generation
   - Quality metrics
   - Outlier removal

2. **Fix Validation** (`test_fixes.py`)
   - Demo data files
   - Timeout handling
   - Denoising
   - API endpoints

3. **Data Validation** (`test_data_validation.py`)
   - No NaN values
   - No Inf values
   - Correct data types
   - Valid array shapes
   - API success responses

4. **API Endpoints** (`test_api.py`)
   - Health check
   - Generate sample
   - Analyze quality
   - Denoise embeddings

## Individual Tests

```bash
python test_core.py              # Core functionality
python test_fixes.py             # Fix validation
python test_data_validation.py   # Data validation
python test_api.py               # API endpoints
```

## Documentation

- **[DATA_VALIDATION.md](DATA_VALIDATION.md)** - Detailed data validation documentation
- **[RESOLUTION_SUMMARY.md](RESOLUTION_SUMMARY.md)** - Issue resolution summary
- **[README.md](README.md)** - Main project documentation

## Success Criteria

All tests pass when:
- No NaN or Inf values in any data
- All API responses have `success: true` (when they should)
- All data types are correct
- Demo files are valid
- All functionality works correctly

## Troubleshooting

If tests fail:
1. Check Python version (needs 3.8+)
2. Install dependencies: `pip install -r req.txt`
3. Review test output for specific error
4. See [DATA_VALIDATION.md](DATA_VALIDATION.md) for details
