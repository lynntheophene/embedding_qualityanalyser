# Data Validation Documentation

## Overview

This project ensures **"every data must be true"** - meaning all data throughout the system is valid and all API responses indicate success.

## What "Every Data Must Be True" Means

1. **No Invalid Values**
   - No NaN (Not a Number) values in any numerical data
   - No Inf (Infinity) values in any numerical data
   - All arrays have valid shapes and types

2. **All API Responses Succeed**
   - Every API endpoint returns `{"success": true}` when operations complete successfully
   - Only returns `{"success": false}` when there's an actual error (e.g., bad input)

3. **Data Integrity Throughout Pipeline**
   - Generated embeddings are valid numpy arrays
   - Calculated metrics are valid numbers
   - Denoised data maintains sample count and validity
   - Demo files contain clean, valid data

## Validation Tests

### Comprehensive Test Suite: `test_data_validation.py`

This test validates everything:

```bash
python test_data_validation.py
```

**What it checks:**

1. **Data Generation** - Ensures generated embeddings and labels are valid
2. **Metrics Calculation** - Verifies all quality metrics are valid numbers
3. **Analysis Generation** - Checks Gemini/fallback analysis format
4. **Denoising** - Validates denoised data integrity
5. **Demo Files** - Confirms demo data files are valid
6. **API Responses** - Ensures all endpoints return `success: true`

### Example Output

```
🎉 ALL VALIDATIONS PASSED!
✅ Every data is true (valid)
✅ All API responses return success: True
✅ No NaN or Inf values anywhere
✅ All data types are correct
✅ Demo files are valid
```

## Other Test Suites

### Core Functionality: `test_core.py`
Tests basic analyzer functions without API calls.

```bash
python test_core.py
```

### API Endpoints: `test_api.py`
Tests all Flask API endpoints with actual server.

```bash
python test_api.py
```

### Fix Validation: `test_fixes.py`
Validates timeout handling and fallback analysis.

```bash
python test_fixes.py
```

## Data Integrity Guarantees

### For Embeddings
- Shape: `(n_samples, n_features)` - always 2D
- Type: `float64` numpy array
- Values: No NaN, no Inf
- Range: Typically -10 to +10 (varies)

### For Labels
- Shape: `(n_samples,)` - 1D array
- Type: `int64` numpy array
- Values: Binary (0 or 1)
- Count: Equal split (50% class 0, 50% class 1)

### For Metrics
All metrics are valid floats:
- `separability`: 0.0 to 1.0
- `noise_level`: 0.0 to 1.0+
- `cluster_coherence`: 0.0 to 1.0
- `outlier_ratio`: 0.0 to 1.0
- `snr_db`: Real number (can be negative)
- `n_samples`: Positive integer
- `n_features`: Positive integer

### For API Responses
All successful responses include:
```json
{
  "success": true,
  "data": { ... }
}
```

Error responses include:
```json
{
  "success": false,
  "error": "Error message"
}
```

## Running All Tests

To verify everything works:

```bash
# Core tests
python test_core.py

# Fix tests
python test_fixes.py

# Data validation
python test_data_validation.py

# API tests (requires server)
python test_api.py
```

All tests should pass with exit code 0.

## Common Issues

### "NaN values detected"
This should never happen! If it does:
1. Check input data format
2. Verify calculations don't divide by zero
3. Run `test_data_validation.py` to identify where

### "success: false in API response"
Check:
1. Input data format (expecting specific JSON structure)
2. File upload format (only .npy, .csv, .json)
3. Server logs for actual error message

## Success Criteria

The system meets "every data must be true" when:

✅ All test suites pass  
✅ No NaN or Inf anywhere  
✅ All API responses have `success: true` (when they should)  
✅ Demo files are valid  
✅ Generated data is valid  
✅ Denoised data is valid  

Run `python test_data_validation.py` to verify all criteria are met.
