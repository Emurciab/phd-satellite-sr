# Workspace Reorganization Plan

## Current State Analysis

### Main Pipeline (PRODUCTION - KEEP)
**Location**: `scripts/preprocessing/`
- ✅ `numpy_efficient_pipeline.py` - **MAIN PIPELINE** (with spatial + spectral harmonization)
- ✅ `generate_sta2_full_dataset.py` - **DATASET GENERATOR**
- ✅ `chunked_download_fast.py` - **OPTIMIZED DOWNLOADER** (9×9 grid, 12 workers)

### Legacy Files (TO ARCHIVE)
**Location**: `scripts/preprocessing/`
- 🗄️ `efficient_data_handler.py` - Old TensorFlow-based approach
- 🗄️ `sr_efficient_pipeline.py` - Old tile-based pipeline
- 🗄️ `streaming_pipeline.py` - Old streaming approach
- 🗄️ `chunked_download.py` - Old slower downloader
- 🗄️ `create_chunked_dataset.py` - Old dataset generator
- 🗄️ `create_fixed_dataset.py` - Old fixed dataset script
- 🗄️ `Preprocessing.py` - Very old preprocessing

### Test/Development Files (TO ARCHIVE)
**Location**: Various
- 🗄️ `test_harmonization.py` - Development test (matplotlib version)
- 🗄️ `test_harmonization_simple.py` - Development test (no matplotlib)
- ✅ `test_harmonization_standalone.py` - **KEEP IN ROOT** (minimal test, useful)

### Documentation (ORGANIZE)
**Location**: `docs/`
- ✅ `SPECTRAL_HARMONIZATION.md` - **CURRENT** (just created)
- ✅ `COREGISTRATION_COMPARISON.md` - **CURRENT** (recent)
- ✅ `TILING_FIX_SUMMARY.md` - **CURRENT** (recent)
- 🗄️ `EFFICIENT_SR_SUMMARY.md` - OLD approach
- 🗄️ `memory_efficient_integration.md` - OLD approach
- 🗄️ Many others from old implementations

## Reorganization Structure

```
SR/
├── scripts/
│   ├── preprocessing/
│   │   ├── numpy_efficient_pipeline.py          # MAIN PIPELINE
│   │   ├── generate_sta2_full_dataset.py        # DATASET GENERATOR
│   │   ├── chunked_download_fast.py             # DOWNLOADER
│   │   └── gee_export_helper.py                 # GEE helper
│   ├── visualization/
│   │   └── (existing viz scripts - keep)
│   ├── validation/
│   │   └── (existing validation - keep)
│   └── training/
│       └── (training scripts - when created)
├── tests/
│   └── (all test files - keep)
├── docs/
│   ├── current/                                  # NEW: Active docs
│   │   ├── PIPELINE_OVERVIEW.md                 # NEW: Main reference
│   │   ├── SPECTRAL_HARMONIZATION.md
│   │   ├── COREGISTRATION_COMPARISON.md
│   │   ├── TILING_FIX_SUMMARY.md
│   │   ├── WORKSPACE_STRUCTURE.md
│   │   └── QUICK_REFERENCE.md
│   └── archive/                                  # NEW: Historical docs
│       ├── EFFICIENT_SR_SUMMARY.md
│       ├── memory_efficient_integration.md
│       └── (other old docs)
├── legacy/                                       # NEW: Archived code
│   └── preprocessing/
│       ├── efficient_data_handler.py
│       ├── sr_efficient_pipeline.py
│       ├── streaming_pipeline.py
│       ├── chunked_download.py
│       ├── create_chunked_dataset.py
│       ├── create_fixed_dataset.py
│       ├── Preprocessing.py
│       ├── test_harmonization.py
│       └── test_harmonization_simple.py
├── Auth/
│   └── (keep as-is - active GEE auth)
├── cache/
│   └── (keep as-is - runtime cache)
├── data/
│   └── (keep as-is - datasets)
├── coreg/
│   └── (keep as-is - AROSICS module)
├── test_harmonization_standalone.py             # Quick validation test
└── README.md                                     # NEW: Main project readme
```

## Actions

### 1. Create New Directories
- ✅ `legacy/preprocessing/`
- ✅ `docs/current/`
- ✅ `docs/archive/`

### 2. Move Legacy Code
Move to `legacy/preprocessing/`:
- efficient_data_handler.py
- sr_efficient_pipeline.py
- streaming_pipeline.py
- chunked_download.py
- create_chunked_dataset.py
- create_fixed_dataset.py
- Preprocessing.py
- test_harmonization.py
- test_harmonization_simple.py

### 3. Organize Documentation
Move to `docs/current/`:
- SPECTRAL_HARMONIZATION.md
- COREGISTRATION_COMPARISON.md
- TILING_FIX_SUMMARY.md
- WORKSPACE_STRUCTURE.md
- QUICK_REFERENCE.md

Move to `docs/archive/`:
- EFFICIENT_SR_SUMMARY.md
- memory_efficient_integration.md
- SETUP_GUIDE.md (old)
- UV_SETUP_GUIDE.md (old)
- NEXT_STEPS.md (old)
- INTEGRATION_SUMMARY.md (old)
- IMPORT_FIXES.md (old)
- VALIDATION_RESULTS.md (old)

### 4. Create New Main README
Create comprehensive `README.md` with:
- Project overview
- Current pipeline architecture
- Quick start guide
- Link to detailed docs in `docs/current/`

## Benefits

1. ✅ **Clear separation**: Production vs legacy code
2. ✅ **Easy navigation**: Current docs vs historical docs
3. ✅ **Maintainability**: Only maintain active code
4. ✅ **Onboarding**: New users see clean structure
5. ✅ **History preserved**: Old code available but not cluttering
6. ✅ **Documentation organized**: Current vs archived

## What Stays in Main Directory

**Production Code**:
- `scripts/preprocessing/numpy_efficient_pipeline.py`
- `scripts/preprocessing/generate_sta2_full_dataset.py`
- `scripts/preprocessing/chunked_download_fast.py`

**Active Documentation**:
- `docs/current/` (new)

**Testing**:
- `tests/` (all tests)
- `test_harmonization_standalone.py` (quick validation)

**Data & Cache**:
- `Auth/` (GEE credentials)
- `cache/` (runtime)
- `data/` (datasets)
- `coreg/` (AROSICS module)
