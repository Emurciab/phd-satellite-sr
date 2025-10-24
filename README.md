# Satellite Remote Sensing (SR) - Planet Labs & Sentinel-2 Image Pairing

This PhD research project focuses on temporal pairing and analysis of Planet Labs and Sentinel-2 satellite imagery using Google Earth Engine (GEE) for environmental monitoring applications.

## Project Overview

The system processes satellite imagery from multiple sources to create matched pairs of observations over study areas, enabling temporal analysis and change detection studies.

### Key Features
- **Multi-sensor Integration**: Combines Planet Labs (3-4m resolution) and Sentinel-2 (10-20m resolution) imagery
- **Adaptive Mosaicking**: Intelligent band harmonization between 4-band and 8-band Planet imagery
- **Quality-based Selection**: Advanced cloud masking and quality scoring for optimal image selection
- **Temporal Matching**: Nearest-neighbor pairing with configurable tolerance windows
- **Batch Processing**: Automated processing across multiple study areas and time periods

## Architecture

- **`Auth/AuthGEE.py`**: Main processing pipeline with GEE authentication and image pairing logic
- **`coreg/coreg.py`**: Co-registration utilities for multi-sensor alignment
- **`.github/copilot-instructions.md`**: Detailed guidance for AI coding agents

## Study Areas

The project focuses on multiple Study Areas (STAs) in South Florida:
- STA-1E, STA-1W: Eastern and Western sections of Study Area 1
- STA-2: Study Area 2
- STA-3/4: Combined Study Areas 3 and 4
- STA-5/6: Combined Study Areas 5 and 6

Each area is processed across multiple yearly time periods (2020-2025).

## Dependencies

- **Google Earth Engine Python API**: Satellite data access and processing
- **pandas**: Data manipulation and CSV export
- **Service Account Authentication**: Required for GEE access

## Setup

1. Ensure Google Earth Engine service account credentials are properly configured
2. Update credential paths in `Auth/AuthGEE.py`
3. Configure study area geometries and date ranges as needed

## Output

Results are exported to CSV format containing:
- Matched image pairs with acquisition dates
- Unmatched images from both sensors
- Quality metrics and processing metadata

## Usage

The main processing pipeline is executed by running the batch processing loop in `Auth/AuthGEE.py`, which iterates through all configured regions and time periods.

---

**Note**: This project is part of ongoing PhD research in remote sensing and environmental monitoring.