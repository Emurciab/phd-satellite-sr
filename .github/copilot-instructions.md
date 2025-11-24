# Satellite Remote Sensing - AI Coding Agent Instructions

## Project Overview
This is a PhD research project focused on satellite image pairing and analysis using Google Earth Engine (GEE), specifically for matching Planet Labs and Sentinel-2 imagery over multiple regions and time periods. The project processes satellite data for environmental monitoring and analysis.

## Core Architecture

### Main Components
- **`Auth/AuthGEE.py`**: Central orchestration module containing all image processing, authentication, and pairing logic
- **Google Earth Engine Integration**: All satellite data access and processing happens through GEE API
- **CSV Output Pipeline**: Results exported to CSV format for further analysis

### Data Flow
1. **Authentication**: GEE service account credentials loaded from JSON file
2. **Image Collection**: Separate pipelines for Planet Labs and Sentinel-2 data
3. **Quality Assessment**: Multiple quality scoring systems for different satellite sources
4. **Temporal Matching**: Smart pairing algorithm with configurable tolerance windows
5. **Export**: Batch processing results saved to CSV with detailed metadata

## Key Patterns & Conventions

### GEE Function Naming
- Helper functions prefixed with `_` (e.g., `_add_common_fields`, `_add_s2_quality`)
- Factory functions use `make_` prefix (e.g., `make_add_udm_mask_and_score_for_aoi`)
- Processing pipelines follow verb-noun pattern (`find_pairs`, `planet_best_per_day`)

### Image Quality Management
```python
# Sentinel-2: Uses CLOUDY_PIXEL_PERCENTAGE (0-100, lower better)
def _add_s2_quality(img):
    q = _num_prop_or_default(img, 'CLOUDY_PIXEL_PERCENTAGE', 100)
    return img.set({'q_cloud': q})

# Planet: Handles multiple cloud coverage formats with fallbacks
def _add_planet_quality(img):
    # Prefers 'cloud_cover' (0-1) -> %, else 'cloud_percent' (0-100)
```

### Composite Scoring System
All images get a unified `score` field combining:
- Cloud quality (primary factor, weighted 1000x)
- Distance from solar noon (temporal optimization)
- Timestamp (tie-breaking)

### Adaptive Mosaicking Strategy
The `planet_best_per_day_adaptive` function implements intelligent band handling:
- **8-band priority**: Uses 8-band Planet images when sufficient AOI coverage (95%+)
- **Harmonization fallback**: Reduces all to 4-band when mixing sources needed
- **Strategy tagging**: Each result tagged with processing approach used

### Region Configuration
```python
regions = [
    {
        "name": "STA-2",  # Study area identifier
        "aoi": ee.Geometry.Polygon([...]),  # Area of interest
        "start": "2020-01-01",
        "end": "2021-01-01",
        "planet_collection_id": "custom_collection"  # Optional override
    }
]
```

## Critical Development Workflows

### Authentication Setup
- Service account JSON must be in `Auth/` directory
- Update credentials in `AuthGEE.py` with correct email and path
- Initialize GEE with: `ee.Initialize(credentials)`

### Adding New Regions
1. Define geometry using `ee.Geometry.Polygon()` format
2. Set appropriate date ranges (typically yearly intervals)
3. Optional: specify custom Planet collection ID for different data sources
4. Add to `regions` list for batch processing

### Quality Threshold Tuning
- **Planet UDM confidence**: Default 80%, adjust in `make_add_udm_mask_and_score_for_aoi(conf_threshold=80)`
- **S2 cloud filtering**: `filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))`
- **Planet cloud filtering**: `filter(ee.Filter.lt('q_cloud', 20))`
- **AOI cloud fraction**: `filter(ee.Filter.lt('AOI_CLOUD_FRAC', 0.1))`

### Temporal Matching Configuration
The `_match_dates_with_tolerance_nn` function uses nearest-neighbor matching:
- Default tolerance: ±7 days (configurable)
- Prevents double-matching (each Planet image used once)
- Returns matched pairs + unmatched dates from both sources

## Integration Points

### External Dependencies
- **Google Earth Engine**: `import ee` - all satellite data access
- **Pandas**: CSV export and data manipulation
- **Service Account**: JSON credentials for GEE authentication

### Collection IDs
- **Primary Planet**: `'projects/remotesensingeduart/assets/testSTAs'`
- **Fallback Planet**: `'projects/ee-eeamurciabb/assets/testSTAs'`
- **Sentinel-2**: `'COPERNICUS/S2_SR_HARMONIZED'`

### Output Format
Results exported to CSV with columns:
- `Region`, `Start`, `End`: Configuration metadata
- `Planet_date`, `Sentinel_date`: Unmatched acquisition dates
- `Paired_planet_date`, `Paired_s2_dates`: Successfully matched pairs

## Error Handling Patterns
- **Collection fallback**: Automatic retry with secondary Planet collection if primary fails
- **Property defaults**: `_num_prop_or_default()` prevents crashes from missing metadata
- **Exception wrapping**: Try-catch around collection access with informative logging

## Super Resolution Integration

### Memory-Efficient Processing
- **`sr_efficient_pipeline.py`**: Tile-based processing pipeline for SR training data
- **Server-side operations**: Coregistration and harmonization done in GEE cloud to minimize data transfer
- **Smart caching**: SQLite + pickle for fast access to processed tiles
- **On-demand generation**: Training data created as needed, not stored in full

### Temporal Sequence Processing
```python
# Create temporal sequences for attention mechanism training
sequences = create_temporal_sequences(matched_pairs, sequence_length=3)

# Process tiles instead of full images to minimize memory
for tile in spatial_tiles:
    lr_sequence, hr_sequence = process_temporal_tile(sequence, tile)
```

### Training Data Pipeline
- **Tile size**: 256-512px tiles based on GPU memory (configurable)
- **Temporal length**: 3-5 frame sequences for attention learning
- **Augmentations**: Temporal-consistent spatial and temporal augmentations
- **Batch generation**: `SuperResolutionDataGenerator` yields training batches on-demand

## Development Tips
- **Scale parameters**: Use `scale=50, tileScale=2` for most reduce operations to balance speed/accuracy
- **Memory management**: `maxPixels=1e8` or `1e9` depending on operation complexity
- **Date formatting**: All dates use 'YYYY-MM-DD' format consistently
- **CSV save handling**: Includes fallback path when external drive unavailable
- **SR processing**: Use tile-based approach for memory efficiency - process 256px tiles instead of full images
- **Cache management**: Clear SR cache when changing processing parameters or region configurations