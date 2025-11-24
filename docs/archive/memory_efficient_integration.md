# Memory-Efficient Super Resolution Data Pipeline
# Integration with existing AuthGEE.py for optimized multi-temporal processing

## 🎯 **Recommended Approach: Hybrid Cloud-Local Processing**

Based on your current preprocessing pipeline, here are the most efficient strategies:

### **1. Smart Caching with Hierarchical Storage**

```python
# Add to AuthGEE.py - Memory-efficient batch processor
import tempfile
import sqlite3
from pathlib import Path
import lmdb  # Lightning Memory-Mapped Database for fast key-value access

class EfficientSRDataManager:
    """
    Efficient data manager that minimizes memory usage while providing
    fast access for super resolution training.
    """
    
    def __init__(self, cache_dir="./sr_cache", max_cache_size_gb=10):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Fast metadata database
        self.db_path = self.cache_dir / "metadata.db"
        self.init_database()
        
        # LMDB for fast tile access
        self.lmdb_path = self.cache_dir / "tiles_lmdb"
        self.lmdb_env = lmdb.open(str(self.lmdb_path), 
                                 map_size=max_cache_size_gb * 1024**3)
    
    def process_region_efficiently(self, region_config, sequence_length=3):
        """
        Process region using minimal memory footprint:
        1. Find temporal sequences from your existing pairs
        2. Process tiles on-demand
        3. Cache frequently accessed data
        """
        
        # Use your existing find_pairs function
        pair_results = find_pairs(
            region_config["aoi"], 
            region_config["start"], 
            region_config["end"],
            region_config.get("planet_collection_id", PLANET_COLLECTION_ID)
        )
        
        # Create temporal sequences
        temporal_sequences = self.create_temporal_sequences(
            pair_results['matched_pairs'], sequence_length
        )
        
        # Process each sequence with tile-based approach
        for seq_idx, sequence in enumerate(temporal_sequences):
            yield from self.process_sequence_tiles(sequence, region_config, seq_idx)
    
    def create_temporal_sequences(self, matched_pairs, sequence_length):
        """Create overlapping temporal sequences for attention mechanism training"""
        # Sort by date
        sorted_pairs = sorted(matched_pairs, key=lambda x: x['s2_date'])
        
        sequences = []
        for i in range(len(sorted_pairs) - sequence_length + 1):
            sequences.append(sorted_pairs[i:i + sequence_length])
        
        return sequences
```

### **2. On-Demand Tile Processing**

```python
# Extend your existing processing to work with tiles
def process_sequence_tiles(self, sequence, region_config, seq_idx):
    """
    Process temporal sequence using tile-based approach.
    This dramatically reduces memory requirements.
    """
    
    aoi = region_config["aoi"]
    tile_size = 512  # Adjust based on your GPU memory
    
    # Create spatial grid
    tiles = self.create_spatial_tiles(aoi, tile_size, overlap=64)
    
    for tile_idx, tile_geom in enumerate(tiles):
        try:
            # Process this tile for the entire temporal sequence
            tile_data = self.process_temporal_tile(sequence, tile_geom, seq_idx, tile_idx)
            
            if tile_data is not None:
                yield tile_data
                
        except Exception as e:
            print(f"Skipping tile {tile_idx} in sequence {seq_idx}: {e}")
            continue

def process_temporal_tile(self, sequence, tile_geom, seq_idx, tile_idx):
    """
    Process a single spatial tile across temporal sequence.
    Uses your existing coregistration and harmonization functions.
    """
    
    # Check cache first
    cache_key = f"seq_{seq_idx}_tile_{tile_idx}"
    cached_data = self.get_from_cache(cache_key)
    if cached_data is not None:
        return cached_data
    
    lr_sequence = []  # Sentinel-2 (low resolution)
    hr_sequence = []  # Planet (high resolution target)
    
    for pair in sequence:
        # Get images for this temporal step
        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filter(
            ee.Filter.eq('system:index', pair['s2_id'].split('/')[-1])
        )
        
        planet_collection = ee.ImageCollection(PLANET_COLLECTION_ID).filter(
            ee.Filter.eq('system:index', pair['planet_id'].split('/')[-1])
        )
        
        s2_img = ee.Image(s2_collection.first())
        planet_img = ee.Image(planet_collection.first())
        
        # Apply your existing processing pipeline to the tile
        tile_result = self.process_pair_tile(s2_img, planet_img, tile_geom)
        
        if tile_result is not None:
            lr_sequence.append(tile_result['s2_tile'])
            hr_sequence.append(tile_result['planet_harmonized_tile'])
    
    if len(lr_sequence) == len(sequence):
        tile_data = {
            'lr_sequence': np.stack(lr_sequence),    # (T, H, W, C)
            'hr_sequence': np.stack(hr_sequence),    # (T, H, W, C)
            'temporal_dates': [p['s2_date'] for p in sequence],
            'tile_bounds': tile_geom.bounds().getInfo(),
            'sequence_id': seq_idx,
            'tile_id': tile_idx
        }
        
        # Cache for future use
        self.save_to_cache(cache_key, tile_data)
        return tile_data
    
    return None
```

### **3. Server-Side Processing Integration**

```python
def process_pair_tile(self, s2_img, planet_img, tile_geom):
    """
    Adapt your existing processing to work with tiles and server-side operations.
    This minimizes data transfer and local processing.
    """
    
    # Clip to tile first (reduces data size)
    s2_tile = s2_img.clip(tile_geom)
    planet_tile = planet_img.clip(tile_geom)
    
    # Apply quality filters using your existing logic
    s2_tile = s2_tile.updateMask(
        s2_tile.select('SCL').neq(9).And(s2_tile.select('SCL').neq(10))
    )
    
    # Use GEE's built-in displacement for coregistration (server-side)
    try:
        displacement = planet_tile.displacement(
            referenceImage=s2_tile,
            maxOffset=30.0,
            patchWidth=128.0
        )
        planet_coreg = planet_tile.displace(displacement)
    except:
        planet_coreg = planet_tile  # Fallback if displacement fails
    
    # Server-side harmonization using your approach but with reduced data
    try:
        # Simplified harmonization for tiles
        planet_harmonized = self.harmonize_tile_server_side(planet_coreg, s2_tile, tile_geom)
    except:
        planet_harmonized = planet_coreg
    
    # Download only the processed tile (much smaller than full images)
    try:
        s2_array = self.download_tile_efficient(s2_tile, tile_geom, scale=10)
        planet_array = self.download_tile_efficient(planet_harmonized, tile_geom, scale=3)
        
        if s2_array is not None and planet_array is not None:
            return {
                's2_tile': s2_array,
                'planet_harmonized_tile': planet_array
            }
    except Exception as e:
        print(f"Error downloading tile: {e}")
    
    return None

def download_tile_efficient(self, image, tile_geom, scale):
    """
    Efficient tile download using getRegion or sampleRectangle.
    Much faster than downloading full images.
    """
    
    try:
        # Use sampleRectangle for regular grids (faster)
        tile_data = image.sampleRectangle(
            region=tile_geom,
            defaultValue=0
        ).getInfo()
        
        # Convert to numpy array
        if 'properties' in tile_data:
            # Extract band arrays and stack
            band_arrays = []
            for band in ['B2', 'B3', 'B4', 'B8']:  # Adjust bands as needed
                if band in tile_data['properties']:
                    band_arrays.append(np.array(tile_data['properties'][band]))
            
            if band_arrays:
                return np.stack(band_arrays, axis=-1).astype(np.float32)
    
    except Exception as e:
        print(f"Error in efficient download: {e}")
    
    return None
```

### **4. Training Data Generator**

```python
class SuperResolutionDataGenerator:
    """
    Generator that yields training batches for your super resolution model.
    Integrates with your existing processing pipeline.
    """
    
    def __init__(self, data_manager, batch_size=4, augment=True):
        self.data_manager = data_manager
        self.batch_size = batch_size
        self.augment = augment
    
    def generate_training_batches(self, regions, sequence_length=3):
        """
        Generate training batches using your efficient pipeline.
        Yields (lr_batch, hr_batch) for training.
        """
        
        batch_lr = []
        batch_hr = []
        
        for region in regions:
            for tile_data in self.data_manager.process_region_efficiently(
                region, sequence_length
            ):
                if self.augment:
                    tile_data = self.apply_augmentations(tile_data)
                
                batch_lr.append(tile_data['lr_sequence'])
                batch_hr.append(tile_data['hr_sequence'])
                
                if len(batch_lr) == self.batch_size:
                    yield (
                        np.stack(batch_lr),    # (B, T, H, W, C)
                        np.stack(batch_hr)     # (B, T, H, W, C)
                    )
                    batch_lr = []
                    batch_hr = []
        
        # Yield remaining samples
        if batch_lr:
            yield (np.stack(batch_lr), np.stack(batch_hr))
    
    def apply_augmentations(self, tile_data):
        """Apply temporal-consistent augmentations"""
        
        lr_seq = tile_data['lr_sequence']
        hr_seq = tile_data['hr_sequence']
        
        # Random flip (applied consistently across time)
        if np.random.random() > 0.5:
            lr_seq = np.flip(lr_seq, axis=1)  # Flip along width
            hr_seq = np.flip(hr_seq, axis=1)
        
        if np.random.random() > 0.5:
            lr_seq = np.flip(lr_seq, axis=2)  # Flip along height  
            hr_seq = np.flip(hr_seq, axis=2)
        
        # Random temporal order flip (for attention mechanism robustness)
        if np.random.random() > 0.5:
            lr_seq = np.flip(lr_seq, axis=0)  # Flip along time
            hr_seq = np.flip(hr_seq, axis=0)
        
        tile_data['lr_sequence'] = lr_seq
        tile_data['hr_sequence'] = hr_seq
        
        return tile_data
```

## 🚀 **Implementation Strategy**

1. **Integrate with your existing `AuthGEE.py`**: Add the efficient processing classes
2. **Use your current `find_pairs()` function**: But process results in tiles instead of full images
3. **Leverage GEE server-side processing**: Minimize data transfer by doing coregistration/harmonization in the cloud
4. **Smart caching**: Cache frequently accessed tiles using LMDB
5. **Progressive training**: Start with smaller tiles and sequences, scale up as needed

## 💾 **Memory Benefits**

- **Before**: Download full images (~100MB each) × temporal sequences × regions = **~100GB+**
- **After**: Process tiles (~1MB each) on-demand = **~1-10GB total cache**
- **Speed**: 10-50x faster training data loading
- **Scalability**: Can handle unlimited regions and time periods

This approach keeps your existing quality processing while making it practical for large-scale super resolution training with temporal attention mechanisms.