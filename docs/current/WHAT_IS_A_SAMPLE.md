# What is a "Sample"?

## Definition

A **sample** is ONE complete training example consisting of:

```python
@dataclass
class TrainingSample:
    lr_sequence: np.ndarray  # (T, H, W, C) - Low Resolution (Sentinel-2)
    hr_sequence: np.ndarray  # (T, H, W, C) - High Resolution (Planet)
    temporal_dates: List[str]
    region_name: str
    sequence_id: int
    tile_id: int
    tile_bounds: Dict
```

## Breakdown

### 1. **Spatial Component** (Tile)
- A **tile** is a spatial crop from the satellite image
- Size: e.g., 512×512 pixels
- Covers a specific geographic area (a few km²)

### 2. **Temporal Component** (Sequence)
- A **sequence** is multiple dates of the same location
- Length: e.g., 3 temporal frames
- Example: [Jan 1, Jan 6, Apr 15]

### 3. **Resolution Pairs**
- **LR (Low Resolution)**: Sentinel-2 data
  - Shape: `(3, 64, 64, 4)` = 3 dates, 64×64 pixels, 4 bands
  - Spatial resolution: ~10-20m per pixel
  
- **HR (High Resolution)**: Planet data
  - Shape: `(3, 256, 256, 4)` = 3 dates, 256×256 pixels, 4 bands
  - Spatial resolution: ~3m per pixel
  - **4x better** than LR

## Concrete Example

**Sample #1:**
```
Region: STA-2
Tile: Spatial crop #5 (centered at lat=26.4, lon=-80.6)
Sequence: Temporal frames [2020-01-01, 2020-01-06, 2020-04-15]

Data:
  LR (Sentinel-2): array with shape (3, 64, 64, 4)
    - Date 0: 64×64 image with 4 bands (Blue, Green, Red, NIR)
    - Date 1: 64×64 image with 4 bands
    - Date 2: 64×64 image with 4 bands
    
  HR (Planet): array with shape (3, 256, 256, 4)
    - Date 0: 256×256 image with 4 bands (Blue, Green, Red, NIR)
    - Date 1: 256×256 image with 4 bands
    - Date 2: 256×256 image with 4 bands

Memory: ~3.2 MB (uncompressed)
```

## How Samples Are Created

### Step 1: Region
Start with a study area (e.g., STA-2 in Florida)

### Step 2: Temporal Sequences
From matched satellite pairs:
```
Matched pairs: [Jan 1, Jan 6, Jan 11, Jan 16, Apr 15, ...]

Create sequences of length 3:
  Sequence 0: [Jan 1, Jan 6, Jan 11]
  Sequence 1: [Jan 6, Jan 11, Jan 16]
  Sequence 2: [Jan 11, Jan 16, Apr 15]
  ...
```

### Step 3: Spatial Tiles
Split each region into tiles:
```
Region: 20km × 20km
Tile size: 512 pixels (at Planet resolution)

Grid of tiles:
  [Tile 0] [Tile 1] [Tile 2] ...
  [Tile 5] [Tile 6] [Tile 7] ...
  ...
```

### Step 4: Combine
Every combination of sequence + tile = 1 sample:
```
Sequence 0 + Tile 0 = Sample 0
Sequence 0 + Tile 1 = Sample 1
Sequence 0 + Tile 2 = Sample 2
...
Sequence 1 + Tile 0 = Sample 25
Sequence 1 + Tile 1 = Sample 26
...
```

## Your Current Dataset

From the output:
```
🔄 Processing STA-2 (NumPy pipeline)...
   📅 Created 10 temporal sequences
   🗺️ Created 25 spatial tiles
   ✅ Collected 50 samples...
```

**Calculation:**
- 10 sequences × 25 tiles = **250 possible samples**
- But you limited to **50 samples** for quick testing

**Each sample:**
- LR: `(3, 64, 64, 4)` = 3 dates × 64×64 pixels × 4 bands × 4 bytes = **196,608 bytes** (~0.19 MB)
- HR: `(3, 256, 256, 4)` = 3 dates × 256×256 pixels × 4 bands × 4 bytes = **3,145,728 bytes** (~3.0 MB)
- **Total per sample: ~3.2 MB**

**50 samples:**
- Total: 50 × 3.2 MB = **~160 MB** ✅ (matches your output: 159.4 MB)

## Training Usage

During training:
- **Batch size = 4** means 4 samples processed together
- Model sees: `(4, 3, 4, 64, 64)` for LR, `(4, 3, 4, 256, 256)` for HR
- After 50 samples ÷ 4 per batch = **12.5 batches per epoch**

## Why This Matters

### More Samples = Better Training
- **50 samples**: Quick testing, proves it works
- **200 samples**: Better model, more variety
- **1,000 samples**: Production quality
- **6,000+ samples**: Full dataset (all regions, all tiles)

### Sample Diversity
Each sample provides:
- ✅ Different geographic locations (spatial diversity)
- ✅ Different dates/seasons (temporal diversity)
- ✅ Different land covers (content diversity)

## Summary

**1 Sample = 1 (spatial tile, temporal sequence) pair**

Contains:
- 3 dates of low-res Sentinel-2 images
- 3 dates of high-res Planet images (same dates, same location)
- Training the model to learn: LR → HR transformation

Your 50 samples = 50 different spatiotemporal locations for the model to learn from!
