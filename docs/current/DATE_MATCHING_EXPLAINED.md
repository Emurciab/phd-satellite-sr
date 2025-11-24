# Date Matching: What Happens When Dates Don't Match Exactly?

## The Matching Algorithm

From your `AuthGEE.py`, the function `_match_dates_with_tolerance_nn()` does **nearest-neighbor matching within tolerance**:

```python
def _match_dates_with_tolerance_nn(s2_map, pl_map, days_tol=7):
    # For each Sentinel-2 date:
    #   1. Find all Planet dates within ±7 days
    #   2. Pick the CLOSEST one (minimum days difference)
    #   3. Each Planet date can only be used ONCE
```

## Example Scenario

### Input Data
```
Sentinel-2 dates:  [2020-01-01, 2020-01-06, 2020-01-11, 2020-04-15]
Planet dates:      [2020-01-03, 2020-01-08, 2020-04-13]
Tolerance: ±7 days
```

### Matching Process

**Step 1: Match S2 2020-01-01**
```
S2 date: 2020-01-01
Candidates within ±7 days:
  - 2020-01-03 (diff: 2 days) ✅ CLOSEST
  - 2020-01-08 (diff: 7 days)

Match: (S2: 2020-01-01, Planet: 2020-01-03)
Used Planet dates: [2020-01-03]
```

**Step 2: Match S2 2020-01-06**
```
S2 date: 2020-01-06
Candidates within ±7 days:
  - 2020-01-03 (already used ❌)
  - 2020-01-08 (diff: 2 days) ✅ CLOSEST

Match: (S2: 2020-01-06, Planet: 2020-01-08)
Used Planet dates: [2020-01-03, 2020-01-08]
```

**Step 3: Match S2 2020-01-11**
```
S2 date: 2020-01-11
Candidates within ±7 days:
  - 2020-01-08 (already used ❌)
  
No match! ❌
```

**Step 4: Match S2 2020-04-15**
```
S2 date: 2020-04-15
Candidates within ±7 days:
  - 2020-04-13 (diff: 2 days) ✅ CLOSEST

Match: (S2: 2020-04-15, Planet: 2020-04-13)
Used Planet dates: [2020-01-03, 2020-01-08, 2020-04-13]
```

### Final Result
```python
matched_pairs = [
    {'s2_date': '2020-01-01', 'planet_date': '2020-01-03', ...},  # 2 days apart
    {'s2_date': '2020-01-06', 'planet_date': '2020-01-08', ...},  # 2 days apart
    {'s2_date': '2020-04-15', 'planet_date': '2020-04-13', ...},  # 2 days apart
]

s2_only = ['2020-01-11']  # No Planet match within ±7 days
planet_only = []          # All Planet dates were used
```

## What Date Gets Stored in the Sample?

Each sample stores **BOTH dates**:

```python
@dataclass
class TrainingSample:
    lr_sequence: np.ndarray       # Sentinel-2 images
    hr_sequence: np.ndarray       # Planet images
    temporal_dates: List[str]     # ← BOTH S2 AND PLANET DATES!
    region_name: str
    sequence_id: int
    tile_id: int
    tile_bounds: Dict
```

### In Your Case

From your actual output:
```
[matcher] first S2 2020-01-01 has 2 PL candidates within ±7d
[matcher] nn result: 12 matched, 10 PL-only, 11 S2-only
```

This means:
- **12 pairs matched** (within ±7 days tolerance)
- **10 Planet-only** dates had no S2 within range
- **11 S2-only** dates had no Planet within range

### Example Matched Pair Storage

Let's say you have:
```python
matched_pair = {
    's2_date': '2020-01-01',
    'planet_date': '2020-01-03',  # 2 days later
    's2_id': 'COPERNICUS/S2_SR_HARMONIZED/20200101T160511_...',
    'planet_id': 'projects/.../20200103_153950_...'
}
```

When creating a sample, the `temporal_dates` field stores:
```python
sample.temporal_dates = ['2020-01-01', '2020-01-06', '2020-04-15']  
# These are the S2 dates (reference dates)
```

**But the actual images come from both:**
- `lr_sequence` contains S2 images from: 2020-01-01, 2020-01-06, 2020-04-15
- `hr_sequence` contains Planet images from: 2020-01-03, 2020-01-08, 2020-04-13

## Why This Matters for Training

### Temporal Alignment
The model learns to super-resolve images that are:
- ✅ Spatially aligned (same location)
- ✅ Temporally aligned (±7 days, typically 0-3 days apart)
- ⚠️ **Not necessarily from the exact same moment**

### Real-World Scenario
This is actually **realistic** because:
1. Satellites don't pass over the same location at the same time
2. Clouds might obscure one satellite but not the other on the same day
3. In practice, you often need to match "closest available" dates

### Model Robustness
The small temporal offset (±7 days) helps the model:
- ✅ Be more robust to temporal variations
- ✅ Handle seasonal changes
- ✅ Work with imperfect data (real-world scenario)

## Summary

**Q: What date registers the sample when dates don't match exactly?**

**A: BOTH dates are stored:**
- **Reference date** (S2 date): Used for `temporal_dates` field
- **Actual acquisition dates**: 
  - LR images from S2 dates
  - HR images from matched Planet dates (within ±7 days)

**Example from your output:**
```python
Sample with sequence: ['2020-01-01', '2020-01-06', '2020-04-15']

Actual data:
  - LR[0]: S2 from 2020-01-01
  - HR[0]: Planet from 2020-01-03 (2 days later)
  
  - LR[1]: S2 from 2020-01-06  
  - HR[1]: Planet from 2020-01-08 (2 days later)
  
  - LR[2]: S2 from 2020-04-15
  - HR[2]: Planet from 2020-04-13 (2 days earlier)
```

The model learns: "Given LR from date X, produce HR that looks like Planet from date X±δ"
where δ is typically 0-3 days in your dataset.
