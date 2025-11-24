# BUG FIX: LR-HR Date Mismatch

## 🐛 Bug Description

**Critical Issue:** LR (Sentinel-2) and HR (Planet) images were mislabeled with the same date, even though they were acquired on different dates.

### What Was Wrong

The `find_pairs()` function in `Auth/AuthGEE.py` matches Planet and Sentinel-2 images with **±7 days tolerance**. This means:
- Planet image might be from **2020-01-06**
- Sentinel-2 image might be from **2020-01-01**
- Both get paired because they're within 7 days

However, the generation code was only storing the **S2 date** for both images:

```python
# OLD CODE (BUGGY):
dates.append(pair['s2_date'])  # ❌ Only S2 date stored!
```

This meant visualizations showed:
- **LR from 2020-01-01** (correct)
- **HR from 2020-01-01** (WRONG! Actually from 2020-01-06)

### Example from Logs

**Sequence 1, Frame 1:**
- Planet ID: `20200106_153950` = **January 6, 2020**
- S2 ID: `20200101T160511` = **January 1, 2020**
- Stored date: `2020-01-01` (only S2 date)
- Result: HR labeled as Jan 1 when it's actually Jan 6 ❌

## ✅ Fix Applied

### Code Changes

**1. Updated TrainingSample dataclass** (`generate_sta2_grid_parallel.py:28-36`):
```python
@dataclass
class TrainingSample:
    lr_sequence: np.ndarray  # (T, H, W, C)
    hr_sequence: np.ndarray  # (T, H, W, C)
    temporal_dates: List[str]  # Sentinel-2 dates
    planet_dates: List[str]  # Planet dates (may differ by ±7 days) ← NEW!
    region_name: str
    sequence_id: int
    tile_id: int
    tile_bounds: Dict
```

**2. Store both dates** (`generate_sta2_grid_parallel.py:385`):
```python
# NEW CODE (FIXED):
dates.append((pair['s2_date'], pair['planet_date']))  # ✅ Both dates stored!
```

**3. Separate dates when creating sample** (`generate_sta2_grid_parallel.py:393-405`):
```python
# Separate S2 and Planet dates
s2_dates = [d[0] for d in dates]
planet_dates = [d[1] for d in dates]

sample = TrainingSample(
    lr_sequence=np.stack(lr_frames, axis=0),
    hr_sequence=np.stack(hr_frames, axis=0),
    temporal_dates=s2_dates,      # LR dates
    planet_dates=planet_dates,     # HR dates ← NEW!
    ...
)
```

**4. Updated visualizations** to show both dates:
- `visualize_sequences.py`: Shows "S2:2020-01-01/PL:2020-01-06" in title
- `visualize_sequences_grid.py`: Labels columns as "LR-T1 S2:date" and "HR-T1 PL:date"

## 📊 Impact

### Old Dataset (400 samples)
- ❌ **INVALID**: All samples have incorrect date labels
- ❌ Cannot be used for training (dates don't match actual acquisition)
- ❌ Must be regenerated with fixed code

### New Dataset (after fix)
- ✅ Correct date labels for both LR and HR
- ✅ Visualizations clearly show both dates
- ✅ Can verify temporal alignment visually

## 🔄 Next Steps

**Required Action:** Regenerate the 400 samples with the fixed code

```bash
# Regenerate with correct date pairing
python scripts/preprocessing/generate_sta2_grid_parallel.py \
    --max-samples 400 \
    --coverage-threshold 90 \
    --coverage-workers 15
```

**Expected Time:** ~14 hours (same as before)

## 🔍 Verification

After regeneration, verify the fix:

```python
# Check that dates are stored correctly
import pickle
with open('data/datasets/sta2_grid_parallel/dataset.pkl', 'rb') as f:
    data = pickle.load(f)

sample = data['samples'][0]
print(f"LR dates (S2): {sample.temporal_dates}")
print(f"HR dates (Planet): {sample.planet_dates}")

# They should differ by a few days (within ±7 days tolerance)
```

Visualizations will now show:
- **LR (S2) Frame 1: 2020-01-01**
- **HR (Planet) Frame 1: 2020-01-06**

This correctly represents the actual acquisition dates!

## 📝 Notes

- The ±7 day tolerance is intentional (Planet and S2 have different revisit times)
- Coregistration and harmonization handle the temporal differences
- For super-resolution training, the model needs to learn this temporal variability
- Always verify dates match between LR and HR in visualizations

---

**Bug discovered:** November 23, 2025
**Fix applied:** November 23, 2025
**Status:** Fixed, requires data regeneration
