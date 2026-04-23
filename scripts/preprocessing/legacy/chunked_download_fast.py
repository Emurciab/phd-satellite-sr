"""
Fast chunked download with parallel requests
"""
import ssl

# Fix SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context

import ee
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_chunk(image, chunk_geom, bands, scale):
    """Download a single chunk"""
    try:
        # unmask(0) converts masked pixels to 0 so we get actual arrays instead of (1,1) shape
        # reproject() forces sampling at specific resolution (scale in meters)
        chunk_img = image.select(bands).clip(chunk_geom).unmask(0)
        
        # Force reprojection to EPSG:4326 at specified scale (meters)
        chunk_img = chunk_img.reproject(crs='EPSG:4326', scale=scale)
        
        chunk_sample = chunk_img.sampleRectangle(region=chunk_geom, defaultValue=0)
        chunk_data = chunk_sample.getInfo()
        
        if chunk_data and 'properties' in chunk_data:
            props = chunk_data['properties']
            band_arrays = [np.array(props[b], dtype=np.float32) for b in bands]
            return np.stack(band_arrays, axis=-1)  # (H, W, C)
        return None
    except Exception as e:
        print(f"Chunk error: {e}")
        return None

def download_image_chunked_parallel(image, aoi, bands, scale=3, chunk_grid_size=9, max_workers=12):
    """
    Download image using parallel chunk downloads (FAST!)
    
    Args:
        image: ee.Image
        aoi: ee.Geometry 
        bands: List of band names
        scale: Resolution in meters
        chunk_grid_size: Split into NxN grid (9x9 = 81 chunks keeps each under 262k pixels)
        max_workers: Parallel downloads (12 = optimal for high-spec systems, 6 = good for most systems)
    
    Returns:
        numpy array (H, W, C) or None
    """
    print(f"   Downloading {chunk_grid_size}x{chunk_grid_size} chunks in parallel...", end=' ', flush=True)
    
    # Get AOI bounds
    bounds = aoi.bounds().getInfo()['coordinates'][0]
    lons = [p[0] for p in bounds]
    lats = [p[1] for p in bounds]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    lon_step = (max_lon - min_lon) / chunk_grid_size
    lat_step = (max_lat - min_lat) / chunk_grid_size
    
    # Create all chunk geometries
    chunks = []
    for i in range(chunk_grid_size):
        for j in range(chunk_grid_size):
            chunk_min_lon = min_lon + j * lon_step
            chunk_max_lon = min_lon + (j + 1) * lon_step
            chunk_min_lat = max_lat - (i + 1) * lat_step
            chunk_max_lat = max_lat - i * lat_step
            
            chunk_geom = ee.Geometry.Rectangle([
                chunk_min_lon, chunk_min_lat,
                chunk_max_lon, chunk_max_lat
            ])
            chunks.append((i, j, chunk_geom))
    
    # Download all chunks in parallel
    chunk_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_chunk, image, geom, bands, scale): (i, j)
            for i, j, geom in chunks
        }
        
        completed = 0
        for future in as_completed(futures):
            i, j = futures[future]
            chunk_results[(i, j)] = future.result()
            completed += 1
            if completed % 10 == 0:  # Update every 10 chunks
                print(f"{completed}/{len(chunks)}", end=' ', flush=True)
    
    print(f"✓")
    
    # Stitch chunks
    print(f"   Stitching...", end=' ', flush=True)
    
    # Find expected chunk size from first valid chunk
    expected_h, expected_w = None, None
    for pos, chunk in chunk_results.items():
        if chunk is not None and chunk.size > 0:
            expected_h, expected_w = chunk.shape[:2]
            break
    
    if expected_h is None:
        print(f"❌ All chunks empty!")
        return None
    
    # Build grid
    grid = []
    for i in range(chunk_grid_size):
        row = []
        for j in range(chunk_grid_size):
            chunk = chunk_results.get((i, j))
            if chunk is None or chunk.size == 0:
                # Fill with zeros
                chunk = np.zeros((expected_h, expected_w, len(bands)), dtype=np.float32)
            row.append(chunk)
        grid.append(np.concatenate(row, axis=1))  # Horizontal concat
    
    result = np.concatenate(grid, axis=0)  # Vertical concat
    print(f"✓ {result.shape}")
    
    return result
