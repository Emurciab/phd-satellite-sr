"""
Efficient GEE download using Export API + Drive
This is the production-standard approach for large-scale data
"""
import ee
import time
import os
from pathlib import Path

def export_and_download_image(image, region, filename, scale=3, drive_folder='SR_exports', local_dir='./gee_downloads'):
    """
    Export image from GEE to Drive, then download locally.
    
    Args:
        image: ee.Image to export
        region: ee.Geometry region
        filename: Base filename (without extension)
        scale: Resolution in meters
        drive_folder: Google Drive folder name
        local_dir: Local directory for downloads
        
    Returns:
        Local file path or None if failed
    """
    
    local_path = Path(local_dir) / f"{filename}.tif"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if local_path.exists():
        print(f"   ✓ Using cached: {local_path.name}")
        return str(local_path)
    
    print(f"   🚀 Exporting {filename} to Drive...", end=' ', flush=True)
    
    # Start export task
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=filename,
        folder=drive_folder,
        fileNamePrefix=filename,
        region=region,
        scale=scale,
        maxPixels=1e9,
        fileFormat='GeoTIFF'
    )
    
    task.start()
    
    # Wait for task to complete
    start_time = time.time()
    while task.active():
        time.sleep(5)
        elapsed = time.time() - start_time
        if elapsed > 600:  # 10 minute timeout
            print(f"⏱️ Timeout")
            return None
    
    status = task.status()
    if status['state'] == 'COMPLETED':
        elapsed = time.time() - start_time
        print(f"✓ ({elapsed:.0f}s)")
        
        # TODO: Download from Drive to local_path
        # For now, return the drive location marker
        return f"DRIVE:{drive_folder}/{filename}.tif"
    else:
        print(f"❌ Failed: {status.get('error_message', 'Unknown error')}")
        return None

# This approach preserves ALL preprocessing but avoids GEE download limits
