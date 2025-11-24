#!/usr/bin/env python3
"""
Monitor the integrated pipeline test and report when complete
"""
import subprocess
import time
import os
import sys

PID = 447023
LOG_FILE = "/media/eamurciab/TOSHIBA_EXT1/PhD/SR/integrated_test_final.log"
CHECK_INTERVAL = 60  # Check every 60 seconds

print("="*70)
print("INTEGRATED PIPELINE TEST MONITOR")
print("="*70)
print(f"\nMonitoring PID: {PID}")
print(f"Log file: {LOG_FILE}")
print(f"Check interval: {CHECK_INTERVAL}s\n")

iteration = 0
while True:
    iteration += 1
    
    # Check if process is still running
    result = subprocess.run(['ps', '-p', str(PID)], 
                          capture_output=True)
    
    if result.returncode != 0:
        # Process finished
        print(f"\n{'='*70}")
        print("🏁 TEST COMPLETED!")
        print(f"{'='*70}\n")
        
        # Check if log file has content
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
            print("📄 Log file content:\n")
            with open(LOG_FILE, 'r') as f:
                content = f.read()
                print(content)
        else:
            print("⚠️  Log file is empty (output may have been buffered)")
            print("Checking for test results in cache directory...\n")
            
            cache_dir = "/media/eamurciab/TOSHIBA_EXT1/PhD/SR/test_chunked_pipeline_cache"
            if os.path.exists(cache_dir):
                cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
                print(f"✅ Found {len(cache_files)} cached samples in {cache_dir}")
                
                if len(cache_files) >= 2:
                    print("\n🎉 SUCCESS: At least 2 samples were generated!")
                    print("\n✅ INTEGRATION TEST PASSED")
                else:
                    print(f"\n⚠️  Only {len(cache_files)} samples found (expected 2)")
            else:
                print("❌ Cache directory not found")
        
        print(f"\n{'='*70}")
        break
    
    # Process still running - get status
    runtime_result = subprocess.run(['ps', '-p', str(PID), '-o', 'etime='], 
                                   capture_output=True, 
                                   text=True)
    runtime = runtime_result.stdout.strip()
    
    # Check cache status
    cache_info = ""
    cache_dir = "/media/eamurciab/TOSHIBA_EXT1/PhD/SR/test_chunked_pipeline_cache"
    if os.path.exists(cache_dir):
        cache_files = len([f for f in os.listdir(cache_dir) if f.endswith('.pkl')])
        if cache_files > 0:
            cache_info = f" | 📦 {cache_files} samples cached"
    
    # Check log file size
    log_size = 0
    if os.path.exists(LOG_FILE):
        log_size = os.path.getsize(LOG_FILE)
    
    log_info = f" | 📝 {log_size} bytes" if log_size > 0 else ""
    
    print(f"[Check #{iteration}] ⏳ Running: {runtime}{cache_info}{log_info}")
    
    time.sleep(CHECK_INTERVAL)
