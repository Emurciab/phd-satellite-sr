#!/bin/bash
# Monitor dataset generation progress

echo "═══════════════════════════════════════════════════════════════════"
echo "📊 STA-2 20-Sample Dataset Generation Monitor"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Check if process is running
PROCESS_COUNT=$(ps aux | grep '[p]ython.*generate_sta2' | wc -l)
if [ $PROCESS_COUNT -gt 0 ]; then
    echo "✅ Generation process: RUNNING ($PROCESS_COUNT process(es))"
    PID=$(ps aux | grep '[p]ython.*generate_sta2' | grep -v grep | head -1 | awk '{print $2}')
    echo "   PID: $PID"
    echo "   CPU: $(ps aux | grep $PID | grep -v grep | awk '{print $3}')%"
    echo "   MEM: $(ps aux | grep $PID | grep -v grep | awk '{print $4}')%"
    echo "   TIME: $(ps aux | grep $PID | grep -v grep | awk '{print $10}')"
else
    echo "❌ Generation process: NOT RUNNING"
fi

echo ""

# Check cache directory
CACHE_DIR="./cache/sta2_20samples_harmonized"
if [ -d "$CACHE_DIR" ]; then
    CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
    COREG_COUNT=$(find "$CACHE_DIR/coregistered" -name "*.npy" 2>/dev/null | wc -l)
    HARM_COUNT=$(find "$CACHE_DIR/harmonized" -name "*.npy" 2>/dev/null | wc -l)
    
    echo "📦 Cache Status:"
    echo "   Total size: $CACHE_SIZE"
    echo "   Coregistered files: $COREG_COUNT"
    echo "   Harmonized files: $HARM_COUNT ⭐"
else
    echo "📦 Cache: Not created yet"
fi

echo ""

# Check output directory
OUTPUT_DIR="./data/datasets/sta2_20samples_harmonized"
if [ -d "$OUTPUT_DIR" ]; then
    if [ -f "$OUTPUT_DIR/dataset.pkl" ]; then
        DATASET_SIZE=$(du -h "$OUTPUT_DIR/dataset.pkl" 2>/dev/null | cut -f1)
        echo "✅ Dataset: COMPLETE ($DATASET_SIZE)"
        echo "   Location: $OUTPUT_DIR/dataset.pkl"
        echo ""
        echo "🎨 Ready to visualize! Run:"
        echo "   conda activate tf_gpu && python visualize_20samples.py"
    else
        echo "⏳ Dataset: IN PROGRESS"
        echo "   Directory exists but dataset.pkl not yet created"
    fi
else
    echo "⏳ Dataset: NOT STARTED"
fi

echo ""

# Check log file
if [ -f "dataset_generation.log" ]; then
    LOG_SIZE=$(du -h dataset_generation.log 2>/dev/null | cut -f1)
    echo "📋 Log file: dataset_generation.log ($LOG_SIZE)"
    echo ""
    echo "Last 10 lines:"
    echo "───────────────────────────────────────────────────────────────────"
    tail -10 dataset_generation.log 2>/dev/null || echo "   (empty or not readable)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "💡 Tip: Run this script periodically to check progress:"
echo "   watch -n 30 bash monitor_generation.sh"
echo "═══════════════════════════════════════════════════════════════════"
