#!/bin/bash
# Monitor integrated pipeline test progress

echo "==================================================================="
echo "INTEGRATED PIPELINE TEST MONITOR"
echo "==================================================================="
echo ""

LOG_FILE="/media/eamurciab/TOSHIBA_EXT1/PhD/SR/integrated_test_output.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found. Test may not have started yet."
    exit 1
fi

# Check if process is still running
if ps aux | grep -v grep | grep "test_integrated_pipeline.py" > /dev/null; then
    echo "✅ Test is RUNNING"
else
    echo "⚠️  Test process not found (may have completed or failed)"
fi

echo ""
echo "-------------------------------------------------------------------"
echo "LATEST OUTPUT (last 30 lines):"
echo "-------------------------------------------------------------------"
tail -30 "$LOG_FILE"

echo ""
echo "-------------------------------------------------------------------"
echo "STATISTICS:"
echo "-------------------------------------------------------------------"
echo "Total lines: $(wc -l < "$LOG_FILE")"
echo "File size: $(ls -lh "$LOG_FILE" | awk '{print $5}')"

# Check for key indicators
if grep -q "SUCCESS! Generated" "$LOG_FILE"; then
    echo "✅ Test completed successfully!"
elif grep -q "ERROR" "$LOG_FILE"; then
    echo "❌ Errors detected in log"
elif grep -q "Downloading" "$LOG_FILE"; then
    echo "📥 Currently downloading data..."
elif grep -q "Processing" "$LOG_FILE"; then
    echo "⚙️  Processing data..."
fi

echo ""
echo "-------------------------------------------------------------------"
echo "To view full log: cat $LOG_FILE"
echo "To follow live: tail -f $LOG_FILE"
echo "==================================================================="
