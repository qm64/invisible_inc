#!/bin/bash
# Run this from ~/git/qm/invisible_inc/gameplay_analyser

# First, copy the alarm_level_extractor.py to the current directory
echo "Setting up alarm level extractor..."
cp alarm_level_extractor.py . 2>/dev/null || echo "alarm_level_extractor.py should be in current directory"

# Run the batch test
echo ""
echo "Running batch test on captures/20251024_170049..."
echo ""
python batch_test_alarm.py captures/20251024_170049
