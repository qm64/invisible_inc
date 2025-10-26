#!/bin/bash
# Test structural_detector.py on representative frames from the capture session
#
# Usage: ./test_structural_detector.sh <path_to_captures_folder> <path_to_structural_detector.py>
# Example: ./test_structural_detector.sh ~/captures/20251022_201216 ~/structural_detector.py

CAPTURES_DIR="$1"
DETECTOR_SCRIPT="$2"

if [ -z "$CAPTURES_DIR" ] || [ -z "$DETECTOR_SCRIPT" ]; then
    echo "Usage: $0 <captures_dir> <structural_detector.py>"
    echo "Example: $0 ~/captures/20251022_201216 ~/structural_detector.py"
    exit 1
fi

if [ ! -d "$CAPTURES_DIR/frames" ]; then
    echo "Error: $CAPTURES_DIR/frames not found"
    exit 1
fi

if [ ! -f "$DETECTOR_SCRIPT" ]; then
    echo "Error: $DETECTOR_SCRIPT not found"
    exit 1
fi

# Create output directory
OUTPUT_DIR="$CAPTURES_DIR/structural_test_results"
mkdir -p "$OUTPUT_DIR"

echo "=================================="
echo "Testing Structural Detector"
echo "=================================="
echo "Captures: $CAPTURES_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Test frames representing different game states
TEST_FRAMES=(
    "frame_000007.png:player_normal:Planning phase (position 0/377)"
    "frame_000015.png:player_action:Agent action (profile hidden)"
    "frame_000031.png:opponent:Opponent turn (minimal UI)"
    "frame_000172.png:player_action:Agent action (profile hidden)"
    "frame_000223.png:player_normal:Planning phase (position 125/377)"
    "frame_000292.png:player_action:Agent action (profile hidden)"
    "frame_000351.png:player_normal:Planning phase (position 188/377)"
    "frame_000487.png:opponent:Opponent turn (minimal UI)"
    "frame_000540.png:opponent:Opponent turn (minimal UI)"
)

for test_case in "${TEST_FRAMES[@]}"; do
    IFS=':' read -r frame phase description <<< "$test_case"
    
    FRAME_PATH="$CAPTURES_DIR/frames/$frame"
    
    if [ ! -f "$FRAME_PATH" ]; then
        echo "⚠ Skipping $frame (not found)"
        continue
    fi
    
    echo "=================================="
    echo "Testing: $frame"
    echo "Phase: $phase"
    echo "Description: $description"
    echo "=================================="
    
    # Run detector
    python3 "$DETECTOR_SCRIPT" "$FRAME_PATH" > "$OUTPUT_DIR/${frame%.png}_output.txt" 2>&1
    
    # Copy visualization if created
    if [ -f "${FRAME_PATH%.png}_detected.png" ]; then
        cp "${FRAME_PATH%.png}_detected.png" "$OUTPUT_DIR/"
    fi
    
    if [ -f "${FRAME_PATH%.png}_detected.json" ]; then
        cp "${FRAME_PATH%.png}_detected.json" "$OUTPUT_DIR/"
    fi
    
    echo "✓ Results saved to $OUTPUT_DIR/${frame%.png}_*"
    echo ""
done

echo "=================================="
echo "Test Complete!"
echo "=================================="
echo "Results in: $OUTPUT_DIR"
echo ""
echo "Files created:"
ls -lh "$OUTPUT_DIR" | tail -n +2

echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/frame_000007_output.txt"
echo "  open $OUTPUT_DIR/frame_000007_detected.png"