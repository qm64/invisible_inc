# Holistic Game State Analysis Tools

Version 1.0.0

Tools for batch analyzing 15k+ gameplay screenshots to discover patterns in UI elements and game states through data-driven analysis rather than hand-crafted detectors.

## Quick Start

```bash
# 1. Install dependencies
pip install opencv-python numpy matplotlib --break-system-packages

# 2. Create frame list
find captures -type f -name "frame_*.png" > frame_list.txt

# 3. Run batch scanner (5-10 minutes for 15k frames)
python batch_scanner.py -f frame_list.txt

# 4. Generate visualizations
python generate_reports.py

# 5. Open reports/summary.html in browser
```

## Workflow

### Phase 1: Data Collection (batch_scanner.py)

Processes thousands of frames in parallel and extracts:
- **Frame classification**: game vs non-game windows
- **Color signatures**: HSV histograms showing which colors exist
- **Spatial data**: 10×10 grid showing where colors appear
- **Structural metrics**: edge density showing UI complexity

Output: `analysis.db` (SQLite database)

### Phase 2: Pattern Discovery (generate_reports.py)

Analyzes the database to reveal:
- **Color distribution**: which hues are distinctive vs common
- **Spatial heatmaps**: where UI elements consistently appear
- **Element presence statistics**: how often cyan/yellow/red appear

Output: `reports/` directory with visualizations

### Phase 3: Custom Analysis (SQL queries)

Query the database directly for deeper insights:

```sql
-- Find frames with END TURN button (cyan) but no agent AP (yellow)
SELECT path FROM frames 
WHERE has_cyan = 1 AND has_yellow = 0 AND is_game_frame = 1;

-- Which grid cells consistently have cyan UI?
SELECT grid_x, grid_y, 
       SUM(has_cyan) * 1.0 / COUNT(*) as cyan_frequency
FROM spatial_data sd
JOIN frames f ON sd.path = f.path
WHERE f.is_game_frame = 1
GROUP BY grid_x, grid_y
HAVING cyan_frequency > 0.9;

-- Element co-occurrence analysis
SELECT 
    has_cyan, has_yellow, has_red, 
    COUNT(*) as frame_count
FROM frames 
WHERE is_game_frame = 1
GROUP BY has_cyan, has_yellow, has_red
ORDER BY frame_count DESC;
```

## Input Methods

```bash
# From file list
python batch_scanner.py -f frame_list.txt

# From stdin (pipe from find)
find captures -name "frame_*.png" | python batch_scanner.py -f -

# Direct arguments
python batch_scanner.py frame_001.png frame_002.png

# Combined
python batch_scanner.py -f list.txt extra_frame.png

# More workers for faster processing
python batch_scanner.py -f frame_list.txt -w 12
```

## Incremental Updates

The scanner automatically skips frames that have already been processed:

```bash
# Initial scan (processes all 15k frames)
python batch_scanner.py -f frame_list.txt

# Later, after capturing more gameplay
find captures -name "frame_*.png" | python batch_scanner.py -f -
# Only processes new frames (takes seconds)

# Force reprocess everything (if you change extraction logic)
python batch_scanner.py -f frame_list.txt --force
```

## Understanding the Output

### Classification Statistics
- Shows game vs non-game frame ratio
- Reveals how often UI colors appear
- Helps validate that your captures are mostly game windows

### Color Distribution
- **Hue plot**: peaks indicate distinctive colors
  - 85-100° = cyan (END TURN, UI elements)
  - 20-30° = yellow (text, AP values)
  - 0-10° = red (alarms, alerts)
- **Saturation/Value**: understand color intensity

### Spatial Heatmaps
- **Bright areas** = consistent UI element locations (good anchors)
- **Dark areas** = dynamic content or mode-specific elements
- **Edge density** = structural complexity (where UI borders are)

## Expected Insights

This analysis should reveal:

1. **Which UI elements are actually reliable**
   - Your END TURN button (cyan 85-100°) should show high presence
   - Spatial heatmap should show cyan concentrated in lower-right

2. **Natural game state groupings**
   - Frames with cyan + yellow = player planning phase?
   - Frames with cyan only = opponent turn?
   - Element co-occurrence patterns reveal mode structure

3. **Why current detectors fail 20-25% of the time**
   - Maybe failures cluster in specific resolutions?
   - Maybe certain game states lack expected colors?
   - Spatial analysis shows if search regions are wrong

4. **Better detection strategies**
   - Which elements have highest "information value"
   - Minimum element set needed for robust state detection
   - Optimal anchor hierarchy based on actual reliability

## Database Schema

```sql
-- Per-frame metrics
frames (
    path TEXT PRIMARY KEY,
    is_game_frame INTEGER,
    width, height, aspect_ratio,
    mean_brightness,
    has_cyan, has_yellow, has_red,
    edge_density
)

-- HSV histogram data (36 hue × 3 sat × 3 val bins)
color_signatures (
    path TEXT,
    hue_bin INTEGER,  -- 0-35 (multiply by 5 for degrees)
    sat_bin INTEGER,  -- 0=low, 1=med, 2=high
    val_bin INTEGER,  -- 0=dark, 1=med, 2=bright
    pixel_count INTEGER
)

-- Spatial distribution (10×10 grid)
spatial_data (
    path TEXT,
    grid_x INTEGER,   -- 0-9 (left to right)
    grid_y INTEGER,   -- 0-9 (top to bottom)
    has_cyan, has_yellow, has_orange, has_red, has_white,
    edge_density
)
```

## Next Steps After Analysis

Based on the heatmaps and statistics, you can:

1. **Build element presence matrix**: create binary vectors for clustering
2. **Temporal analysis**: how do patterns change frame-to-frame?
3. **Failure mode analysis**: run existing detectors on all frames, study the 20-25% failures
4. **Information theory approach**: rank elements by discriminative power
5. **Derive turn detection**: test if element combinations reliably indicate turns

## Performance

- **Scanning**: ~100-200 fps (depending on CPU cores)
- **15k frames**: 5-10 minutes
- **Report generation**: <30 seconds
- **Database size**: ~5-10 MB per 1000 frames

## Troubleshooting

**"No valid frame paths provided"**
- Check that frame_list.txt contains valid paths
- Verify files exist with: `head -5 frame_list.txt | xargs ls`

**"Database not found"**
- Run batch_scanner.py first to create analysis.db

**Reports show mostly non-game frames**
- Adjust classification heuristics in batch_scanner.py
- The defaults assume aspect ratio 1.5-2.5, brightness 20-80

**Processing is slow**
- Increase workers: `-w 16`
- Check CPU usage - may be I/O bound on slow disks
