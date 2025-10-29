#!/usr/bin/env python3
"""
Report Generator for Batch Frame Analysis
Version 1.0.0

Generates visualizations and statistics from the batch scanner database.
"""

import sqlite3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List, Tuple
import json

# ============================================================================
# DATABASE QUERIES
# ============================================================================

def load_frame_stats(conn: sqlite3.Connection) -> Dict:
    """Load overall frame statistics."""
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM frames")
    total_frames = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM frames WHERE is_game_frame = 1")
    game_frames = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(mean_brightness) FROM frames WHERE is_game_frame = 1")
    avg_brightness = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(aspect_ratio) FROM frames WHERE is_game_frame = 1")
    avg_aspect = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM frames WHERE has_cyan = 1 AND is_game_frame = 1")
    cyan_frames = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM frames WHERE has_yellow = 1 AND is_game_frame = 1")
    yellow_frames = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM frames WHERE has_red = 1 AND is_game_frame = 1")
    red_frames = cursor.fetchone()[0]
    
    return {
        'total_frames': total_frames,
        'game_frames': game_frames,
        'non_game_frames': total_frames - game_frames,
        'avg_brightness': avg_brightness,
        'avg_aspect_ratio': avg_aspect,
        'cyan_presence': cyan_frames / game_frames if game_frames > 0 else 0,
        'yellow_presence': yellow_frames / game_frames if game_frames > 0 else 0,
        'red_presence': red_frames / game_frames if game_frames > 0 else 0
    }

def load_color_histogram(conn: sqlite3.Connection) -> np.ndarray:
    """Load aggregated HSV histogram from all game frames."""
    cursor = conn.cursor()
    
    # Get histogram data only from game frames
    cursor.execute("""
        SELECT cs.hue_bin, cs.sat_bin, cs.val_bin, SUM(cs.pixel_count)
        FROM color_signatures cs
        JOIN frames f ON cs.path = f.path
        WHERE f.is_game_frame = 1
        GROUP BY cs.hue_bin, cs.sat_bin, cs.val_bin
    """)
    
    hist = np.zeros((36, 3, 3))
    for h, s, v, count in cursor.fetchall():
        hist[h, s, v] = count
    
    return hist

def load_spatial_heatmap(conn: sqlite3.Connection, color: str) -> np.ndarray:
    """Load spatial heatmap for a specific color."""
    cursor = conn.cursor()
    
    color_column = f'has_{color}'
    
    cursor.execute(f"""
        SELECT sd.grid_x, sd.grid_y, 
               SUM(sd.{color_column}) * 1.0 / COUNT(*) as frequency
        FROM spatial_data sd
        JOIN frames f ON sd.path = f.path
        WHERE f.is_game_frame = 1
        GROUP BY sd.grid_x, sd.grid_y
    """)
    
    heatmap = np.zeros((10, 10))
    for x, y, freq in cursor.fetchall():
        heatmap[y, x] = freq
    
    return heatmap

def load_edge_density_heatmap(conn: sqlite3.Connection) -> np.ndarray:
    """Load spatial heatmap of edge density."""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT sd.grid_x, sd.grid_y, AVG(sd.edge_density)
        FROM spatial_data sd
        JOIN frames f ON sd.path = f.path
        WHERE f.is_game_frame = 1
        GROUP BY sd.grid_x, sd.grid_y
    """)
    
    heatmap = np.zeros((10, 10))
    for x, y, density in cursor.fetchall():
        heatmap[y, x] = density
    
    return heatmap

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_frame_classification(stats: Dict, output_dir: Path):
    """Plot frame classification statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart: game vs non-game
    ax = axes[0]
    sizes = [stats['game_frames'], stats['non_game_frames']]
    labels = [f"Game Frames\n{stats['game_frames']}", 
              f"Non-Game\n{stats['non_game_frames']}"]
    colors = ['#2ecc71', '#e74c3c']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Frame Classification', fontsize=14, fontweight='bold')
    
    # Bar chart: color presence
    ax = axes[1]
    colors_data = {
        'Cyan': stats['cyan_presence'],
        'Yellow': stats['yellow_presence'],
        'Red': stats['red_presence']
    }
    bars = ax.bar(colors_data.keys(), colors_data.values(), 
                   color=['cyan', 'yellow', 'red'])
    ax.set_ylabel('Presence Frequency', fontsize=12)
    ax.set_title('UI Color Presence in Game Frames', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'classification_stats.png', dpi=150)
    plt.close()

def plot_hsv_distribution(hist: np.ndarray, output_dir: Path):
    """Plot HSV color distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Hue distribution (sum across S and V)
    ax = axes[0, 0]
    hue_dist = np.sum(hist, axis=(1, 2))
    hue_angles = np.arange(36) * 5  # Convert bins to degrees
    ax.bar(hue_angles, hue_dist, width=4, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Hue (degrees)', fontsize=12)
    ax.set_ylabel('Pixel Count', fontsize=12)
    ax.set_title('Hue Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Mark key color regions
    ax.axvspan(85, 100, alpha=0.2, color='cyan', label='Cyan (END TURN)')
    ax.axvspan(20, 30, alpha=0.2, color='yellow', label='Yellow')
    ax.axvspan(0, 10, alpha=0.2, color='red', label='Red (Alarm)')
    ax.legend()
    
    # Saturation distribution
    ax = axes[0, 1]
    sat_dist = np.sum(hist, axis=(0, 2))
    sat_labels = ['Low', 'Med', 'High']
    ax.bar(sat_labels, sat_dist, color='coral', edgecolor='black')
    ax.set_ylabel('Pixel Count', fontsize=12)
    ax.set_title('Saturation Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Value distribution
    ax = axes[1, 0]
    val_dist = np.sum(hist, axis=(0, 1))
    val_labels = ['Dark', 'Med', 'Bright']
    ax.bar(val_labels, val_dist, color='lightgreen', edgecolor='black')
    ax.set_ylabel('Pixel Count', fontsize=12)
    ax.set_title('Value/Brightness Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 2D Hue vs Saturation (averaged over Value)
    ax = axes[1, 1]
    hs_map = np.sum(hist, axis=2).T  # Sum over V dimension
    im = ax.imshow(hs_map, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xlabel('Hue Bin', fontsize=12)
    ax.set_ylabel('Saturation Bin', fontsize=12)
    ax.set_title('Hue vs Saturation Heatmap', fontsize=14, fontweight='bold')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Low', 'Med', 'High'])
    plt.colorbar(im, ax=ax, label='Pixel Count')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'color_distribution.png', dpi=150)
    plt.close()

def plot_spatial_heatmap(heatmap: np.ndarray, title: str, output_path: Path, 
                         cmap: str = 'hot'):
    """Plot a single spatial heatmap."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(heatmap, cmap=cmap, vmin=0, vmax=1, origin='upper')
    ax.set_xlabel('Grid X', fontsize=12)
    ax.set_ylabel('Grid Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Presence Frequency', fontsize=12)
    
    # Annotate cells with values
    for i in range(10):
        for j in range(10):
            value = heatmap[i, j]
            if value > 0.01:  # Only show non-trivial values
                text_color = 'white' if value > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=text_color, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_all_heatmaps(conn: sqlite3.Connection, output_dir: Path):
    """Generate all spatial heatmaps."""
    heatmap_dir = output_dir / 'spatial_heatmaps'
    heatmap_dir.mkdir(exist_ok=True)
    
    colors = ['cyan', 'yellow', 'orange', 'red', 'white']
    cmaps = {
        'cyan': 'Blues',
        'yellow': 'YlOrBr',
        'orange': 'Oranges',
        'red': 'Reds',
        'white': 'Greys'
    }
    
    for color in colors:
        heatmap = load_spatial_heatmap(conn, color)
        plot_spatial_heatmap(heatmap, 
                            f'{color.capitalize()} UI Element Locations',
                            heatmap_dir / f'{color}_heatmap.png',
                            cmaps[color])
    
    # Edge density heatmap
    edge_heatmap = load_edge_density_heatmap(conn)
    plot_spatial_heatmap(edge_heatmap,
                        'Edge Density (Structural Complexity)',
                        heatmap_dir / 'edge_density_heatmap.png',
                        'viridis')

# ============================================================================
# TEXT REPORTS
# ============================================================================

def generate_text_summary(stats: Dict, output_dir: Path):
    """Generate text summary report."""
    report = f"""
INVISIBLE INC GAMEPLAY ANALYSIS REPORT
======================================
Generated: {Path.cwd()}

FRAME STATISTICS
----------------
Total frames scanned:     {stats['total_frames']:,}
Game frames detected:     {stats['game_frames']:,} ({stats['game_frames']/stats['total_frames']*100:.1f}%)
Non-game frames:          {stats['non_game_frames']:,} ({stats['non_game_frames']/stats['total_frames']*100:.1f}%)

GAME FRAME CHARACTERISTICS
---------------------------
Average brightness:       {stats['avg_brightness']:.1f} / 255
Average aspect ratio:     {stats['avg_aspect_ratio']:.2f}

UI COLOR PRESENCE
-----------------
Cyan (UI elements):       {stats['cyan_presence']*100:.1f}% of game frames
Yellow (text/warnings):   {stats['yellow_presence']*100:.1f}% of game frames
Red (alarms/threats):     {stats['red_presence']*100:.1f}% of game frames

INSIGHTS
--------
"""
    
    # Add insights based on data
    if stats['cyan_presence'] > 0.9:
        report += "✓ Cyan UI elements are nearly universal - excellent anchor for detection\n"
    else:
        report += f"⚠ Cyan only present in {stats['cyan_presence']*100:.1f}% of frames - may not be reliable anchor\n"
    
    if stats['yellow_presence'] > 0.5:
        report += "✓ Yellow text appears frequently - good for secondary detection\n"
    else:
        report += f"• Yellow text appears in {stats['yellow_presence']*100:.1f}% of frames - context-dependent\n"
    
    if stats['red_presence'] > 0.3:
        report += f"• Red elements present in {stats['red_presence']*100:.1f}% of frames - likely alarm states\n"
    else:
        report += f"• Red elements rare ({stats['red_presence']*100:.1f}%) - probably alarm-specific\n"
    
    report += f"\nNEXT STEPS\n----------\n"
    report += "1. Review spatial heatmaps to identify stable UI regions\n"
    report += "2. Examine color distribution for distinctive signatures\n"
    report += "3. Query database for element co-occurrence patterns\n"
    report += "4. Build element presence matrix for clustering analysis\n"
    
    output_path = output_dir / 'summary_report.txt'
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(report)

# ============================================================================
# HTML REPORT
# ============================================================================

def generate_html_report(stats: Dict, output_dir: Path):
    """Generate HTML report with embedded images."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Invisible Inc Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .stats {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-item {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 15px 0;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .insight {{
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <h1>Invisible Inc Gameplay Analysis Report</h1>
    
    <div class="stats">
        <h2>Frame Statistics</h2>
        <div class="stat-grid">
            <div class="stat-item">
                <div class="stat-value">{stats['total_frames']:,}</div>
                <div class="stat-label">Total Frames</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats['game_frames']:,}</div>
                <div class="stat-label">Game Frames</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats['cyan_presence']*100:.0f}%</div>
                <div class="stat-label">Cyan Presence</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats['avg_aspect_ratio']:.2f}</div>
                <div class="stat-label">Avg Aspect Ratio</div>
            </div>
        </div>
    </div>
    
    <h2>Frame Classification & Color Presence</h2>
    <img src="classification_stats.png" alt="Classification Statistics">
    
    <h2>Color Distribution Analysis</h2>
    <img src="color_distribution.png" alt="Color Distribution">
    
    <div class="insight">
        <strong>Key Insight:</strong> The hue distribution reveals which colors are distinctive enough 
        for reliable UI element detection. Look for peaks in the cyan (85-100°) and yellow (20-30°) 
        regions that correspond to game UI elements.
    </div>
    
    <h2>Spatial Heatmaps - UI Element Locations</h2>
    <p>These heatmaps show where different colors appear across all game frames (10×10 grid). 
       Bright areas indicate consistent UI element locations.</p>
    
    <div class="image-grid">
        <div>
            <h3>Cyan Elements</h3>
            <img src="spatial_heatmaps/cyan_heatmap.png" alt="Cyan Heatmap">
        </div>
        <div>
            <h3>Yellow Elements</h3>
            <img src="spatial_heatmaps/yellow_heatmap.png" alt="Yellow Heatmap">
        </div>
        <div>
            <h3>Red Elements</h3>
            <img src="spatial_heatmaps/red_heatmap.png" alt="Red Heatmap">
        </div>
        <div>
            <h3>Edge Density</h3>
            <img src="spatial_heatmaps/edge_density_heatmap.png" alt="Edge Density">
        </div>
    </div>
    
    <div class="insight">
        <strong>Next Steps:</strong> Use these spatial heatmaps to identify stable anchor regions.
        Areas with high, consistent presence are ideal for detection anchors. Low presence areas 
        indicate dynamic or mode-specific UI elements.
    </div>
    
    <h2>Database Queries for Deeper Analysis</h2>
    <pre style="background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto;">
# Find frames with specific color combinations
SELECT path FROM frames 
WHERE has_cyan = 1 AND has_yellow = 0 AND is_game_frame = 1
LIMIT 10;

# Get spatial distribution of a color
SELECT grid_x, grid_y, COUNT(*) as frequency 
FROM spatial_data 
WHERE has_cyan = 1 
GROUP BY grid_x, grid_y 
ORDER BY frequency DESC;

# Analyze color co-occurrence
SELECT 
    SUM(CASE WHEN has_cyan = 1 AND has_yellow = 1 THEN 1 ELSE 0 END) as cyan_yellow,
    SUM(CASE WHEN has_cyan = 1 AND has_red = 1 THEN 1 ELSE 0 END) as cyan_red
FROM frames WHERE is_game_frame = 1;
    </pre>
    
</body>
</html>
"""
    
    output_path = output_dir / 'summary.html'
    with open(output_path, 'w') as f:
        f.write(html)

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate analysis reports from batch scanner database')
    parser.add_argument('-i', '--input', default='analysis.db',
                       help='Input database path (default: analysis.db)')
    parser.add_argument('-o', '--output', default='reports',
                       help='Output directory for reports (default: reports)')
    
    args = parser.parse_args()
    
    # Check database exists
    if not Path(args.input).exists():
        print(f"Error: Database not found: {args.input}")
        print("Run batch_scanner.py first to create the database")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Generating reports from {args.input}...")
    
    # Connect to database
    conn = sqlite3.connect(args.input)
    
    # Load statistics
    stats = load_frame_stats(conn)
    
    # Generate text summary
    print("\n" + "="*60)
    generate_text_summary(stats, output_dir)
    print("="*60 + "\n")
    
    # Generate visualizations
    print("Generating classification plots...")
    plot_frame_classification(stats, output_dir)
    
    print("Generating color distribution plots...")
    hist = load_color_histogram(conn)
    plot_hsv_distribution(hist, output_dir)
    
    print("Generating spatial heatmaps...")
    generate_all_heatmaps(conn, output_dir)
    
    print("Generating HTML report...")
    generate_html_report(stats, output_dir)
    
    conn.close()
    
    print(f"\n✓ Reports generated in {output_dir}/")
    print(f"  - summary.html (open in browser)")
    print(f"  - summary_report.txt")
    print(f"  - classification_stats.png")
    print(f"  - color_distribution.png")
    print(f"  - spatial_heatmaps/ (6 heatmap images)")

if __name__ == '__main__':
    main()
