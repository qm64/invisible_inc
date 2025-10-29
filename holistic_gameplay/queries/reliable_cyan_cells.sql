-- Most Reliable Cyan Grid Cells
-- Shows which 10x10 grid cells consistently have cyan UI elements
-- Cells with >90% frequency are excellent anchor candidates

SELECT 
    sd.grid_x,
    sd.grid_y,
    ROUND(SUM(sd.has_cyan) * 100.0 / COUNT(*), 1) as cyan_frequency_pct,
    COUNT(*) as total_frames
FROM spatial_data sd
JOIN frames f ON sd.path = f.path
WHERE f.is_game_frame = 1
GROUP BY sd.grid_x, sd.grid_y
HAVING cyan_frequency_pct >= 90.0
ORDER BY cyan_frequency_pct DESC, total_frames DESC;
