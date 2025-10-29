-- Yellow Spatial Distribution
-- Shows where yellow text appears (agent AP, turn numbers)
-- Bottom-left should show agent AP locations
-- Top-right should show turn number location

SELECT 
    sd.grid_x,
    sd.grid_y,
    ROUND(SUM(sd.has_yellow) * 100.0 / COUNT(*), 1) as yellow_frequency_pct,
    COUNT(*) as total_frames
FROM spatial_data sd
JOIN frames f ON sd.path = f.path
WHERE f.is_game_frame = 1
GROUP BY sd.grid_x, sd.grid_y
HAVING yellow_frequency_pct >= 40.0
ORDER BY yellow_frequency_pct DESC;
