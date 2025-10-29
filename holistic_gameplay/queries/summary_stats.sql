-- Database Summary Statistics
-- Quick overview of all analyzed frames

SELECT 'Total Frames' as metric, COUNT(*) as value FROM frames
UNION ALL
SELECT 'Game Frames', COUNT(*) FROM frames WHERE is_game_frame = 1
UNION ALL
SELECT 'Non-Game Frames', COUNT(*) FROM frames WHERE is_game_frame = 0
UNION ALL
SELECT 'Avg Brightness (game)', ROUND(AVG(mean_brightness), 1) FROM frames WHERE is_game_frame = 1
UNION ALL
SELECT 'Avg Aspect Ratio (game)', ROUND(AVG(aspect_ratio), 2) FROM frames WHERE is_game_frame = 1
UNION ALL
SELECT 'Frames with Cyan', COUNT(*) FROM frames WHERE is_game_frame = 1 AND has_cyan = 1
UNION ALL
SELECT 'Frames with Yellow', COUNT(*) FROM frames WHERE is_game_frame = 1 AND has_yellow = 1
UNION ALL
SELECT 'Frames with Red', COUNT(*) FROM frames WHERE is_game_frame = 1 AND has_red = 1
UNION ALL
SELECT 'Frames with ALL colors', COUNT(*) FROM frames WHERE is_game_frame = 1 AND has_cyan = 1 AND has_yellow = 1 AND has_red = 1;
