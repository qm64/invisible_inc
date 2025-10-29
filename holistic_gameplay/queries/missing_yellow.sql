-- Frames Missing Yellow
-- Yellow typically shows agent AP values and turn numbers
-- When missing, likely indicates agent actions in progress or opponent turns

SELECT 
    path,
    has_cyan,
    has_red,
    mean_brightness,
    edge_density
FROM frames 
WHERE is_game_frame = 1 
  AND has_yellow = 0
ORDER BY path
LIMIT 50;
