-- What's in the bottom-right that's yellow?
SELECT f.path FROM spatial_data sd
JOIN frames f ON sd.path = f.path
WHERE f.is_game_frame = 1 
  AND sd.grid_x = 9 AND sd.grid_y = 9
  AND sd.has_yellow = 1
LIMIT 10;