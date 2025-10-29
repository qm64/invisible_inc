-- Element Co-occurrence Patterns
-- Shows which combinations of UI colors appear together in game frames
-- Helps identify game states (player turn, opponent turn, agent actions, etc.)

SELECT 
    has_cyan,
    has_yellow,
    has_red,
    COUNT(*) as frame_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM frames WHERE is_game_frame = 1), 2) as percentage
FROM frames 
WHERE is_game_frame = 1
GROUP BY has_cyan, has_yellow, has_red
ORDER BY frame_count DESC;
