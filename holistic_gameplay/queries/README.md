# SQL Query Library

Pre-built queries for analyzing Invisible Inc gameplay data.

## Usage

```bash
# Run a query
python sql.py -d analysis.db -q queries/element_cooccurrence.sql

# Run multiple queries
python sql.py -d analysis.db -q queries/summary_stats.sql -q queries/element_cooccurrence.sql

# CSV output
python sql.py -d analysis.db -q queries/missing_yellow.sql --csv

# Save to file
python sql.py -d analysis.db -q queries/reliable_cyan_cells.sql -o results.txt

# Inline query
python sql.py -d analysis.db --inline "SELECT COUNT(*) FROM frames"
```

## Available Queries

See individual .sql files for details on each query.
