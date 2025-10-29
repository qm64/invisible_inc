#!/usr/bin/env python3
"""
SQL Query Runner with Pretty Formatting
Version 1.0.0

Run SQL queries from files against SQLite database with nicely formatted output.
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple, Any

# ============================================================================
# TABLE FORMATTING
# ============================================================================

def format_table(headers: List[str], rows: List[Tuple], max_width: int = 80) -> str:
    """Format query results as a nice ASCII table."""
    if not rows:
        return "No results."
    
    # Convert all values to strings
    str_rows = [[str(val) if val is not None else 'NULL' for val in row] for row in rows]
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in str_rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))
    
    # Limit column widths to reasonable max
    max_col_width = max_width // len(headers) if len(headers) > 0 else 40
    col_widths = [min(w, max_col_width) for w in col_widths]
    
    # Build separator line
    separator = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'
    
    # Build header
    header_line = '|'
    for i, h in enumerate(headers):
        header_line += f' {h:<{col_widths[i]}} |'
    
    # Build rows
    result = [separator, header_line, separator]
    for row in str_rows:
        row_line = '|'
        for i, val in enumerate(row):
            # Truncate if too long
            if len(val) > col_widths[i]:
                val = val[:col_widths[i]-3] + '...'
            row_line += f' {val:<{col_widths[i]}} |'
        result.append(row_line)
    result.append(separator)
    
    return '\n'.join(result)

# ============================================================================
# CSV FORMATTING
# ============================================================================

def format_csv(headers: List[str], rows: List[Tuple]) -> str:
    """Format query results as CSV."""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue()

# ============================================================================
# QUERY EXECUTION
# ============================================================================

def execute_query(conn: sqlite3.Connection, query: str, format_type: str = 'table') -> str:
    """Execute a SQL query and return formatted results."""
    cursor = conn.cursor()
    
    try:
        cursor.execute(query)
        
        # Check if this is a SELECT query (has results)
        if cursor.description is None:
            # Non-SELECT query (INSERT, UPDATE, DELETE, etc.)
            conn.commit()
            return f"Query executed successfully. Rows affected: {cursor.rowcount}"
        
        # SELECT query - fetch results
        headers = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        if not rows:
            return "Query returned no results."
        
        # Format results
        if format_type == 'csv':
            return format_csv(headers, rows)
        else:
            result = format_table(headers, rows)
            result += f"\n\n{len(rows)} row(s) returned."
            return result
            
    except sqlite3.Error as e:
        return f"SQL Error: {e}"
    finally:
        cursor.close()

def execute_query_file(db_path: str, query_path: str, format_type: str = 'table') -> str:
    """Execute SQL from a file and return formatted results."""
    # Check files exist
    if not Path(db_path).exists():
        return f"Error: Database not found: {db_path}"
    
    if not Path(query_path).exists():
        return f"Error: Query file not found: {query_path}"
    
    # Read query
    with open(query_path, 'r') as f:
        query = f.read().strip()
    
    if not query:
        return "Error: Query file is empty"
    
    # Connect and execute
    conn = sqlite3.connect(db_path)
    
    # Handle multiple statements separated by semicolons
    statements = [s.strip() for s in query.split(';') if s.strip()]
    
    results = []
    results.append(f"Executing query from: {query_path}")
    results.append(f"Database: {db_path}")
    results.append("=" * 80)
    results.append("")
    
    for i, statement in enumerate(statements, 1):
        if len(statements) > 1:
            results.append(f"Statement {i}:")
            results.append("-" * 80)
        
        result = execute_query(conn, statement, format_type)
        results.append(result)
        results.append("")
    
    conn.close()
    return '\n'.join(results)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run SQL queries from files with pretty formatting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run query from file
  %(prog)s -d analysis.db -q queries/element_cooccurrence.sql
  
  # Multiple queries
  %(prog)s -d analysis.db -q query1.sql -q query2.sql
  
  # CSV output
  %(prog)s -d analysis.db -q query.sql --csv
  
  # Inline query
  %(prog)s -d analysis.db --inline "SELECT COUNT(*) FROM frames"
  
  # Save output to file
  %(prog)s -d analysis.db -q query.sql -o results.txt
        """
    )
    
    parser.add_argument('-d', '--database', required=True,
                       help='SQLite database path')
    parser.add_argument('-q', '--query', action='append', dest='queries',
                       help='SQL query file path (can specify multiple)')
    parser.add_argument('--inline', 
                       help='Execute inline SQL query instead of file')
    parser.add_argument('--csv', action='store_true',
                       help='Output as CSV instead of table')
    parser.add_argument('-o', '--output',
                       help='Write output to file instead of stdout')
    
    args = parser.parse_args()
    
    # Check that either query files or inline query provided
    if not args.queries and not args.inline:
        print("Error: Must specify either -q/--query or --inline", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    format_type = 'csv' if args.csv else 'table'
    
    all_results = []
    
    # Execute inline query
    if args.inline:
        if not Path(args.database).exists():
            print(f"Error: Database not found: {args.database}", file=sys.stderr)
            sys.exit(1)
        
        conn = sqlite3.connect(args.database)
        result = execute_query(conn, args.inline, format_type)
        all_results.append(f"Database: {args.database}")
        all_results.append("=" * 80)
        all_results.append(result)
        conn.close()
    
    # Execute query files
    if args.queries:
        for query_file in args.queries:
            result = execute_query_file(args.database, query_file, format_type)
            all_results.append(result)
            if len(args.queries) > 1:
                all_results.append("\n" + "=" * 80 + "\n")
    
    output = '\n'.join(all_results)
    
    # Output to file or stdout
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)

if __name__ == '__main__':
    main()
