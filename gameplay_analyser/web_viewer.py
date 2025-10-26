"""
Web-based browser and viewer for Invisible Inc capture sessions
Shows list of sessions, then lets you view frames from selected session

Usage:
    python web_viewer.py captures
    
Then open: http://localhost:8000
"""

import json
import sys
import signal
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime


class SessionBrowserHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves session list and viewer"""
    
    captures_root = None
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/' or self.path == '/index.html':
            self.serve_session_list()
        elif self.path.startswith('/view/'):
            self.serve_viewer()
        elif self.path.startswith('/api/sessions'):
            self.serve_sessions_api()
        elif self.path.startswith('/api/session/'):
            self.serve_session_data()
        elif self.path.startswith('/api/frame/'):
            self.serve_frame()
        else:
            self.send_error(404)
    
    def serve_session_list(self):
        """Serve the session browser page"""
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Capture Sessions</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
        }
        .header {
            max-width: 1200px;
            margin: 0 auto 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #333;
        }
        h1 { color: #4a9eff; margin-bottom: 10px; }
        .sessions-table {
            max-width: 1400px;
            margin: 0 auto;
            background: #2a2a2a;
            border-radius: 8px;
            overflow: hidden;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th {
            background: #1a1a1a;
            padding: 15px;
            text-align: left;
            color: #4a9eff;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
            border-bottom: 2px solid #404040;
        }
        th:hover {
            background: #252525;
        }
        th.sortable:after {
            content: ' ⇅';
            color: #666;
            font-size: 12px;
        }
        th.sorted-asc:after {
            content: ' ↑';
            color: #4a9eff;
        }
        th.sorted-desc:after {
            content: ' ↓';
            color: #4a9eff;
        }
        td {
            padding: 15px;
            border-bottom: 1px solid #333;
        }
        tr {
            cursor: pointer;
            transition: background 0.1s;
        }
        tr:hover {
            background: #333;
        }
        .session-id { 
            color: #4a9eff;
            font-weight: 500;
        }
        .datetime { 
            color: #e0e0e0;
            font-size: 14px;
        }
        .stat { 
            color: #999;
            text-align: center;
        }
        .loading { text-align: center; padding: 60px; font-size: 18px; color: #666; }
        .error { 
            max-width: 600px; 
            margin: 60px auto; 
            padding: 30px; 
            background: #3a1a1a; 
            border: 1px solid #a00;
            border-radius: 8px;
            color: #faa;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎮 Invisible Inc Capture Sessions</h1>
        <p>Select a session to view captured frames and input events</p>
    </div>
    <div id="content" class="loading">Loading sessions...</div>
    
    <script>
        let allSessions = [];
        let sortColumn = 'datetime';
        let sortDirection = 'desc'; // Start with latest at top
        
        function formatDateTime(timestamp) {
            const date = new Date(timestamp * 1000);
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            
            const day = String(date.getDate()).padStart(2, '0');
            const month = months[date.getMonth()];
            const year = String(date.getFullYear()).slice(-2);
            
            let hours = date.getHours();
            const ampm = hours >= 12 ? 'pm' : 'am';
            hours = hours % 12 || 12;
            const minutes = String(date.getMinutes()).padStart(2, '0');
            
            return `${day}/${month}/${year} ${hours}:${minutes}${ampm}`;
        }
        
        function sortSessions(column) {
            if (sortColumn === column) {
                sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                sortColumn = column;
                sortDirection = column === 'datetime' ? 'desc' : 'asc';
            }
            renderTable();
        }
        
        function renderTable() {
            const sorted = [...allSessions].sort((a, b) => {
                let aVal, bVal;
                
                switch(sortColumn) {
                    case 'datetime':
                        aVal = a.timestamp;
                        bVal = b.timestamp;
                        break;
                    case 'frames':
                        aVal = a.frame_count;
                        bVal = b.frame_count;
                        break;
                    case 'events':
                        aVal = a.event_count;
                        bVal = b.event_count;
                        break;
                    case 'duration':
                        aVal = a.duration_seconds || 0;
                        bVal = b.duration_seconds || 0;
                        break;
                    default:
                        aVal = a.id;
                        bVal = b.id;
                }
                
                if (sortDirection === 'asc') {
                    return aVal > bVal ? 1 : -1;
                } else {
                    return aVal < bVal ? 1 : -1;
                }
            });
            
            const tbody = sorted.map(session => `
                <tr onclick="window.location.href='/view/${session.id}'">
                    <td class="session-id">${session.id}</td>
                    <td class="datetime">${formatDateTime(session.timestamp)}</td>
                    <td class="stat">${session.frame_count}</td>
                    <td class="stat">${session.event_count}</td>
                    <td class="stat">${session.duration}</td>
                </tr>
            `).join('');
            
            // Update sort indicators
            document.querySelectorAll('th').forEach(th => {
                th.classList.remove('sorted-asc', 'sorted-desc');
            });
            const currentTh = document.querySelector(`th[data-column="${sortColumn}"]`);
            if (currentTh) {
                currentTh.classList.add(`sorted-${sortDirection}`);
            }
            
            document.getElementById('table-body').innerHTML = tbody;
        }
        
        async function loadSessions() {
            try {
                const response = await fetch('/api/sessions');
                if (!response.ok) throw new Error('Failed to load sessions');
                allSessions = await response.json();
                
                if (allSessions.length === 0) {
                    document.getElementById('content').innerHTML = 
                        '<div class="error">No capture sessions found in the captures directory</div>';
                    return;
                }
                
                const table = `
                    <div class="sessions-table">
                        <table>
                            <thead>
                                <tr>
                                    <th class="sortable" data-column="id" onclick="sortSessions('id')">Session ID</th>
                                    <th class="sortable sorted-desc" data-column="datetime" onclick="sortSessions('datetime')">Date & Time</th>
                                    <th class="sortable" data-column="frames" onclick="sortSessions('frames')">Frames</th>
                                    <th class="sortable" data-column="events" onclick="sortSessions('events')">Events</th>
                                    <th class="sortable" data-column="duration" onclick="sortSessions('duration')">Duration</th>
                                </tr>
                            </thead>
                            <tbody id="table-body"></tbody>
                        </table>
                    </div>
                `;
                
                document.getElementById('content').innerHTML = table;
                renderTable();
            } catch (error) {
                document.getElementById('content').innerHTML = 
                    `<div class="error">Error loading sessions: ${error.message}</div>`;
            }
        }
        
        loadSessions();
    </script>
</body>
</html>"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_viewer(self):
        """Serve the frame viewer for a specific session"""
        session_id = self.path.split('/view/')[-1]
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Session {session_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            overflow: hidden;
        }}
        .viewer {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        .header {{
            padding: 15px 20px;
            background: #2a2a2a;
            border-bottom: 2px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .back-button {{
            padding: 8px 16px;
            background: #404040;
            border: 1px solid #555;
            border-radius: 4px;
            color: #e0e0e0;
            text-decoration: none;
            cursor: pointer;
        }}
        .back-button:hover {{ background: #4a4a4a; }}
        .session-title {{ color: #4a9eff; font-size: 18px; font-weight: bold; }}
        .frame-display {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #0a0a0a;
            overflow: hidden;
        }}
        .frame-display img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        .controls {{
            padding: 20px;
            background: #2a2a2a;
            border-top: 2px solid #333;
        }}
        .timeline {{
            width: 100%;
            height: 6px;
            background: #404040;
            border-radius: 3px;
            margin-bottom: 15px;
            cursor: pointer;
            position: relative;
        }}
        .timeline-progress {{
            height: 100%;
            background: #4a9eff;
            border-radius: 3px;
            width: 0%;
            transition: width 0.1s;
        }}
        .playback-controls {{
            display: flex;
            gap: 10px;
            align-items: center;
            justify-content: center;
            margin-bottom: 10px;
        }}
        button {{
            padding: 8px 16px;
            background: #404040;
            border: 1px solid #555;
            border-radius: 4px;
            color: #e0e0e0;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{ background: #4a4a4a; }}
        button:active {{ background: #505050; }}
        .frame-info {{
            text-align: center;
            color: #999;
            font-size: 14px;
        }}
        .events {{
            max-height: 120px;
            overflow-y: auto;
            margin-top: 10px;
            padding: 10px;
            background: #1a1a1a;
            border-radius: 4px;
            font-size: 12px;
            font-family: monospace;
        }}
        .event {{
            padding: 4px;
            margin-bottom: 2px;
            background: #2a2a2a;
            border-left: 3px solid #4a9eff;
            padding-left: 8px;
        }}
        .loading {{
            text-align: center;
            padding: 60px;
            font-size: 18px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="viewer">
        <div class="header">
            <a href="/" class="back-button">← Back to Sessions</a>
            <div class="session-title">Session: {session_id}</div>
            <div style="width: 100px;"></div>
        </div>
        
        <div class="frame-display">
            <div id="loading" class="loading">Loading session...</div>
            <img id="frame" style="display: none;">
        </div>
        
        <div class="controls">
            <div class="timeline" id="timeline">
                <div class="timeline-progress" id="progress"></div>
            </div>
            
            <div class="playback-controls">
                <button id="prevBtn">⏮ Prev</button>
                <button id="playBtn">▶ Play</button>
                <button id="nextBtn">Next ⏭</button>
                <button id="speedBtn">1x</button>
            </div>
            
            <div class="frame-info" id="info">Frame 0 / 0</div>
            
            <div class="events" id="events" style="display: none;"></div>
        </div>
    </div>
    
    <script>
        let frames = [];
        let events = [];
        let currentFrame = 0;
        let isPlaying = false;
        let playbackSpeed = 1;
        let playInterval = null;
        
        const sessionId = '{session_id}';
        
        async function loadSession() {{
            try {{
                const response = await fetch(`/api/session/${{sessionId}}`);
                if (!response.ok) throw new Error('Session not found');
                const data = await response.json();
                
                frames = data.frames;
                events = data.events || [];
                
                document.getElementById('loading').style.display = 'none';
                document.getElementById('frame').style.display = 'block';
                
                if (events.length > 0) {{
                    document.getElementById('events').style.display = 'block';
                }}
                
                showFrame(0);
            }} catch (error) {{
                document.getElementById('loading').innerHTML = 
                    `<div style="color: #faa;">Error: ${{error.message}}</div>`;
            }}
        }}
        
        function showFrame(index) {{
            if (index < 0 || index >= frames.length) return;
            currentFrame = index;
            
            const frameImg = document.getElementById('frame');
            frameImg.src = `/api/frame/${{sessionId}}/${{frames[index]}}`;
            
            const progress = ((index + 1) / frames.length) * 100;
            document.getElementById('progress').style.width = progress + '%';
            
            // Extract frame number from filename (e.g., "frame_000123.png" -> 123)
            const filename = frames[index];
            const frameNumber = filename.match(/frame_(\\d+)/)?.[1] || '???';
            
            document.getElementById('info').innerHTML = 
                `Index ${{index}} / ${{frames.length - 1}} <span style="color: #666">|</span> <span style="color: #888">${{filename}}</span> <span style="color: #666">|</span> <span style="color: #4a9eff">Frame #${{frameNumber}}</span>`;
            
            // Show nearby events
            const eventsDiv = document.getElementById('events');
            const nearbyEvents = events.filter(e => 
                Math.abs(e.frame_number - index) <= 5
            );
            
            if (nearbyEvents.length > 0) {{
                eventsDiv.innerHTML = nearbyEvents.map(e => 
                    `<div class="event">[F${{e.frame_number}}] ${{e.event_type}}: ${{JSON.stringify(e.data)}}</div>`
                ).join('');
            }}
        }}
        
        function play() {{
            if (isPlaying) {{
                clearInterval(playInterval);
                isPlaying = false;
                document.getElementById('playBtn').textContent = '▶ Play';
            }} else {{
                isPlaying = true;
                document.getElementById('playBtn').textContent = '⏸ Pause';
                playInterval = setInterval(() => {{
                    if (currentFrame < frames.length - 1) {{
                        showFrame(currentFrame + 1);
                    }} else {{
                        clearInterval(playInterval);
                        isPlaying = false;
                        document.getElementById('playBtn').textContent = '▶ Play';
                    }}
                }}, 500 / playbackSpeed);
            }}
        }}
        
        function cycleSpeed() {{
            const speeds = [0.5, 1, 2, 4];
            const currentIndex = speeds.indexOf(playbackSpeed);
            playbackSpeed = speeds[(currentIndex + 1) % speeds.length];
            document.getElementById('speedBtn').textContent = playbackSpeed + 'x';
            
            if (isPlaying) {{
                clearInterval(playInterval);
                play();
                play();
            }}
        }}
        
        // Event listeners
        document.getElementById('prevBtn').onclick = () => showFrame(currentFrame - 1);
        document.getElementById('nextBtn').onclick = () => showFrame(currentFrame + 1);
        document.getElementById('playBtn').onclick = play;
        document.getElementById('speedBtn').onclick = cycleSpeed;
        
        document.getElementById('timeline').onclick = (e) => {{
            const rect = e.target.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const percent = x / rect.width;
            const frame = Math.floor(percent * frames.length);
            showFrame(frame);
        }};
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') showFrame(currentFrame - 1);
            if (e.key === 'ArrowRight') showFrame(currentFrame + 1);
            if (e.key === ' ') {{ e.preventDefault(); play(); }}
        }});
        
        loadSession();
    </script>
</body>
</html>"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_sessions_api(self):
        """Return JSON list of available sessions"""
        sessions = []
        
        for session_dir in sorted(self.captures_root.iterdir(), reverse=True):
            if not session_dir.is_dir():
                continue
            
            frames_dir = session_dir / "frames"
            if not frames_dir.exists():
                continue
            
            frame_files = list(frames_dir.glob("frame_*.png"))
            
            # Try to load session info
            metadata_file = session_dir / "metadata.json"
            inputs_file = session_dir / "inputs.jsonl"
            event_count = 0
            duration = None
            timestamp = None
            
            if metadata_file.exists():
                with open(metadata_file) as f:
                    info = json.load(f)
                    event_count = info.get('total_events', 0)
                    duration_sec = info.get('duration_seconds')
                    if duration_sec:
                        # Convert seconds to readable format
                        if duration_sec >= 60:
                            mins = int(duration_sec // 60)
                            secs = duration_sec % 60
                            duration = f"{mins}m {secs:.1f}s"
                        else:
                            duration = f"{duration_sec:.1f}s"
                    
                    # Parse start_time (ISO format string)
                    start_time_str = info.get('start_time')
                    if start_time_str:
                        try:
                            dt = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                            timestamp = dt.timestamp()
                        except:
                            timestamp = 0
            
            # Count events from inputs.jsonl if metadata missing
            if event_count == 0 and inputs_file.exists():
                with open(inputs_file) as f:
                    event_count = sum(1 for _ in f)
            
            # Parse timestamp from directory name as fallback
            if timestamp is None:
                try:
                    dt = datetime.strptime(session_dir.name, "%Y%m%d_%H%M%S")
                    timestamp = dt.timestamp()
                except:
                    timestamp = 0
            
            # Parse duration to seconds for sorting
            duration_seconds = 0
            if duration and duration != 'N/A':
                try:
                    # Parse formats like "55.0s" or "3m 25s"
                    if 'm' in duration:
                        parts = duration.replace('s', '').split('m')
                        duration_seconds = int(parts[0]) * 60 + float(parts[1].strip())
                    else:
                        duration_seconds = float(duration.replace('s', ''))
                except:
                    duration_seconds = 0
            
            sessions.append({
                'id': session_dir.name,
                'frame_count': len(frame_files),
                'event_count': event_count,
                'duration': duration or 'N/A',
                'duration_seconds': duration_seconds,
                'timestamp': timestamp
            })
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(sessions).encode())
    
    def serve_session_data(self):
        """Return data for a specific session"""
        session_id = self.path.split('/api/session/')[-1]
        session_dir = self.captures_root / session_id
        
        if not session_dir.exists():
            self.send_error(404, "Session not found")
            return
        
        frames_dir = session_dir / "frames"
        # Filter out detected images - only show original frames
        frame_files = sorted([f for f in frames_dir.glob("frame_*.png") 
                            if not f.name.endswith('_detected.png')])
        
        # Load input events if available
        events = []
        inputs_file = session_dir / "inputs.jsonl"
        if inputs_file.exists():
            with open(inputs_file) as f:
                for line in f:
                    events.append(json.loads(line))
        
        data = {
            'session_id': session_id,
            'frames': [f.name for f in frame_files],
            'events': events
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def serve_frame(self):
        """Serve a specific frame image"""
        parts = self.path.split('/')
        session_id = parts[3]
        frame_name = parts[4]
        
        frame_path = self.captures_root / session_id / "frames" / frame_name
        
        if not frame_path.exists():
            self.send_error(404, "Frame not found")
            return
        
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.end_headers()
        with open(frame_path, 'rb') as f:
            self.wfile.write(f.read())


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n\nShutting down server...")
    sys.exit(0)


def main():
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, 'SIGQUIT'):
        signal.signal(signal.SIGQUIT, signal_handler)
    
    if len(sys.argv) < 2:
        print("Usage: python web_viewer.py <captures_directory>")
        print("\nExample:")
        print("  python web_viewer.py captures")
        sys.exit(1)
    
    captures_root = Path(sys.argv[1])
    if not captures_root.exists():
        print(f"Error: Directory not found: {captures_root}")
        sys.exit(1)
    
    # Count sessions
    sessions = [d for d in captures_root.iterdir() if d.is_dir() and (d / "frames").exists()]
    
    print(f"\n{'='*60}")
    print(f"Invisible Inc Capture Browser")
    print(f"{'='*60}")
    print(f"\nCaptures directory: {captures_root}")
    print(f"Sessions found: {len(sessions)}")
    print(f"\n✓ Server started at http://localhost:8000")
    print(f"\n→ Open your browser to: http://localhost:8000")
    print(f"→ Press Ctrl+C or Ctrl+\\ to stop\n")
    
    # Set the captures root for the handler
    SessionBrowserHandler.captures_root = captures_root
    
    # Start server
    port = 8000
    server = HTTPServer(('localhost', port), SessionBrowserHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()