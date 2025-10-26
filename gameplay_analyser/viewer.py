"""
Simple viewer for Invisible Inc capture sessions

Usage:
    python viewer.py captures/20251020_143022

Controls:
    Left/Right arrows: Navigate frames
    Space: Play/Pause
    I: Show input events near current frame
    M: Add marker/note at current frame
    Q: Quit
"""

import json
import sys
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk


class CaptureViewer:
    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.frames_dir = self.session_dir / "frames"
        
        # Load metadata
        with open(self.session_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        
        # Load frame metadata
        self.frame_metadata = []
        frames_meta_file = self.session_dir / "frames_metadata.jsonl"
        if frames_meta_file.exists():
            with open(frames_meta_file) as f:
                for line in f:
                    self.frame_metadata.append(json.loads(line))
        
        # Load input events
        self.input_events = []
        inputs_file = self.session_dir / "inputs.jsonl"
        if inputs_file.exists():
            with open(inputs_file) as f:
                for line in f:
                    self.input_events.append(json.loads(line))
        
        # Get all frame files
        self.frame_files = sorted(self.frames_dir.glob("frame_*.png"))
        self.current_frame = 0
        self.playing = False
        
        # Setup UI
        self.root = tk.Tk()
        self.root.title(f"Capture Viewer - {self.metadata['session_id']}")
        
        self._setup_ui()
        self._bind_keys()
        self._update_display()
        
    def _setup_ui(self):
        """Create UI components"""
        # Main frame for image
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        
        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Timeline slider
        self.timeline = ttk.Scale(
            control_frame,
            from_=0,
            to=len(self.frame_files) - 1,
            orient=tk.HORIZONTAL,
            command=self._on_timeline_change
        )
        self.timeline.pack(fill=tk.X, padx=5)
        
        # Info label
        self.info_label = tk.Label(
            self.root,
            text="",
            font=("Courier", 10),
            justify=tk.LEFT,
            anchor=tk.W
        )
        self.info_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Input events display
        self.events_text = tk.Text(self.root, height=8, font=("Courier", 9))
        self.events_text.pack(fill=tk.BOTH, padx=5, pady=5)
        
    def _bind_keys(self):
        """Bind keyboard controls"""
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('<space>', lambda e: self.toggle_play())
        self.root.bind('i', lambda e: self.show_nearby_inputs())
        self.root.bind('q', lambda e: self.root.quit())
        
    def _on_timeline_change(self, value):
        """Handle timeline slider change"""
        self.current_frame = int(float(value))
        self._update_display()
        
    def _update_display(self):
        """Update displayed frame and info"""
        if not self.frame_files:
            return
            
        # Load and display image
        img_path = self.frame_files[self.current_frame]
        img = Image.open(img_path)
        
        # Resize to fit window if needed
        max_size = (1200, 800)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep reference
        
        # Update timeline
        self.timeline.set(self.current_frame)
        
        # Update info
        frame_meta = self.frame_metadata[self.current_frame] if self.current_frame < len(self.frame_metadata) else {}
        timestamp = frame_meta.get('timestamp', 0)
        
        info = f"Frame: {self.current_frame}/{len(self.frame_files)-1}\n"
        info += f"Timestamp: {timestamp}ms\n"
        info += f"File: {img_path.name}\n"
        
        if frame_meta:
            info += f"Window: {frame_meta.get('window_title', 'unknown')}\n"
            info += f"Mode: {'FULLSCREEN' if frame_meta.get('is_fullscreen') else 'WINDOWED'}\n"
            info += f"Resolution: {frame_meta.get('resolution', 'unknown')}"
        
        self.info_label.config(text=info)
        
        # Show nearby input events
        self.show_nearby_inputs()
        
    def show_nearby_inputs(self):
        """Display input events near current frame"""
        if not self.frame_metadata or self.current_frame >= len(self.frame_metadata):
            return
            
        current_timestamp = self.frame_metadata[self.current_frame]['timestamp']
        window = 2000  # Show events within 2 seconds
        
        nearby = [
            e for e in self.input_events
            if abs(e['timestamp'] - current_timestamp) < window
        ]
        
        # Sort by timestamp
        nearby.sort(key=lambda e: e['timestamp'])
        
        # Display
        self.events_text.delete(1.0, tk.END)
        self.events_text.insert(1.0, f"Input events near frame {self.current_frame}:\n")
        self.events_text.insert(tk.END, "-" * 60 + "\n")
        
        for event in nearby[-20:]:  # Show last 20 events
            delta = event['timestamp'] - current_timestamp
            time_marker = f"+{delta}ms" if delta > 0 else f"{delta}ms"
            
            if event['type'].startswith('key'):
                self.events_text.insert(tk.END, f"{time_marker:>8} | KEY: {event['key']}\n")
            elif event['type'].startswith('mouse'):
                details = event.get('button', f"({event.get('x')}, {event.get('y')})")
                self.events_text.insert(tk.END, f"{time_marker:>8} | MOUSE: {event['type']} {details}\n")
    
    def next_frame(self):
        """Go to next frame"""
        if self.current_frame < len(self.frame_files) - 1:
            self.current_frame += 1
            self._update_display()
    
    def prev_frame(self):
        """Go to previous frame"""
        if self.current_frame > 0:
            self.current_frame -= 1
            self._update_display()
    
    def toggle_play(self):
        """Toggle playback"""
        self.playing = not self.playing
        if self.playing:
            self._play()
    
    def _play(self):
        """Playback frames"""
        if self.playing and self.current_frame < len(self.frame_files) - 1:
            self.next_frame()
            # Schedule next frame (roughly matching capture FPS)
            fps = self.metadata.get('fps', 2)
            delay = int(1000 / fps)
            self.root.after(delay, self._play)
        else:
            self.playing = False
    
    def run(self):
        """Start the viewer"""
        print(f"\nViewing session: {self.metadata['session_id']}")
        print(f"Total frames: {len(self.frame_files)}")
        print(f"Total input events: {len(self.input_events)}")
        print("\nControls:")
        print("  ← → : Navigate frames")
        print("  Space: Play/Pause")
        print("  I: Show input events")
        print("  Q: Quit")
        print()
        
        self.root.mainloop()


def main():
    if len(sys.argv) < 2:
        print("Usage: python viewer.py <session_directory>")
        print("\nExample:")
        print("  python viewer.py captures/20251020_143022")
        sys.exit(1)
    
    session_dir = Path(sys.argv[1])
    if not session_dir.exists():
        print(f"Error: Directory not found: {session_dir}")
        sys.exit(1)
    
    viewer = CaptureViewer(session_dir)
    viewer.run()


if __name__ == "__main__":
    main()