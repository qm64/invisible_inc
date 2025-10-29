"""
Invisible Inc Game State Capture System - Cross-Platform

Tries multiple window detection methods:
1. Window title matching
2. Process/owner name matching
3. Fallback to fullscreen capture

Works on Windows, macOS, and Linux with appropriate dependencies.

Requirements:
    pip install mss pillow pynput sounddevice scipy numpy
    
    Optional (improves detection):
    pip install pygetwindow          # Windows/Linux
    pip install pyobjc-framework-Quartz  # macOS

Usage:
    python invisible_capture.py
"""

import os
import time
import json
import threading
import queue
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import mss
import numpy as np
from PIL import Image
from pynput import keyboard, mouse
import sounddevice as sd
from scipy.io import wavfile


class UnifiedWindowFinder:
    """Cross-platform window finder with multiple detection strategies"""
    
    def __init__(self, 
                 window_owner_name = None,
                 window_title_pattern = None):
        # Ensure strings, not lists
        if isinstance(window_owner_name, list):
            self.window_owner_name = window_owner_name[0] if window_owner_name else "InvisibleInc"
        else:
            self.window_owner_name = window_owner_name or "InvisibleInc"
            
        if isinstance(window_title_pattern, list):
            self.window_title_pattern = window_title_pattern[0] if window_title_pattern else "Invisible"
        else:
            self.window_title_pattern = window_title_pattern or "Invisible"
            
        self.system = platform.system()
        
        # Try to import platform-specific libraries
        self.use_quartz = False
        self.use_pygetwindow = False
        
        if self.system == "Darwin":  # macOS
            try:
                import Quartz
                self.Quartz = Quartz
                self.use_quartz = True
                print("✓ Using macOS Quartz APIs for window detection")
            except ImportError:
                print("⚠ Quartz not available (pip install pyobjc-framework-Quartz)")
        
        try:
            import pygetwindow as gw
            self.gw = gw
            self.use_pygetwindow = True
            print(f"✓ Using pygetwindow for {self.system} window detection")
        except ImportError:
            print("⚠ pygetwindow not available (pip install pygetwindow)")
            
    def find_window(self) -> Optional[Dict]:
        """
        Try multiple methods to find the game window
        Returns window info dict or None
        """
        print("\nSearching for Invisible Inc window...")
        print(f"  Owner name: {self.window_owner_name}")
        print(f"  Title pattern: {self.window_title_pattern}")
        
        # Method 1: macOS Quartz API (most reliable on macOS)
        if self.use_quartz:
            window = self._find_with_quartz()
            if window:
                return window
        
        # Method 2: pygetwindow (works on Windows/Linux)
        if self.use_pygetwindow:
            window = self._find_with_pygetwindow()
            if window:
                return window
        
        # # Method 3: Fallback to fullscreen
        # print("  → No window found, using fullscreen capture")
        # return self._fallback_fullscreen()
        print("  → No window found, skipping capture")
        return None
    
    def is_game_focused(self) -> bool:
        """
        Check if the game window is currently the active/focused window
        Returns: True if game is focused, False otherwise
        """
        if not self.use_quartz:
            return True  # Can't detect on other platforms, assume focused
        
        try:
            # Get the frontmost (active) application
            from AppKit import NSWorkspace
            workspace = NSWorkspace.sharedWorkspace()
            active_app = workspace.frontmostApplication()
            
            if active_app:
                active_name = active_app.localizedName()
                # Also try bundleIdentifier for more reliable matching
                bundle_id = active_app.bundleIdentifier()
                
                # Check multiple possible matches
                is_focused = (active_name == self.window_owner_name or
                             self.window_owner_name.lower() in active_name.lower() or
                             (bundle_id and 'invisible' in bundle_id.lower()))
                
                # Debug output (only print when focus changes)
                if not hasattr(self, '_last_focused_app'):
                    self._last_focused_app = None
                
                if active_name != self._last_focused_app:
                    print(f"[Focus] Active app: '{active_name}' (bundle: {bundle_id}) - {'✓ MATCH' if is_focused else '✗ NO MATCH'}")
                    self._last_focused_app = active_name
                
                return is_focused
            
        except Exception as e:
            print(f"[Focus] Error detecting focus: {e}")
            # If we can't detect, assume focused to continue capturing
            return True
        
        return False
    
    def _find_with_quartz(self) -> Optional[Dict]:
        """Find window using macOS Quartz APIs - using proven window_detector.py logic"""
        try:
            # Get list of all windows (excluding desktop elements)
            window_list = self.Quartz.CGWindowListCopyWindowInfo(
                self.Quartz.kCGWindowListOptionOnScreenOnly | 
                self.Quartz.kCGWindowListExcludeDesktopElements,
                self.Quartz.kCGNullWindowID
            )
            
            # First pass: look for EXACT owner match (most reliable)
            for window in window_list:
                window_owner = window.get('kCGWindowOwnerName', '')
                
                if window_owner == self.window_owner_name:
                    bounds = window.get('kCGWindowBounds', {})
                    width = int(bounds.get('Width', 0))
                    height = int(bounds.get('Height', 0))
                    
                    # Skip tiny windows (menu bars, tooltips, etc.)
                    if width < 200 or height < 200:
                        continue
                    
                    result = {
                        'title': window.get('kCGWindowName', window_owner),
                        'owner': window_owner,
                        'left': int(bounds.get('X', 0)),
                        'top': int(bounds.get('Y', 0)),
                        'width': width,
                        'height': height,
                        'is_fullscreen': self._is_fullscreen_size(width, height),
                        'detection_method': 'quartz_exact_owner'
                    }
                    
                    print(f"  ✓ Found by exact owner match: '{window_owner}'")
                    print(f"    Title: '{result['title']}'")
                    print(f"    Size: {result['width']}x{result['height']}")
                    print(f"    Mode: {'FULLSCREEN' if result['is_fullscreen'] else 'WINDOWED'}")
                    return result
            
            # Second pass: look for partial match (fallback)
            for window in window_list:
                window_title = window.get('kCGWindowName', '')
                window_owner = window.get('kCGWindowOwnerName', '')
                
                # Check title pattern - but be more specific
                # Only match if title STARTS with pattern or contains ", Inc"
                title_match = (window_title.lower().startswith(self.window_title_pattern.lower()) or
                              ", inc" in window_title.lower())
                
                # Don't match terminal/editor windows
                if 'terminal' in window_owner.lower() or 'code' in window_owner.lower():
                    title_match = False
                
                owner_match = self.window_owner_name.lower() in window_owner.lower()
                
                if title_match or owner_match:
                    bounds = window.get('kCGWindowBounds', {})
                    width = int(bounds.get('Width', 0))
                    height = int(bounds.get('Height', 0))
                    
                    # Skip tiny windows
                    if width < 200 or height < 200:
                        continue
                    
                    result = {
                        'title': window_title or window_owner,
                        'owner': window_owner,
                        'left': int(bounds.get('X', 0)),
                        'top': int(bounds.get('Y', 0)),
                        'width': width,
                        'height': height,
                        'is_fullscreen': self._is_fullscreen_size(width, height),
                        'detection_method': 'quartz_partial_match'
                    }
                    
                    match_type = 'title' if title_match else 'owner'
                    print(f"  ✓ Found by partial {match_type} match")
                    print(f"    Title: '{window_title}'")
                    print(f"    Owner: '{window_owner}'")
                    print(f"    Size: {result['width']}x{result['height']}")
                    print(f"    Mode: {'FULLSCREEN' if result['is_fullscreen'] else 'WINDOWED'}")
                    return result
            
        except Exception as e:
            print(f"  ✗ Quartz detection failed: {e}")
        
        return None
    
    def _find_with_pygetwindow(self) -> Optional[Dict]:
        """Find window using pygetwindow"""
        try:
            all_windows = self.gw.getAllTitles()
            
            # Try title pattern
            for title in all_windows:
                if self.window_title_pattern.lower() in title.lower():
                    try:
                        win = self.gw.getWindowsWithTitle(title)[0]
                        
                        result = {
                            'title': title,
                            'owner': 'unknown',
                            'left': win.left,
                            'top': win.top,
                            'width': win.width,
                            'height': win.height,
                            'is_fullscreen': self._is_fullscreen_size(win.width, win.height),
                            'detection_method': 'pygetwindow_title',
                            'window_obj': win
                        }
                        
                        print(f"  ✓ Found via pygetwindow (title)")
                        print(f"    Title: '{title}'")
                        print(f"    Size: {result['width']}x{result['height']}")
                        print(f"    Mode: {'FULLSCREEN' if result['is_fullscreen'] else 'WINDOWED'}")
                        
                        return result
                    except Exception as e:
                        print(f"  ✗ Could not access window '{title}': {e}")
                        continue
            
        except Exception as e:
            print(f"  ✗ pygetwindow detection failed: {e}")
        
        return None
    
    def _is_fullscreen_size(self, width: int, height: int) -> bool:
        """Check if window size matches screen size"""
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                width_match = abs(width - monitor['width']) < 50
                height_match = abs(height - monitor['height']) < 50
                return width_match and height_match
        except:
            return False
    
    def _fallback_fullscreen(self) -> Dict:
        """Fallback: capture entire primary monitor"""
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            return {
                'title': 'Fullscreen Capture (Fallback)',
                'owner': 'fallback',
                'left': monitor['left'],
                'top': monitor['top'],
                'width': monitor['width'],
                'height': monitor['height'],
                'is_fullscreen': True,
                'detection_method': 'fallback'
            }


class ScreenshotCapture:
    """Captures screenshots at specified FPS"""
    
    def __init__(self, output_dir: Path, fps: float = 2.0, 
                 window_finder: UnifiedWindowFinder = None, skip_unfocused: bool = False):
        self.output_dir = output_dir / "frames"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.interval = 1.0 / fps
        self.window_finder = window_finder or UnifiedWindowFinder()
        self.skip_unfocused = skip_unfocused  # New flag
        
        self.running = False
        self.thread = None
        self.frame_count = 0
        self.metadata_queue = queue.Queue()
        
    def start(self):
        """Start screenshot capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"\n✓ Screenshot capture started at {self.fps} fps")
        
    def stop(self):
        """Stop screenshot capture thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print(f"\n✓ Screenshot capture stopped. Total frames: {self.frame_count}")
        
    def _capture_loop(self):
        """Main capture loop with precise timing"""
        window_info = None
        last_window_check = 0
        window_check_interval = 3.0  # Re-check window every 3 seconds
        skipped_frames = 0
        last_focus_state = None
        
        with mss.mss() as sct:
            next_capture = time.time()
            
            while self.running:
                current_time = time.time()
                
                # Periodically re-find window
                if current_time - last_window_check > window_check_interval:
                    new_window_info = self.window_finder.find_window()
                    
                    # Only print if window changed
                    if new_window_info != window_info:
                        window_info = new_window_info
                        if window_info:
                            print(f"\n[{self.frame_count}] Window update: "
                                  f"{window_info['width']}x{window_info['height']} "
                                  f"{'[FULLSCREEN]' if window_info['is_fullscreen'] else '[WINDOWED]'}")
                    
                    last_window_check = current_time
                
                if current_time >= next_capture:
                    if window_info:
                        # Check focus only if enabled
                        should_capture = True
                        if self.skip_unfocused:
                            is_focused = self.window_finder.is_game_focused()
                            
                            # Track focus state changes
                            if is_focused != last_focus_state:
                                if is_focused:
                                    print(f"[{self.frame_count}] ✓ Game focused - capturing")
                                    if skipped_frames > 0:
                                        print(f"    (skipped {skipped_frames} frames)")
                                        skipped_frames = 0
                                else:
                                    print(f"[{self.frame_count}] Game not focused - skipping frames...")
                                last_focus_state = is_focused
                            
                            should_capture = is_focused
                        
                        # Capture frame
                        if should_capture:
                            self._capture_frame(sct, window_info, current_time)
                        else:
                            skipped_frames += 1
                    
                    next_capture = current_time + self.interval
                    
                # Sleep until next capture
                sleep_time = max(0, next_capture - time.time() - 0.001)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    
    def _capture_frame(self, sct, window_info: Dict, timestamp: float):
        """Capture a single frame"""
        try:
            # Define capture region
            monitor = {
                'left': window_info['left'],
                'top': window_info['top'],
                'width': window_info['width'],
                'height': window_info['height']
            }
            
            # Capture screenshot
            screenshot = sct.grab(monitor)
            img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
            
            # Save with frame number
            frame_filename = f"frame_{self.frame_count:06d}.png"
            frame_path = self.output_dir / frame_filename
            img.save(frame_path, 'PNG', optimize=True)
            
            # Queue metadata
            metadata = {
                'frame_id': self.frame_count,
                'timestamp': int(timestamp * 1000),
                'filename': frame_filename,
                'window_title': window_info['title'],
                'window_owner': window_info.get('owner', 'unknown'),
                'is_fullscreen': window_info['is_fullscreen'],
                'resolution': f"{window_info['width']}x{window_info['height']}",
                'detection_method': window_info.get('detection_method', 'unknown')
            }
            self.metadata_queue.put(metadata)
            
            self.frame_count += 1
            
        except Exception as e:
            print(f"\n✗ Error capturing frame {self.frame_count}: {e}")


class AudioCapture:
    """Captures system audio"""
    
    def __init__(self, output_dir: Path, sample_rate: int = 44100):
        self.output_dir = output_dir / "audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_rate = sample_rate
        self.channels = 2
        
        self.running = False
        self.thread = None
        self.audio_data = []
        self.start_time = None
        
    def start(self):
        """Start audio capture thread"""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"✓ Audio capture started at {self.sample_rate} Hz")
        
        system = platform.system()
        if system == "Darwin":
            print("  Note: macOS may need BlackHole for system audio")
        elif system == "Windows":
            print("  Note: Windows may need 'Stereo Mix' enabled")
        
    def stop(self):
        """Stop audio capture and save file"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.audio_data:
            audio_array = np.concatenate(self.audio_data, axis=0)
            output_file = self.output_dir / "session_audio.wav"
            wavfile.write(str(output_file), self.sample_rate, audio_array)
            
            duration = len(audio_array) / self.sample_rate
            print(f"✓ Audio saved: {duration:.1f}s to {output_file}")
        else:
            print("⚠ No audio data captured")
        
    def _capture_loop(self):
        """Main audio capture loop"""
        try:
            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"  Audio status: {status}")
                self.audio_data.append(indata.copy())
            
            with sd.InputStream(samplerate=self.sample_rate, 
                              channels=self.channels,
                              callback=audio_callback):
                while self.running:
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"✗ Audio capture error: {e}")
            print("  Continuing without audio.")


class InputLogger:
    """Logs keyboard and mouse events"""
    
    def __init__(self, output_dir: Path):
        self.output_file = output_dir / "inputs.jsonl"
        self.file_handle = None
        
        self.keyboard_listener = None
        self.mouse_listener = None
        self.running = False
        self.start_time = None
        self.event_count = 0
        
    def start(self):
        """Start input logging"""
        self.running = True
        self.start_time = time.time()
        self.file_handle = open(self.output_file, 'w')
        
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.keyboard_listener.start()
        
        self.mouse_listener = mouse.Listener(
            on_click=self._on_mouse_click,
            on_move=self._on_mouse_move,
            on_scroll=self._on_mouse_scroll
        )
        self.mouse_listener.start()
        
        print("✓ Input logging started")
        
    def stop(self):
        """Stop input logging"""
        self.running = False
        
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        if self.mouse_listener:
            self.mouse_listener.stop()
            
        if self.file_handle:
            self.file_handle.close()
            
        print(f"✓ Input logging stopped. Total events: {self.event_count}")
        
    def _log_event(self, event: Dict):
        """Write event to JSONL file"""
        if self.file_handle and self.running:
            event['timestamp'] = int(time.time() * 1000)
            self.file_handle.write(json.dumps(event) + '\n')
            self.file_handle.flush()
            self.event_count += 1
    
    def _on_key_press(self, key):
        try:
            key_name = key.char if hasattr(key, 'char') else str(key)
        except:
            key_name = str(key)
        self._log_event({'type': 'key_down', 'key': key_name})
    
    def _on_key_release(self, key):
        try:
            key_name = key.char if hasattr(key, 'char') else str(key)
        except:
            key_name = str(key)
        self._log_event({'type': 'key_up', 'key': key_name})
    
    def _on_mouse_click(self, x, y, button, pressed):
        self._log_event({
            'type': 'mouse_click' if pressed else 'mouse_release',
            'button': str(button),
            'x': x, 'y': y
        })
    
    def _on_mouse_move(self, x, y):
        # Throttle mouse moves
        if self.event_count % 50 == 0:
            self._log_event({'type': 'mouse_move', 'x': x, 'y': y})
    
    def _on_mouse_scroll(self, x, y, dx, dy):
        self._log_event({'type': 'mouse_scroll', 'x': x, 'y': y, 'dx': dx, 'dy': dy})


class CaptureSession:
    """Main coordinator for capture session"""
    
    def __init__(self, output_base_dir: str = "captures", fps: float = 2.0,
                 window_owner_name = None,
                 window_title_pattern = None):
        # Create session directory
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(output_base_dir) / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Set defaults
        owner = window_owner_name or "InvisibleInc"
        title = window_title_pattern or "Invisible"
        
        print(f"\n{'='*70}")
        print(f"Invisible Inc Capture Session")
        print(f"{'='*70}")
        print(f"Session ID: {session_name}")
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Output: {self.session_dir}")
        print(f"{'='*70}")
        
        # Initialize components
        self.window_finder = UnifiedWindowFinder(owner, title)
        self.screenshot = ScreenshotCapture(self.session_dir, fps, self.window_finder, 
                                           skip_unfocused=False)  # DISABLED for now
        self.audio = AudioCapture(self.session_dir)
        self.input_logger = InputLogger(self.session_dir)
        
        self.start_time = None
        self.metadata = {
            'session_id': session_name,
            'platform': platform.system(),
            'platform_release': platform.release(),
            'fps': fps,
            'window_owner_name': str(owner),
            'window_title_pattern': str(title),
            'start_time': None,
            'end_time': None,
            'total_frames': 0,
            'total_events': 0
        }
        
    def start(self):
        """Start all capture components"""
        self.start_time = time.time()
        self.metadata['start_time'] = datetime.now().isoformat()
        
        print()
        self.screenshot.start()
        self.audio.start()
        self.input_logger.start()
        
        print(f"\n{'='*70}")
        print("CAPTURE IN PROGRESS")
        print(f"{'='*70}")
        print("\n→ Play Invisible Inc normally")
        print("→ All actions are being recorded")
        print("→ Press Ctrl+C or Ctrl+\\ to stop\n")
        
    def stop(self):
        """Stop all capture components and save metadata"""
        print(f"\n{'='*70}")
        print("Stopping capture...")
        print(f"{'='*70}\n")
        
        self.screenshot.stop()
        self.audio.stop()
        self.input_logger.stop()
        
        # Finalize metadata
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['total_frames'] = self.screenshot.frame_count
        self.metadata['total_events'] = self.input_logger.event_count
        self.metadata['duration_seconds'] = time.time() - self.start_time
        
        # Save session metadata
        metadata_file = self.session_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save frame metadata
        frames_metadata_file = self.session_dir / "frames_metadata.jsonl"
        with open(frames_metadata_file, 'w') as f:
            while not self.screenshot.metadata_queue.empty():
                frame_meta = self.screenshot.metadata_queue.get()
                f.write(json.dumps(frame_meta) + '\n')
        
        print(f"\n{'='*70}")
        print("CAPTURE COMPLETE")
        print(f"{'='*70}")
        print(f"\nSession: {self.session_dir}")
        print(f"Frames: {self.metadata['total_frames']}")
        print(f"Events: {self.metadata['total_events']}")
        print(f"Duration: {self.metadata['duration_seconds']:.1f}s\n")


def main():
    """Run a capture session"""
    import signal
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Capture Invisible Inc gameplay across all platforms'
    )
    parser.add_argument('--fps', type=float, default=2.0,
                       help='Capture frame rate (default: 2.0)')
    parser.add_argument('--owner', type=str, default="InvisibleInc",
                       help='Window owner/process name (default: InvisibleInc)')
    parser.add_argument('--title', type=str, default="Invisible",
                       help='Window title pattern (default: Invisible)')
    
    args = parser.parse_args()
    
    # Create capture session
    session = CaptureSession(
        fps=args.fps,
        window_owner_name=args.owner,
        window_title_pattern=args.title
    )
    
    # Handle both Ctrl+C (SIGINT) and Ctrl+\ (SIGQUIT) gracefully
    def signal_handler(sig, frame):
        signal_name = "SIGQUIT (Ctrl+\\)" if sig == signal.SIGQUIT else "SIGINT (Ctrl+C)"
        print(f"\n\nReceived {signal_name}...")
        session.stop()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)
    
    # Start capture
    session.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        session.stop()


if __name__ == "__main__":
    main()