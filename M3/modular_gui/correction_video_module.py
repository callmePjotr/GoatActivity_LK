"""
correction_gui_video.py
VideoPlayer component with frame index display
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time


class VideoPlayer(tk.Frame):
    """Video Player mit Play/Pause/Seek Controls und Frame Index"""
    
    def __init__(self, parent, video_path=None, width=640, height=480, metadata=None):
        super().__init__(parent)
        self.video_path = video_path
        self.cap = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.playback_thread = None
        self.canvas_width = width
        self.canvas_height = height
        self.metadata = metadata  # Clip metadata with start/end frames
        
        # Canvas für Video
        self.canvas = tk.Canvas(self, width=width, height=height, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Wait for canvas to be rendered
        self.canvas.update_idletasks()
        
        # Controls
        controls_frame = tk.Frame(self)
        controls_frame.pack(fill=tk.X, pady=5)
        
        self.play_btn = ttk.Button(controls_frame, text="▶ Play", command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(controls_frame, text="⏮ -10s", command=lambda: self.seek_relative(-10)).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="⏪ -1s", command=lambda: self.seek_relative(-1)).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="⏩ +1s", command=lambda: self.seek_relative(1)).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="⏭ +10s", command=lambda: self.seek_relative(10)).pack(side=tk.LEFT)
        
        # Timeline Slider
        self.timeline = ttk.Scale(controls_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                  command=self.on_timeline_change)
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Time Label mit Frame Index
        self.time_label = ttk.Label(controls_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.RIGHT, padx=5)
        
        self.frame_label = ttk.Label(controls_frame, text="Frame: 0 / 0")
        self.frame_label.pack(side=tk.RIGHT, padx=5)
        
        if video_path:
            self.load_video(video_path)
    
    def load_video(self, video_path):
        """Lade Video"""
        if self.cap:
            self.cap.release()
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open video: {video_path}")
            return
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.current_frame = 0
        
        self.timeline.config(to=self.total_frames - 1)
        self.show_frame(0)
    
    def show_frame(self, frame_number):
        """Zeige bestimmten Frame"""
        if not self.cap:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame_number
            
            # Konvertiere für Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get canvas size - update dynamically
            self.canvas.update_idletasks()
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            
            # Fallback to initial dimensions if canvas not yet visible
            if canvas_w <= 1:
                canvas_w = self.canvas_width
            if canvas_h <= 1:
                canvas_h = self.canvas_height
            
            h, w = frame_rgb.shape[:2]
            scale = min(canvas_w / w, canvas_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Ensure minimum size
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            
            # PIL Image
            img = Image.fromarray(frame_resized)
            self.photo = ImageTk.PhotoImage(img)
            
            # Display
            self.canvas.delete("all")
            self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.photo)
            
            # Update timeline
            self.timeline.set(frame_number)
            
            # Update time label
            current_time = frame_number / self.fps
            total_time = self.total_frames / self.fps
            self.time_label.config(text=f"{self._format_time(current_time)} / {self._format_time(total_time)}")
            
            # Update frame label with original frame index if metadata available
            if self.metadata:
                original_frame = self.metadata.get('start_frame', 0) + frame_number
                self.frame_label.config(text=f"Frame: {original_frame} (clip: {frame_number}/{self.total_frames})")
            else:
                self.frame_label.config(text=f"Frame: {frame_number} / {self.total_frames}")
    
    def _format_time(self, seconds):
        """Format seconds as MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def toggle_play(self):
        """Play/Pause Toggle"""
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def play(self):
        """Start Playback"""
        if not self.cap or self.is_playing:
            return
        
        self.is_playing = True
        self.play_btn.config(text="⏸ Pause")
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()
    
    def pause(self):
        """Pause Playback"""
        self.is_playing = False
        self.play_btn.config(text="▶ Play")
    
    def _playback_loop(self):
        """Playback Loop (runs in thread)"""
        frame_delay = 1.0 / self.fps
        
        while self.is_playing and self.current_frame < self.total_frames - 1:
            start_time = time.time()
            
            self.current_frame += 1
            self.after(0, self.show_frame, self.current_frame)
            
            # Sleep to maintain FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            time.sleep(sleep_time)
        
        self.is_playing = False
        self.after(0, lambda: self.play_btn.config(text="▶ Play"))
    
    def seek_relative(self, seconds):
        """Seek relative to current position"""
        if not self.cap:
            return
        
        target_frame = int(self.current_frame + seconds * self.fps)
        target_frame = max(0, min(target_frame, self.total_frames - 1))
        
        self.show_frame(target_frame)
    
    def on_timeline_change(self, value):
        """Timeline slider changed"""
        if not self.cap:
            return
        
        frame = int(float(value))
        if abs(frame - self.current_frame) > 2:  # Only seek if significant change
            self.show_frame(frame)
    
    def destroy(self):
        """Cleanup"""
        self.pause()
        if self.cap:
            self.cap.release()
        super().destroy()
