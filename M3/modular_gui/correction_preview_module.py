"""
correction_gui_preview.py
ClipPreviewWindow with split functionality
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import json
import time
from correction_video_module import VideoPlayer


class ClipPreviewWindow(tk.Toplevel):
    """Pop-up Window für Clip Preview mit Split-Funktion"""
    
    def __init__(self, parent, clip_folder, clip_name, person_id, 
                 on_split_callback=None, file_worker=None):
        super().__init__(parent)
        self.title(f"Clip Preview: {clip_name}")
        self.geometry("800x700")
        
        self.clip_folder = Path(clip_folder)
        self.clip_name = clip_name
        self.person_id = person_id
        self.on_split_callback = on_split_callback
        self.file_worker = file_worker
        self.marked_for_deletion = False
        
        # Load metadata
        metadata_file = self.clip_folder / 'metadata.json'
        self.metadata = None
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.metadata = json.load(f)
        
        # Title with metadata info
        title_frame = tk.Frame(self)
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        
        title_text = f"ID {person_id} - {clip_name}"
        if self.metadata:
            start_frame = self.metadata.get('start_frame', 0)
            end_frame = self.metadata.get('end_frame', 0)
            title_text += f" (Frames {start_frame}-{end_frame})"
        
        ttk.Label(title_frame, text=title_text, 
                 font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
        
        # Find video file
        video_file = self.clip_folder / "clip.mp4"
        if not video_file.exists():
            video_file = self.clip_folder / "clip_keypoints.mp4"
        
        if not video_file.exists():
            messagebox.showerror("Error", f"Video file not found in {self.clip_folder}")
            self.destroy()
            return
        
        # Video Player with metadata
        self.player = VideoPlayer(self, video_path=video_file, width=720, height=540, 
                                 metadata=self.metadata)
        self.player.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Split Controls
        split_frame = ttk.LabelFrame(self, text="Split Clip at Current Frame", padding=10)
        split_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(split_frame, 
                 text="Use this to split clip if ID swap occurs within:").pack(anchor=tk.W)
        
        split_btn_frame = tk.Frame(split_frame)
        split_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(split_btn_frame, text="✂ Split Here", 
                  command=self.split_at_current_frame).pack(side=tk.LEFT, padx=5)
        ttk.Label(split_btn_frame, 
                 text="(Creates 2 new clips, deletes original)").pack(side=tk.LEFT, padx=5)
        
        # Action Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="✓ Keep Clip", command=self.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="✗ Delete Clip", 
                  command=self.mark_delete).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close", command=self.destroy).pack(side=tk.RIGHT, padx=5)
    
    def split_at_current_frame(self):
        """Split clip at current player frame"""
        split_frame = self.player.current_frame
        
        if split_frame <= 0 or split_frame >= self.player.total_frames - 1:
            messagebox.showwarning("Invalid Split", 
                                  "Cannot split at first or last frame. Choose a frame in the middle.")
            return
        
        # Show original frame number
        if self.metadata:
            original_frame = self.metadata.get('start_frame', 0) + split_frame
            confirm_text = (f"Split clip at frame {split_frame} (original frame {original_frame})?\n\n"
                          f"This will:\n"
                          f"1. Delete the original clip\n"
                          f"2. Create clip_part1 (clip frames 0-{split_frame-1})\n"
                          f"3. Create clip_part2 (clip frames {split_frame}-end)\n\n"
                          f"You can then move one part to the correct ID folder.")
        else:
            confirm_text = (f"Split clip at frame {split_frame}?\n\n"
                          f"This will:\n"
                          f"1. Delete the original clip\n"
                          f"2. Create clip_part1 (frames 0-{split_frame-1})\n"
                          f"3. Create clip_part2 (frames {split_frame}-end)\n\n"
                          f"You can then move one part to the correct ID folder.")
        
        confirm = messagebox.askyesno("Confirm Split", confirm_text)
        
        if not confirm:
            return
        
        # Load metadata
        metadata_file = self.clip_folder / 'metadata.json'
        if not metadata_file.exists():
            messagebox.showerror("Error", "Metadata file not found")
            return
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Stop player and release video
        self.player.pause()
        if self.player.cap:
            self.player.cap.release()
            self.player.cap = None
        
        self.update()
        time.sleep(0.2)
        
        # Submit to worker thread
        if self.file_worker:
            self.file_worker.submit('split_clip', self.clip_folder, split_frame, 
                                   metadata, self.person_id, self.clip_name)
            self.destroy()
        else:
            messagebox.showerror("Error", "File worker not available")
    
    def mark_delete(self):
        """Mark clip for deletion"""
        confirm = messagebox.askyesno("Confirm Delete", 
                                     f"Delete {self.clip_name}?")
        if confirm:
            if self.file_worker:
                person_folder = self.clip_folder.parent
                person_id = person_folder.name.replace('person_', '')
                self.file_worker.submit('delete_clip', self.clip_folder, person_id, self.clip_name)
                self.destroy()
            else:
                self.marked_for_deletion = True
                self.destroy()
