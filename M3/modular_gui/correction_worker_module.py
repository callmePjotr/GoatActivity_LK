"""
correction_gui_worker.py
Background file operations with progress tracking
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import queue
import shutil
import json
from pathlib import Path
import cv2
import numpy as np


class ProgressBar(tk.Frame):
    """Progress bar for file operations"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Progress label
        self.label = ttk.Label(self, text="")
        self.label.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self, mode='determinate', length=300)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Cancel button
        self.cancel_btn = ttk.Button(self, text="Cancel", command=self.cancel_operation)
        self.cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        self.cancel_btn.pack_forget()  # Hidden by default
        self.pack_forget()  # Hidden by default
        
        self.cancel_requested = False
    
    def show(self, text="Processing..."):
        """Show progress bar"""
        self.cancel_requested = False
        self.label.config(text=text)
        self.progress['value'] = 0
        self.cancel_btn.pack(side=tk.RIGHT, padx=5)
        self.pack(fill=tk.X, side=tk.BOTTOM, before=self.master.status_bar)
    
    def update(self, value, text=None):
        """Update progress"""
        self.progress['value'] = value
        if text:
            self.label.config(text=text)
    
    def hide(self):
        """Hide progress bar"""
        self.pack_forget()
        self.cancel_btn.pack_forget()
    
    def cancel_operation(self):
        """Request cancellation"""
        self.cancel_requested = True
        self.cancel_btn.config(state='disabled', text="Cancelling...")


class FileOperationWorker:
    """Handles file operations in background thread"""
    
    def __init__(self, progress_bar, status_bar, completion_callback):
        self.progress_bar = progress_bar
        self.status_bar = status_bar
        self.completion_callback = completion_callback
        self.queue = queue.Queue()
        self.worker_thread = None
        self.running = True
        self.affected_ids = set()  # Track which IDs were affected
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def _worker_loop(self):
        """Worker thread main loop"""
        while self.running:
            try:
                task = self.queue.get(timeout=0.1)
                if task is None:
                    break
                
                operation, args = task
                self.affected_ids.clear()  # Clear before operation
                
                if operation == 'merge_person':
                    self._merge_person(*args)
                elif operation == 'merge_clip':
                    self._merge_clip(*args)
                elif operation == 'delete_clip':
                    self._delete_clip(*args)
                elif operation == 'split_clip':
                    self._split_clip(*args)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Worker] Error: {e}")
                if self.status_bar:
                    self.status_bar.config(text=f"✗ Error: {str(e)}")
    
    def submit(self, operation, *args):
        """Submit operation to queue"""
        self.queue.put((operation, args))
    
    def get_affected_ids(self):
        """Get list of affected person IDs"""
        return list(self.affected_ids)
    
    def _merge_person(self, source_folder, target_folder, source_id, target_id):
        """Merge entire person folder"""
        self.affected_ids.add(source_id)
        self.affected_ids.add(target_id)
        
        self.progress_bar.show(f"Merging ID {source_id} → ID {target_id}...")
        
        clip_folders = [f for f in source_folder.iterdir() 
                       if f.is_dir() and (f.name.startswith('clip_') or '_part' in f.name)]
        
        total = len(clip_folders)
        moved_count = 0
        
        for i, clip_folder in enumerate(clip_folders):
            if self.progress_bar.cancel_requested:
                self.status_bar.config(text="✗ Merge cancelled")
                self.progress_bar.hide()
                return
            
            new_clip_name = self._get_unique_clip_name(target_folder, clip_folder.name)
            target_clip_folder = target_folder / new_clip_name
            
            shutil.move(str(clip_folder), str(target_clip_folder))
            moved_count += 1
            
            progress = ((i + 1) / total) * 100
            self.progress_bar.update(progress, f"Moving clip {i+1}/{total}...")
        
        # Remove empty source folder
        try:
            if source_folder.exists() and not any(source_folder.iterdir()):
                source_folder.rmdir()
        except:
            pass
        
        self.progress_bar.hide()
        if self.status_bar:
            self.status_bar.config(text=f"✓ Merged {moved_count} clips from ID {source_id} into ID {target_id}")
        
        # Call completion callback with affected IDs
        self.completion_callback(self.get_affected_ids())
    
    def _merge_clip(self, source_clip, target_folder, clip_name, source_id, target_id):
        """Merge single clip"""
        self.affected_ids.add(source_id)
        self.affected_ids.add(target_id)
        
        self.progress_bar.show(f"Moving {clip_name}...")
        
        new_clip_name = self._get_unique_clip_name(target_folder, clip_name)
        target_clip_folder = target_folder / new_clip_name
        
        # Move with progress
        shutil.move(str(source_clip), str(target_clip_folder))
        
        self.progress_bar.update(100, "Complete")
        time.sleep(0.5)
        
        self.progress_bar.hide()
        
        if new_clip_name != clip_name:
            messagebox.showinfo("Renamed", 
                              f"Clip renamed to avoid conflict:\n"
                              f"{clip_name} → {new_clip_name}")
        
        if self.status_bar:
            self.status_bar.config(text=f"✓ Moved {clip_name} from ID {source_id} to ID {target_id}")
        
        self.completion_callback(self.get_affected_ids())
    
    def _delete_clip(self, clip_folder, person_id, clip_name):
        """Delete clip"""
        self.affected_ids.add(person_id)
        
        self.progress_bar.show(f"Deleting {clip_name}...")
        
        shutil.rmtree(clip_folder)
        
        self.progress_bar.update(100, "Complete")
        time.sleep(0.3)
        self.progress_bar.hide()
        
        if self.status_bar:
            self.status_bar.config(text=f"✓ Deleted {clip_name} from ID {person_id}")
        
        self.completion_callback(self.get_affected_ids())
    
    def _split_clip(self, clip_folder, split_frame, metadata, person_id, clip_name):
        """Split clip at frame"""
        self.affected_ids.add(person_id)
        
        self.progress_bar.show(f"Splitting {clip_name}...")
        
        parent_folder = clip_folder.parent
        
        # Get frame data
        frames_data = metadata.get('frames', [])
        if not frames_data:
            raise ValueError("No frame data in metadata")
        
        # WICHTIG: split_frame ist der Index im CLIP (0-basiert)
        # frames_data enthält die original frame_idx aus dem gesamten Video

        # Split frame data basierend auf clip-lokalem Index
        part1_frames = frames_data[:split_frame]
        part2_frames = frames_data[split_frame:]

        if not part1_frames or not part2_frames:
            raise ValueError(f"Invalid split: part1={len(part1_frames)}, part2={len(part2_frames)} frames")

        # Create folders
        part1_folder = parent_folder / f"{clip_name}_part1"
        part2_folder = parent_folder / f"{clip_name}_part2"
        part1_folder.mkdir(exist_ok=True)
        part2_folder.mkdir(exist_ok=True)

        self.progress_bar.update(10, "Creating folders...")

        print(f"[Split] Splitting at clip frame {split_frame}")
        print(f"[Split] Part1: {len(part1_frames)} frames, Part2: {len(part2_frames)} frames")

        # Split videos
        for i, video_name in enumerate(['clip.mp4', 'clip_keypoints.mp4']):
            video_file = clip_folder / video_name
            if not video_file.exists():
                continue

            progress_base = 10 + (i * 40)
            self.progress_bar.update(progress_base, f"Splitting {video_name}...")

            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            part1_video = part1_folder / video_name
            part2_video = part2_folder / video_name

            writer1 = cv2.VideoWriter(str(part1_video), fourcc, fps, (width, height))
            writer2 = cv2.VideoWriter(str(part2_video), fourcc, fps, (width, height))

            if not writer1.isOpened() or not writer2.isOpened():
                cap.release()
                continue

            frame_idx = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx < split_frame:
                    writer1.write(frame)
                else:
                    writer2.write(frame)

                frame_idx += 1

                # Update progress
                if frame_idx % 30 == 0:
                    progress = progress_base + (frame_idx / total_frames * 40)
                    self.progress_bar.update(progress, f"Frame {frame_idx}/{total_frames}")

            cap.release()
            writer1.release()
            writer2.release()

            print(f"[Split] {video_name}: wrote {split_frame} frames to part1, {frame_idx - split_frame} frames to part2")

        self.progress_bar.update(90, "Creating metadata...")

        # Create metadata für Part 1
        part1_meta = {
            'start_frame': part1_frames[0]['frame_idx'],
            'end_frame': part1_frames[-1]['frame_idx'],
            'start_time': part1_frames[0]['timestamp'],
            'end_time': part1_frames[-1]['timestamp'],
            'duration': part1_frames[-1]['timestamp'] - part1_frames[0]['timestamp'],
            'num_frames': len(part1_frames),
            'frames': part1_frames
        }

        # Create metadata für Part 2
        part2_meta = {
            'start_frame': part2_frames[0]['frame_idx'],
            'end_frame': part2_frames[-1]['frame_idx'],
            'start_time': part2_frames[0]['timestamp'],
            'end_time': part2_frames[-1]['timestamp'],
            'duration': part2_frames[-1]['timestamp'] - part2_frames[0]['timestamp'],
            'num_frames': len(part2_frames),
            'frames': part2_frames
        }

        print(f"[Split] Part1 metadata: frames {part1_meta['start_frame']}-{part1_meta['end_frame']}, count={part1_meta['num_frames']}")
        print(f"[Split] Part2 metadata: frames {part2_meta['start_frame']}-{part2_meta['end_frame']}, count={part2_meta['num_frames']}")

        with open(part1_folder / 'metadata.json', 'w') as f:
            json.dump(part1_meta, f, indent=2)

        with open(part2_folder / 'metadata.json', 'w') as f:
            json.dump(part2_meta, f, indent=2)

        self.progress_bar.update(95, "Cleaning up...")

        # Delete original
        shutil.rmtree(clip_folder)

        self.progress_bar.update(100, "Complete")
        time.sleep(0.5)
        self.progress_bar.hide()

        if self.status_bar:
            self.status_bar.config(text=f"✓ Split {clip_name} into 2 parts ({len(part1_frames)} + {len(part2_frames)} frames)")

        self.completion_callback(self.get_affected_ids())

    def _get_unique_clip_name(self, target_folder, original_name):
        """Generate unique clip name"""
        target_path = target_folder / original_name

        if not target_path.exists():
            return original_name

        suffix_num = 1
        while True:
            if '_part' in original_name:
                new_name = f"{original_name}_v{suffix_num}"
            elif original_name.startswith('clip_'):
                new_name = f"{original_name}_v{suffix_num}"
            else:
                new_name = f"{original_name}_v{suffix_num}"

            if not (target_folder / new_name).exists():
                return new_name

            suffix_num += 1

            if suffix_num > 1000:
                timestamp = int(time.time())
                return f"{original_name}_{timestamp}"

    def stop(self):
        """Stop worker thread"""
        self.running = False
        self.queue.put(None)