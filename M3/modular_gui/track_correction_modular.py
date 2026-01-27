"""
Track Correction GUI - Modular Version
Main entry point that imports all modules

Structure:
- correction_gui_main.py (this file)
- correction_gui_video.py (VideoPlayer)
- correction_gui_tree.py (ClipTreeView)
- correction_gui_preview.py (ClipPreviewWindow)
- correction_gui_worker.py (FileOperationWorker, ProgressBar)

Usage:
python correction_gui_main.py --video tracked.avi --clips-dir track_crops
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import argparse

# Import modules (these should be in separate files)
try:
    from correction_video_module import VideoPlayer
    from correction_tree_module import ClipTreeView
    from correction_preview_module import ClipPreviewWindow
    from correction_worker_module import FileOperationWorker, ProgressBar
except ImportError:
    print("Error: Module files not found. Please ensure all modules are in the same directory:")
    print("- correction_gui_video.py")
    print("- correction_gui_tree.py")
    print("- correction_gui_preview.py")
    print("- correction_gui_worker.py")
    print("\nCreating modules from template...")
    # Modules will be provided separately


class TrackCorrectionGUI(tk.Tk):
    """Main GUI Application"""
    
    def __init__(self, video_path, clips_dir):
        super().__init__()
        
        self.title("Track Correction GUI - ID Swap Korrektur")
        
        # Start in fullscreen mode
        self.state('zoomed')  # Windows
        try:
            self.attributes('-zoomed', True)  # Linux
        except:
            pass
        
        # Allow toggling fullscreen with F11
        self.bind('<F11>', self.toggle_fullscreen)
        self.bind('<Escape>', lambda e: self.state('normal'))
        
        self.video_path = video_path
        self.clips_dir = Path(clips_dir)
        
        # Progress Bar
        self.progress_bar = ProgressBar(self)
        
        # File Operation Worker with partial refresh callback
        self.file_worker = FileOperationWorker(
            progress_bar=self.progress_bar,
            status_bar=None,  # Set after creating status bar
            completion_callback=self.on_operation_complete
        )
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI components"""
        # Menu Bar
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Reload Clips (F5)", command=self.reload_clips)
        file_menu.add_command(label="Force Full Refresh", command=self.force_refresh)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        # Main Container
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left Side: Video Player
        video_frame = ttk.LabelFrame(main_paned, text="Tracked Video", padding=10)
        main_paned.add(video_frame, weight=2)
        
        self.video_player = VideoPlayer(video_frame, video_path=self.video_path, 
                                       width=960, height=720)
        self.video_player.pack(fill=tk.BOTH, expand=True)
        
        # Right Side: Clip Tree
        tree_frame = ttk.LabelFrame(main_paned, text="Track Clips Structure", padding=10)
        main_paned.add(tree_frame, weight=1)
        
        # Search bar
        search_frame = tk.Frame(tree_frame)
        search_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(search_frame, text="Search ID:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.on_search)
        ttk.Entry(search_frame, textvariable=self.search_var).pack(side=tk.LEFT, 
                                                                    fill=tk.X, expand=True, padx=5)
        
        # Tree
        tree_container = tk.Frame(tree_frame)
        tree_container.pack(fill=tk.BOTH, expand=True)
        
        self.clip_tree = ClipTreeView(
            parent=tree_container,
            clips_dir=self.clips_dir,
            on_merge_callback=self.on_merge,
            on_reload_callback=self.reload_clips,
            file_worker=self.file_worker
        )
        self.clip_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Status Bar
        self.status_bar = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Update file worker with status bar
        self.file_worker.status_bar = self.status_bar
    
    def on_operation_complete(self, affected_ids):
        """Called when file operation completes - only reload affected IDs"""
        if affected_ids:
            # Partial reload - only affected IDs
            self.clip_tree.reload_specific_ids(affected_ids)
            id_list = ', '.join(affected_ids)
            print(f"[Reload] Updated only IDs: {id_list}")
        else:
            # Full reload fallback
            self.reload_clips()
    
    def on_merge(self, source_id, target_id, merge_type='person', clip_name=None):
        """Handle merge operation - delegates to file worker"""
        source_folder = self.clips_dir / f"person_{source_id}"
        target_folder = self.clips_dir / f"person_{target_id}"
        
        if not source_folder.exists():
            messagebox.showerror("Error", f"Source folder not found: {source_folder}")
            return
        
        if not target_folder.exists():
            target_folder.mkdir(parents=True, exist_ok=True)
            messagebox.showinfo("Info", f"Created new target folder: person_{target_id}")
        
        if merge_type == 'person':
            confirm = messagebox.askyesno(
                "Confirm Merge",
                f"Merge all clips from ID {source_id} into ID {target_id}?\n\n"
                f"This will move all clips and delete the source folder."
            )
            if not confirm:
                return
            self.file_worker.submit('merge_person', source_folder, target_folder, 
                                   source_id, target_id)
        
        elif merge_type == 'clip' and clip_name:
            source_clip = source_folder / clip_name
            if not source_clip.exists():
                messagebox.showerror("Error", f"Clip not found: {source_clip}")
                return
            confirm = messagebox.askyesno(
                "Confirm Merge",
                f"Move {clip_name} from ID {source_id} to ID {target_id}?"
            )
            if not confirm:
                return
            self.file_worker.submit('merge_clip', source_clip, target_folder, 
                                   clip_name, source_id, target_id)
    
    def reload_clips(self):
        """Full reload of clip structure"""
        self.clip_tree.load_clips()
        self.status_bar.config(text="✓ Full reload completed")
    
    def force_refresh(self):
        """Force full refresh (clear cache)"""
        self.clip_tree.clear_cache()
        self.clip_tree.load_clips()
        self.status_bar.config(text="✓ Full refresh completed")
    
    def on_search(self, *args):
        """Filter tree by search term"""
        search_term = self.search_var.get().lower()
        for item in self.clip_tree.get_children():
            item_text = self.clip_tree.item(item, 'text').lower()
            if search_term in item_text or not search_term:
                self.clip_tree.item(item, open=True)
            else:
                self.clip_tree.item(item, open=False)
    
    def toggle_fullscreen(self, event=None):
        """Toggle fullscreen mode"""
        current_state = self.state()
        if current_state == 'zoomed' or current_state == 'normal':
            try:
                self.attributes('-fullscreen', True)
            except:
                pass
        else:
            self.attributes('-fullscreen', False)
            self.state('zoomed')
    
    def destroy(self):
        """Cleanup on destroy"""
        if hasattr(self, 'file_worker'):
            self.file_worker.stop()
        super().destroy()


def main():
    parser = argparse.ArgumentParser(description='Track Correction GUI')
    parser.add_argument('--video', required=True, help='Visualized tracked video')
    parser.add_argument('--clips-dir', required=True, help='Extracted clips directory')
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}")
        return
    
    if not Path(args.clips_dir).exists():
        print(f"Error: Clips directory not found: {args.clips_dir}")
        return
    
    app = TrackCorrectionGUI(args.video, args.clips_dir)
    app.mainloop()


if __name__ == '__main__':
    main()
