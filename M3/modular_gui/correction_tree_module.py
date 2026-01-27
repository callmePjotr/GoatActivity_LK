"""
correction_gui_tree.py
ClipTreeView with optimized partial reloading

Key optimizations:
- reload_specific_ids() only updates affected person folders
- Preserves expanded state and scroll position
- No full tree rebuild unless necessary
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import json
import shutil
import threading


class ClipTreeView(ttk.Treeview):
    """TreeView with intelligent partial reload"""
    
    def __init__(self, parent, clips_dir, on_merge_callback, on_reload_callback, file_worker):
        super().__init__(parent, selectmode='extended')
        
        self.clips_dir = Path(clips_dir)
        self.on_merge_callback = on_merge_callback
        self.on_reload_callback = on_reload_callback
        self.file_worker = file_worker
        
        self._cache = {}  # {track_id: [clip_names]}
        self._node_map = {}  # {track_id: tree_item_id} for fast lookup
        
        # Columns
        self['columns'] = ('clips', 'frames')
        self.heading('#0', text='Track ID')
        self.heading('clips', text='Clips')
        self.heading('frames', text='Total Frames')
        
        self.column('#0', width=150)
        self.column('clips', width=80)
        self.column('frames', width=100)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.yview)
        self.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bindings
        self.bind('<Double-Button-1>', self.on_double_click)
        self.bind('<Button-3>', self.show_context_menu)
        self.bind('<F5>', lambda e: self.load_clips())
        
        # Context menu
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(label="Preview Clip", command=self.preview_selected)
        self.context_menu.add_command(label="Split Clip...", command=self.split_selected)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Merge into...", command=self.merge_selected)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Delete Clip", command=self.delete_selected)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Refresh (F5)", command=self.load_clips)
        
        # Initial load
        self.load_clips()
    
    def clear_cache(self):
        """Clear cache for full reload"""
        self._cache = {}
        self._node_map = {}
    
    def load_clips(self):
        """Full reload of all clips"""
        if not self.clips_dir.exists():
            return
        
        print("[Tree] Full reload initiated")
        
        # Scan person folders
        person_folders = sorted([f for f in self.clips_dir.iterdir() 
                                if f.is_dir() and f.name.startswith('person_')])
        
        # Build current structure
        current_structure = {}
        for person_folder in person_folders:
            track_id = person_folder.name.replace('person_', '')
            clip_folders = sorted([f for f in person_folder.iterdir() 
                                  if f.is_dir() and (f.name.startswith('clip_') or '_part' in f.name)])
            current_structure[track_id] = [f.name for f in clip_folders]
        
        # Check if anything changed
        if current_structure == self._cache:
            print("[Tree] No changes detected, skipping reload")
            return
        
        # Remember expanded state
        expanded_items = {self.item(item, 'text').replace('ID ', ''): True
                         for item in self.get_children() if self.item(item, 'open')}
        
        # Clear tree
        for item in self.get_children():
            self.delete(item)
        self._node_map.clear()
        
        # Rebuild
        for person_folder in person_folders:
            track_id = person_folder.name.replace('person_', '')
            self._add_person_node(person_folder, track_id, 
                                 expanded=track_id in expanded_items)
        
        self._cache = current_structure
        print(f"[Tree] Full reload complete: {len(person_folders)} IDs")
    
    def reload_specific_ids(self, track_ids):
        """
        Optimized: Only reload specific person IDs
        
        Args:
            track_ids: List of track IDs (e.g., ['0002', '0019'])
        """
        if not track_ids:
            return
        
        print(f"[Tree] Partial reload: {track_ids}")
        
        for track_id in track_ids:
            person_folder = self.clips_dir / f"person_{track_id}"
            
            # Check if this ID exists
            if not person_folder.exists():
                # ID was deleted - remove from tree
                if track_id in self._node_map:
                    node = self._node_map[track_id]
                    self.delete(node)
                    del self._node_map[track_id]
                    if track_id in self._cache:
                        del self._cache[track_id]
                print(f"[Tree] Removed deleted ID: {track_id}")
                continue
            
            # Get current clips for this ID
            clip_folders = sorted([f for f in person_folder.iterdir() 
                                  if f.is_dir() and (f.name.startswith('clip_') or '_part' in f.name)])
            clip_names = [f.name for f in clip_folders]
            
            # Check if changed
            if track_id in self._cache and self._cache[track_id] == clip_names:
                print(f"[Tree] No changes for ID {track_id}, skipping")
                continue
            
            # Remember position and state
            old_index = None
            was_expanded = False
            if track_id in self._node_map:
                old_node = self._node_map[track_id]
                was_expanded = self.item(old_node, 'open')
                # Get current position in tree
                old_index = self.index(old_node)
                # Delete old node
                self.delete(old_node)
                del self._node_map[track_id]

            # Add updated node at original position
            self._add_person_node(person_folder, track_id, expanded=was_expanded, position=old_index)

            # Update cache
            self._cache[track_id] = clip_names
            print(f"[Tree] Updated ID {track_id}: {len(clip_names)} clips at position {old_index}")

    def _add_person_node(self, person_folder, track_id, expanded=False, position=None):
        """
        Add or update a single person node

        Args:
            person_folder: Path to person folder
            track_id: Track ID string
            expanded: Whether node should be expanded
            position: Optional position index (None = append to end)
        """
        clip_folders = [f for f in person_folder.iterdir()
                       if f.is_dir() and (f.name.startswith('clip_') or '_part' in f.name)]
        num_clips = len(clip_folders)

        # Count total frames
        total_frames = 0
        for clip_folder in clip_folders:
            metadata_file = clip_folder / 'metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        meta = json.load(f)
                        total_frames += meta.get('num_frames', 0)
                except:
                    pass

        # Insert person node at specific position or end
        person_text = f"ID {track_id}"
        if position is not None:
            # Insert at specific position
            person_node = self.insert('', position, text=person_text,
                                     values=(num_clips, total_frames),
                                     tags=('person',))
        else:
            # Append to end
            person_node = self.insert('', 'end', text=person_text,
                                     values=(num_clips, total_frames),
                                     tags=('person',))

        # Store in map
        self._node_map[track_id] = person_node

        # Add clip nodes
        for clip_folder in sorted(clip_folders):
            clip_name = clip_folder.name
            metadata_file = clip_folder / 'metadata.json'

            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        meta = json.load(f)
                        clip_frames = meta.get('num_frames', 0)
                        start_frame = meta.get('start_frame', 0)
                        end_frame = meta.get('end_frame', 0)
                        clip_info = f"{start_frame}-{end_frame}"
                except:
                    clip_frames = 0
                    clip_info = "?"
            else:
                clip_frames = 0
                clip_info = "?"

            self.insert(person_node, 'end', text=clip_name,
                       values=(clip_info, clip_frames),
                       tags=('clip',))

        # Restore expanded state
        if expanded:
            self.item(person_node, open=True)

    def on_double_click(self, event):
        """Double-click to preview clip"""
        self.preview_selected()

    def preview_selected(self):
        """Preview selected clip"""
        from correction_preview_module import ClipPreviewWindow

        selection = self.selection()
        if not selection:
            return

        item = selection[0]
        tags = self.item(item, 'tags')

        if 'clip' in tags:
            parent = self.parent(item)
            person_id = self.item(parent, 'text').replace('ID ', '')
            clip_name = self.item(item, 'text')

            person_folder = self.clips_dir / f"person_{person_id}"
            clip_folder = person_folder / clip_name

            if clip_folder.exists():
                ClipPreviewWindow(self, clip_folder, clip_name, person_id,
                                 on_split_callback=self.on_reload_callback,
                                 file_worker=self.file_worker)

    def split_selected(self):
        """Open clip for splitting"""
        self.preview_selected()

    def show_context_menu(self, event):
        """Show context menu on right-click"""
        item = self.identify_row(event.y)
        if item:
            self.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)

    def merge_selected(self):
        """Merge selected person/clip into another"""
        selection = self.selection()
        if not selection:
            return

        item = selection[0]
        tags = self.item(item, 'tags')

        if 'person' in tags:
            source_id = self.item(item, 'text').replace('ID ', '')
            target_id = self._select_target_id(exclude=source_id)
            if target_id:
                self.on_merge_callback(source_id, target_id, merge_type='person')

        elif 'clip' in tags:
            parent = self.parent(item)
            source_id = self.item(parent, 'text').replace('ID ', '')
            clip_name = self.item(item, 'text')
            target_id = self._select_target_id(exclude=source_id)
            if target_id:
                self.on_merge_callback(source_id, target_id, merge_type='clip',
                                      clip_name=clip_name)

    def _select_target_id(self, exclude=None):
        """Dialog to select target ID"""
        dialog = tk.Toplevel(self)
        dialog.title("Select Target ID")
        dialog.geometry("300x400")

        ttk.Label(dialog, text="Select target Track ID:",
                 font=('Arial', 10)).pack(pady=10)

        listbox = tk.Listbox(dialog)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for item in self.get_children():
            person_id = self.item(item, 'text').replace('ID ', '')
            if person_id != exclude:
                listbox.insert(tk.END, person_id)

        result = [None]

        def on_select():
            selection = listbox.curselection()
            if selection:
                result[0] = listbox.get(selection[0])
            dialog.destroy()

        ttk.Button(dialog, text="Select", command=on_select).pack(pady=5)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=5)

        dialog.wait_window()
        return result[0]

    def delete_selected(self):
        """Delete selected clip"""
        selection = self.selection()
        if not selection:
            return

        item = selection[0]
        tags = self.item(item, 'tags')

        if 'clip' in tags:
            parent = self.parent(item)
            person_id = self.item(parent, 'text').replace('ID ', '')
            clip_name = self.item(item, 'text')

            if messagebox.askyesno("Confirm",
                                  f"Delete clip {clip_name} from ID {person_id}?"):
                person_folder = self.clips_dir / f"person_{person_id}"
                clip_folder = person_folder / clip_name

                if clip_folder.exists() and self.file_worker:
                    self.file_worker.submit('delete_clip', clip_folder,
                                           person_id, clip_name)