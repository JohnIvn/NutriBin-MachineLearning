"""
NutriBin ML Desktop Application
A comprehensive tkinter-based GUI for managing YOLO model workflows
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from pathlib import Path
import subprocess
import sys
import threading
import json
from datetime import datetime
import os
import re
from PIL import Image, ImageTk
import cv2
import numpy as np
from scheduled_training import ScheduledTraining


class ParametersDialog(tk.Toplevel):
    """Modal dialog for script parameters"""
    def __init__(self, parent, script_name, current_params):
        super().__init__(parent)
        self.script_name = script_name
        self.params = current_params
        self.result = None
        
        # Modal setup
        self.title(f"⚙️  {script_name.replace('.py', '').replace('_', ' ').title()} Parameters")
        self.geometry("500x600")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        # Center on parent
        self.update_idletasks()
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_w = parent.winfo_width()
        parent_h = parent.winfo_height()
        dialog_w = self.winfo_width()
        dialog_h = self.winfo_height()
        x = parent_x + (parent_w - dialog_w) // 2
        y = parent_y + (parent_h - dialog_h) // 2
        self.geometry(f"+{x}+{y}")
        
        self.configure(bg="#ffffff")
        self.setup_ui()
        
    def setup_ui(self):
        """Setup dialog UI with better design"""
        # Header with gradient effect (using colored frame)
        header = tk.Frame(self, bg="#2E7D32", height=80)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        
        # Script icon and title
        title_frame = tk.Frame(header, bg="#2E7D32")
        title_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        icon_map = {
            "create_dataset.py": "📁",
            "train_model.py": "📈",
            "test_model.py": "🔍",
            "upgrade_model.py": "⬆️"
        }
        icon = icon_map.get(self.script_name, "⚙️")
        
        title_text = f"{icon} {self.script_name.replace('.py', '').replace('_', ' ').title()}"
        title = tk.Label(title_frame, text=title_text, font=("Arial", 14, "bold"), 
                        fg="white", bg="#2E7D32")
        title.pack(anchor=tk.W)
        
        subtitle_map = {
            "create_dataset.py": "Convert image folders to YOLO format",
            "train_model.py": "Train a new YOLO detection model",
            "test_model.py": "Run inference on test images",
            "upgrade_model.py": "Continue training from existing weights"
        }
        subtitle = tk.Label(title_frame, text=subtitle_map.get(self.script_name, ""), 
                           font=("Arial", 9), fg="#c8e6c9", bg="#2E7D32")
        subtitle.pack(anchor=tk.W, pady=(5, 0))
        
        # Main content area
        content_frame = tk.Frame(self, bg="#ffffff")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create scrollable area
        canvas = tk.Canvas(content_frame, bg="#ffffff", highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#ffffff")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Bind focus to canvas when hovering
        def on_enter(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def on_leave(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind("<Enter>", on_enter)
        canvas.bind("<Leave>", on_leave)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create parameter fields based on script
        self.fields = {}
        
        if self.script_name == "create_dataset.py":
            self.add_info_label(scrollable_frame, 
                              "📁 This script converts image folders to YOLO format.\n"
                              "✓ No parameters needed - just click Run Script!")
        
        elif self.script_name == "train_model.py":
            self.add_section_title(scrollable_frame, "Training Configuration")
            self.add_param_field(scrollable_frame, "Epochs", "epochs", "20", 
                               "Number of training iterations (default: 20)")
            self.add_param_field(scrollable_frame, "Image Size", "imgsz", "640", 
                               "Input image resolution (default: 640)")
            self.add_param_field(scrollable_frame, "Batch Size", "batch", "8", 
                               "Batch size for training (default: 8)")
            self.add_device_field(scrollable_frame, "Device", "device", "auto", 
                                "GPU/CPU selection (default: auto)")
        
        elif self.script_name == "test_model.py":
            self.add_section_title(scrollable_frame, "Model Configuration")
            self.add_model_path_field(scrollable_frame, "Model Path", "model", "best.pt", 
                                     "Path to model weights file")
            
            self.add_section_title(scrollable_frame, "Inference Settings")
            self.add_param_field(scrollable_frame, "Image Size", "imgsz", "640", 
                               "Input image resolution (default: 640)")
            self.add_param_field(scrollable_frame, "Confidence", "conf", "0.25", 
                               "Detection confidence threshold (default: 0.25)")
            self.add_device_field(scrollable_frame, "Device", "device", "auto", 
                                "GPU/CPU selection (default: auto)")
            self.add_checkbox_field(scrollable_frame, "Save Annotated Images", "save", True,
                                  "Save detection results with bounding boxes")
        
        elif self.script_name == "upgrade_model.py":
            self.add_section_title(scrollable_frame, "Base Model")
            self.add_model_path_field(scrollable_frame, "Base Weights", "base_weights", "best.pt",
                                     "Path to existing model to continue from")
            self.add_checkbox_field(scrollable_frame, "Auto-create Dataset", "auto_create", True,
                                  "Automatically create dataset if missing")
            
            self.add_section_title(scrollable_frame, "Training Configuration")
            self.add_param_field(scrollable_frame, "Epochs", "epochs", "10", 
                               "Number of training iterations (default: 10)")
            self.add_param_field(scrollable_frame, "Image Size", "imgsz", "640", 
                               "Input image resolution (default: 640)")
            self.add_param_field(scrollable_frame, "Batch Size", "batch", "8", 
                               "Batch size for training (default: 8)")
            self.add_device_field(scrollable_frame, "Device", "device", "auto", 
                                "GPU/CPU selection (default: auto)")
        
        elif self.script_name == "live_test.py":
            self.add_section_title(scrollable_frame, "Model & Camera")
            self.add_model_path_field(scrollable_frame, "Model Path", "model", "best.pt",
                                     "Path to model weights file")
            self.add_param_field(scrollable_frame, "Camera ID", "camera", "0",
                               "Camera device ID (default: 0 for webcam)")
            
            self.add_section_title(scrollable_frame, "Detection Settings")
            self.add_param_field(scrollable_frame, "Image Size", "imgsz", "640", 
                               "Input image resolution (default: 640)")
            self.add_param_field(scrollable_frame, "Confidence", "conf", "0.25", 
                               "Detection confidence threshold (default: 0.25)")
            self.add_device_field(scrollable_frame, "Device", "device", "auto", 
                                "GPU/CPU selection (default: auto)")
            
            self.add_section_title(scrollable_frame, "Tips")
            self.add_info_label(scrollable_frame,
                              "Press 'q' to quit\nPress 's' to save snapshot")

        elif self.script_name == "schedule_training.py":
            self.add_section_title(scrollable_frame, "Schedule Settings")
            self.add_param_field(scrollable_frame, "Time (HH:MM)", "time", "23:30",
                               "Local time to start training (24h format)")
            self.add_checkbox_field(scrollable_frame, "Repeat Daily", "repeat_daily", True,
                                  "Run the scheduled job every day at the specified time")
            self.add_checkbox_field(scrollable_frame, "Run Once", "once", False,
                                  "Run only the next occurrence and then stop")

            self.add_section_title(scrollable_frame, "Training Options")
            self.add_param_field(scrollable_frame, "Epochs", "epochs", "20", 
                               "Number of training iterations (default: 20)")
            self.add_param_field(scrollable_frame, "Image Size", "imgsz", "640", 
                               "Input image resolution (default: 640)")
            self.add_param_field(scrollable_frame, "Batch Size", "batch", "8", 
                               "Batch size for training (default: 8)")
            self.add_device_field(scrollable_frame, "Device", "device", "auto", 
                                "GPU/CPU selection (default: auto)")
            self.add_model_path_field(scrollable_frame, "Base Weights", "base_weights", "best.pt",
                                     "Optional base weights to continue training from")
            self.add_checkbox_field(scrollable_frame, "Auto-create Dataset", "auto_create", True,
                                  "Automatically create dataset if missing before training")
        
        # Footer with buttons
        footer = tk.Frame(self, bg="#f5f5f5", height=60)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        
        button_frame = tk.Frame(footer, bg="#f5f5f5")
        button_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", width=15,
                             bg="#e0e0e0", fg="#333333", font=("Arial", 10),
                             relief=tk.FLAT, cursor="hand2",
                             command=self.cancel, activebackground="#d0d0d0")
        cancel_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        run_btn = tk.Button(button_frame, text="▶ Run Script", width=15,
                           bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                           relief=tk.FLAT, cursor="hand2",
                           command=self.ok, activebackground="#45a049")
        run_btn.pack(side=tk.RIGHT)
    
    def add_section_title(self, parent, text):
        """Add section title"""
        frame = tk.Frame(parent, bg="#ffffff")
        frame.pack(fill=tk.X, pady=(15, 10))
        
        tk.Label(frame, text=text, font=("Arial", 10, "bold"), 
                fg="#2E7D32", bg="#ffffff").pack(anchor=tk.W)
        
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(8, 0))
    
    def add_info_label(self, parent, text):
        """Add info label"""
        frame = tk.Frame(parent, bg="#e8f5e9", relief=tk.FLAT, bd=1)
        frame.pack(fill=tk.X, pady=10)
        
        label = tk.Label(frame, text=text, font=("Arial", 9), justify=tk.LEFT,
                        bg="#e8f5e9", fg="#1b5e20")
        label.pack(fill=tk.X, padx=10, pady=10)
    
    def add_param_field(self, parent, label, key, default, help_text=""):
        """Add text entry parameter field"""
        frame = tk.Frame(parent, bg="#ffffff")
        frame.pack(fill=tk.X, pady=10)
        
        label_frame = tk.Frame(frame, bg="#ffffff")
        label_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(label_frame, text=label, font=("Arial", 9, "bold"), 
                fg="#333333", bg="#ffffff").pack(anchor=tk.W)
        
        if help_text:
            tk.Label(label_frame, text=help_text, font=("Arial", 8), 
                    fg="#999999", bg="#ffffff").pack(anchor=tk.W)
        
        value = self.params.get(key, default) if self.params else default
        var = tk.StringVar(value=str(value))
        
        entry = tk.Entry(frame, textvariable=var, font=("Arial", 10),
                        bg="#f9f9f9", fg="#333333", relief=tk.FLAT,
                        bd=0, highlightthickness=1, highlightcolor="#4CAF50",
                        highlightbackground="#e0e0e0")
        entry.pack(fill=tk.X, ipady=8)
        
        self.fields[key] = var
    
    def add_device_field(self, parent, label, key, default, help_text=""):
        """Add device combo field"""
        frame = tk.Frame(parent, bg="#ffffff")
        frame.pack(fill=tk.X, pady=10)
        
        label_frame = tk.Frame(frame, bg="#ffffff")
        label_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(label_frame, text=label, font=("Arial", 9, "bold"), 
                fg="#333333", bg="#ffffff").pack(anchor=tk.W)
        
        if help_text:
            tk.Label(label_frame, text=help_text, font=("Arial", 8), 
                    fg="#999999", bg="#ffffff").pack(anchor=tk.W)
        
        value = self.params.get(key, default) if self.params else default
        var = tk.StringVar(value=value)
        
        combo = ttk.Combobox(frame, textvariable=var, 
                            values=["auto", "cpu", "0", "1"],
                            state="readonly", font=("Arial", 10))
        combo.pack(fill=tk.X, ipady=6)
        
        self.fields[key] = var
    
    def add_model_path_field(self, parent, label, key, default, help_text=""):
        """Add model path field with browse button"""
        frame = tk.Frame(parent, bg="#ffffff")
        frame.pack(fill=tk.X, pady=10)
        
        label_frame = tk.Frame(frame, bg="#ffffff")
        label_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(label_frame, text=label, font=("Arial", 9, "bold"), 
                fg="#333333", bg="#ffffff").pack(anchor=tk.W)
        
        if help_text:
            tk.Label(label_frame, text=help_text, font=("Arial", 8), 
                    fg="#999999", bg="#ffffff").pack(anchor=tk.W)
        
        value = self.params.get(key, default) if self.params else default
        var = tk.StringVar(value=str(value))
        
        input_frame = tk.Frame(frame, bg="#ffffff")
        input_frame.pack(fill=tk.X, pady=(0, 0))
        
        entry = tk.Entry(input_frame, textvariable=var, font=("Arial", 10),
                        bg="#f9f9f9", fg="#333333", relief=tk.FLAT,
                        bd=0, highlightthickness=1, highlightcolor="#4CAF50",
                        highlightbackground="#e0e0e0")
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 8))
        
        browse_btn = tk.Button(input_frame, text="📁 Browse", width=10,
                             bg="#f0f0f0", fg="#333333", font=("Arial", 9),
                             relief=tk.FLAT, cursor="hand2",
                             command=lambda v=var: self.browse_file(v),
                             activebackground="#e0e0e0")
        browse_btn.pack(side=tk.LEFT, ipady=6)
        
        self.fields[key] = var
    
    def add_checkbox_field(self, parent, label, key, default, help_text=""):
        """Add checkbox field"""
        frame = tk.Frame(parent, bg="#ffffff")
        frame.pack(fill=tk.X, pady=10)
        
        value = self.params.get(key, default) if self.params else default
        var = tk.BooleanVar(value=value)
        
        check = tk.Checkbutton(frame, text=label, variable=var,
                             font=("Arial", 9), bg="#ffffff", fg="#333333",
                             activebackground="#ffffff", activeforeground="#333333",
                             selectcolor="#ffffff", highlightthickness=0)
        check.pack(anchor=tk.W)
        
        if help_text:
            help_label = tk.Label(frame, text=help_text, font=("Arial", 8), 
                                fg="#999999", bg="#ffffff")
            help_label.pack(anchor=tk.W, padx=(20, 0), pady=(2, 0))
        
        self.fields[key] = var
    
    def browse_file(self, var):
        """Browse for file"""
        filename = filedialog.askopenfilename(
            title="Select Model Weights",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
            initialdir=str(Path(__file__).resolve().parent.parent / "outputs")
        )
        if filename:
            var.set(filename)
    
    def ok(self):
        """Confirm and return parameters"""
        self.result = {k: v.get() if hasattr(v, 'get') else v for k, v in self.fields.items()}
        self.destroy()
    
    def cancel(self):
        """Cancel dialog"""
        self.result = None
        self.destroy()


class DesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NutriBin ML - YOLO Model Manager")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)
        
        # Professional color scheme
        self.bg_color = "#f5ede0"  # Cream beige like NutriBin
        self.header_color = "#2E7D32"  # Deep green
        self.accent_color = "#4CAF50"  # Light green
        self.card_color = "#ffffff"
        self.text_dark = "#333333"
        self.text_light = "#666666"
        self.border_color = "#e0d5c7"
        
        self.root.configure(bg=self.bg_color)
        
        # Paths
        self.script_dir = Path(__file__).resolve().parent
        self.yolo_root = self.script_dir.parent
        self.repo_root = self.yolo_root.parent
        
        # Process tracking
        self.current_process = None
        self.is_running = False
        self.camera_active = False
        self.output_container = None
        self.output_text = None
        
        # Parameters storage
        self.params = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main UI layout"""
        # Header with logo and title
        self.setup_header()
        
        # Main content area
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Content with script cards and output
        self.setup_content(main_container)
    
    def setup_header(self):
        """Setup header with NutriBin branding"""
        header = tk.Frame(self.root, bg=self.header_color, height=90)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        
        # Header content
        header_content = tk.Frame(header, bg=self.header_color)
        header_content.pack(fill=tk.BOTH, expand=True, padx=30, pady=12)
        
        # Logo area with actual image
        logo_frame = tk.Frame(header_content, bg=self.header_color)
        logo_frame.pack(side=tk.LEFT, anchor=tk.CENTER, padx=(0, 20))
        
        try:
            logo_path = self.yolo_root / 'public' / 'logo.png'
            # print(f"Looking for logo at: {logo_path}")
            # print(f"Logo exists: {logo_path.exists()}")
            if logo_path.exists():
                logo_img = Image.open(str(logo_path))
                logo_img = logo_img.resize((60, 73), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_img)
                logo_label = tk.Label(logo_frame, image=self.logo_photo, bg=self.header_color)
                logo_label.pack()
            else:
                raise FileNotFoundError(f"Logo not found at {logo_path}")
        except Exception as e:
            print(f"Error loading logo: {e}")
            # Fallback to emoji if image loading fails
            logo_text = tk.Label(logo_frame, text="🥬", font=("Arial", 32), bg=self.header_color)
            logo_text.pack()
        
        title_frame = tk.Frame(header_content, bg=self.header_color)
        title_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        title = tk.Label(title_frame, text="NutriBin ML", font=("Arial", 22, "bold"), 
                        fg="white", bg=self.header_color)
        title.pack(anchor=tk.W)
        
        subtitle = tk.Label(title_frame, text="YOLO Model Manager — Unexpected Features 2026", 
                           font=("Arial", 9), fg="#c8e6c9", bg=self.header_color)
        subtitle.pack(anchor=tk.W, pady=(3, 0))
        
        # Right side - Status and Stop button
        right_frame = tk.Frame(header_content, bg=self.header_color)
        right_frame.pack(side=tk.RIGHT, anchor=tk.CENTER, padx=(20, 0))
        
        # Status indicator
        self.status_label = tk.Label(right_frame, text="● Ready", 
                                    font=("Arial", 10, "bold"), fg="#c8e6c9", bg=self.header_color)
        self.status_label.pack(pady=(0, 8))
        
        # Stop button
        self.stop_btn = tk.Button(right_frame, text="⏹  Stop Process", width=16,
                                bg=self.accent_color, fg="white", font=("Arial", 9, "bold"),
                                relief=tk.FLAT, cursor="hand2",
                                command=self.stop_process, state=tk.DISABLED,
                                activebackground="#45a049", activeforeground="white",
                                disabledforeground="#999999")
        self.stop_btn.pack()
        
        # Store original button color for later state changes
        self.stop_btn_normal_color = self.accent_color
        self.stop_btn_active_color = "#45a049"
    
    def setup_content(self, parent):
        """Setup main content area"""
        # Create notebook-like layout with script cards on left, output on right
        content = tk.Frame(parent, bg=self.bg_color)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Script cards
        left_panel = tk.Frame(content, bg=self.bg_color)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        
        self.setup_script_cards(left_panel)
        
        # Right panel - Output console
        right_panel = tk.Frame(content, bg=self.bg_color)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_output_panel(right_panel)
    
    def setup_script_cards(self, parent):
        """Setup script execution cards with scrolling"""
        # Title
        title = tk.Label(parent, text="Workflow Steps", font=("Arial", 12, "bold"), 
                        fg=self.text_dark, bg=self.bg_color)
        title.pack(anchor=tk.W, pady=(0, 15))
        
        # Create scrollable container
        canvas = tk.Canvas(parent, bg=self.bg_color, highlightthickness=0, width=230)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.bg_color)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def on_enter(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def on_leave(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind("<Enter>", on_enter)
        canvas.bind("<Leave>", on_leave)
        
        # Pack canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Scripts info
        scripts = [
            ("1. Create Dataset", "create_dataset.py", "Convert image folders\nto YOLO format", "📁"),
            ("2. Train Model", "train_model.py", "Train a new YOLO\ndetection model", "📈"),
            ("3. Test Model", "test_model.py", "Run inference on\ntest images", "🔍"),
            ("4. Upgrade Model", "upgrade_model.py", "Continue training\nexisting weights", "⬆️"),
            ("5. Live Detection", "live_test.py", "Real-time detection\nwith webcam", "📹"),
            ("6. Schedule Training", "schedule_training.py", "Schedule training\nat a specified time", "⏰"),
        ]
        
        for title_text, script, desc, icon in scripts:
            self.create_script_card(scrollable_frame, title_text, script, desc, icon)
    
    def create_script_card(self, parent, title, script, description, icon):
        """Create a script execution card"""
        card = tk.Frame(parent, bg=self.card_color, relief=tk.FLAT, bd=1)
        card.pack(fill=tk.X, pady=8)
        
        # Add border effect
        card.configure(highlightthickness=1, highlightbackground=self.border_color)
        
        # Hover effect
        def on_enter(e, c=card):
            c.configure(bg="#fafaf8", highlightbackground=self.accent_color)
        
        def on_leave(e, c=card):
            c.configure(bg=self.card_color, highlightbackground=self.border_color)
        
        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)
        
        # Content frame
        content = tk.Frame(card, bg=self.card_color)
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)
        
        # Top row: icon, title, and clickable button
        top_frame = tk.Frame(content, bg=self.card_color)
        top_frame.pack(fill=tk.X, pady=(0, 8))
        
        icon_label = tk.Label(top_frame, text=icon, font=("Arial", 18), bg=self.card_color)
        icon_label.pack(side=tk.LEFT, padx=(0, 10))
        
        title_label = tk.Label(top_frame, text=title, font=("Arial", 10, "bold"), 
                              fg=self.text_dark, bg=self.card_color)
        title_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Description
        desc_label = tk.Label(content, text=description, font=("Arial", 9), 
                             fg=self.text_light, bg=self.card_color, justify=tk.LEFT)
        desc_label.pack(anchor=tk.W, pady=(0, 8))
        
        # Run button
        btn = tk.Button(content, text="▶  Run", width=20,
                       bg=self.accent_color, fg="white", font=("Arial", 9, "bold"),
                       relief=tk.FLAT, cursor="hand2",
                       command=lambda: self.run_script(script),
                       activebackground="#45a049", activeforeground="white")
        btn.pack(fill=tk.X)
        
        # Bind hover to card for better UX
        for widget in [card, content, top_frame, icon_label, title_label, desc_label]:
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            widget.bind("<Button-1>", lambda e, s=script: self.run_script(s))
    
    def setup_output_panel(self, parent):
        """Setup the right output panel with camera preview option"""
        # Header with console title and stats
        header_frame = tk.Frame(parent, bg=self.bg_color)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title = tk.Label(header_frame, text="📊 Console Output", 
                        font=("Arial", 12, "bold"), fg=self.text_dark, bg=self.bg_color)
        title.pack(side=tk.LEFT)
        
        self.line_count_label = tk.Label(header_frame, text="Lines: 0", 
                                        font=("Arial", 9), fg=self.text_light, bg=self.bg_color)
        self.line_count_label.pack(side=tk.RIGHT, padx=10)
        
        # Main output container - will switch between console and camera
        self.output_container = tk.Frame(parent, bg=self.bg_color)
        self.output_container.pack(fill=tk.BOTH, expand=True)
        
        # Setup console output
        self.setup_console_output(self.output_container)
    
    def setup_console_output(self, parent):
        """Setup the console text output"""
        # Clear the container
        for widget in parent.winfo_children():
            widget.destroy()
        
        # Output text area with scrollbar
        output_frame = tk.Frame(parent, bg="#0d1117", relief=tk.SUNKEN, bd=2)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#0d1117",
            fg="#c9d1d9",
            height=20,
            state=tk.DISABLED,
            insertbackground="#58a6ff",
            selectbackground="#3d444d",
            selectforeground="#c9d1d9"
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for colors with better styling
        self.output_text.tag_config("error", foreground="#ff7b72", font=("Consolas", 10, "bold"))
        self.output_text.tag_config("success", foreground="#3fb950", font=("Consolas", 10, "bold"))
        self.output_text.tag_config("warning", foreground="#d29922", font=("Consolas", 10))
        self.output_text.tag_config("info", foreground="#58a6ff", font=("Consolas", 10))
        self.output_text.tag_config("header", foreground="#79c0ff", font=("Consolas", 10, "bold"))
        self.output_text.tag_config("separator", foreground="#30363d")
    
    def setup_camera_preview(self, parent, camera_id=0, model_path=None):
        """Setup camera preview panel"""
        # Clear the container
        for widget in parent.winfo_children():
            widget.destroy()
        
        # Camera frame
        camera_frame = tk.Frame(parent, bg="#000000", relief=tk.SUNKEN, bd=2)
        camera_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for camera display
        self.camera_canvas = tk.Canvas(camera_frame, bg="#000000", highlightthickness=0)
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Info panel at bottom
        info_frame = tk.Frame(camera_frame, bg="#1a1a1a", height=40)
        info_frame.pack(fill=tk.X, side=tk.BOTTOM)
        info_frame.pack_propagate(False)
        
        self.camera_info_label = tk.Label(info_frame, text="📹 Camera: Starting...", 
                                         font=("Arial", 9), fg="#4CAF50", bg="#1a1a1a")
        self.camera_info_label.pack(anchor=tk.W, padx=10, pady=8)
        
        # Start camera capture thread
        self.camera_active = True
        self.camera_thread = threading.Thread(
            target=self._camera_capture_loop,
            args=(camera_id, model_path)
        )
        self.camera_thread.daemon = True
        self.camera_thread.start()
    
    def log_output(self, message, tag=""):
        """Log message to output panel with pretty formatting"""
        # Check if output_text exists and is valid
        if not hasattr(self, 'output_text') or self.output_text is None:
            return
        
        try:
            if not self.output_text.winfo_exists():
                return
        except:
            return
        
        self.output_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Strip ANSI codes from the message
        clean_message = self.strip_ansi_codes(str(message))
        
        # Skip empty lines and progress bar artifacts
        if not clean_message.strip():
            self.output_text.config(state=tk.DISABLED)
            return
        
        # Detect if it's an epoch result line (contains metrics)
        is_epoch_result = "Epoch" in clean_message and ("box_loss" in clean_message or "mAP" in clean_message)
        
        # Add icon based on tag
        icon = ""
        if tag == "error":
            icon = "❌"
        elif tag == "success":
            icon = "✅"
        elif tag == "warning":
            icon = "⚠️ "
        elif tag == "info":
            icon = "ℹ️ "
        elif tag == "header":
            icon = "🚀"
        elif is_epoch_result:
            icon = "📊"
        
        log_msg = f"{icon} [{timestamp}] {clean_message}\n" if icon else f"   [{timestamp}] {clean_message}\n"
        
        # Add extra spacing for epoch results
        if is_epoch_result and not tag:
            log_msg = f"\n{log_msg}"
        
        if tag:
            self.output_text.insert(tk.END, log_msg, tag)
        else:
            # Auto-detect severity from message content
            if "error" in clean_message.lower():
                self.output_text.insert(tk.END, log_msg, "error")
            elif "warning" in clean_message.lower():
                self.output_text.insert(tk.END, log_msg, "warning")
            elif "success" in clean_message.lower() or "completed" in clean_message.lower():
                self.output_text.insert(tk.END, log_msg, "success")
            else:
                self.output_text.insert(tk.END, log_msg)
        
        # Update line count
        line_count = int(self.output_text.index('end-1c').split('.')[0])
        self.line_count_label.config(text=f"Lines: {line_count}")
        
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.root.update()
    
    def clear_output(self):
        """Clear output panel"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)
    
    def strip_ansi_codes(self, text):
        """Remove ANSI color codes from text"""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
    
    def update_status(self, status, color="#4CAF50"):
        """Update status indicator"""
        status_symbol = "●" if color == "#4CAF50" else "⟳" if color == "#FF9800" else "✕"
        self.status_label.config(text=f"{status_symbol} {status}", foreground=color)
    
    def run_script(self, script_name):
        """Show parameter dialog and run a script"""
        if self.is_running:
            messagebox.showwarning("In Progress", "A process is already running!")
            return
        
        script_path = self.script_dir / script_name
        
        if not script_path.exists():
            messagebox.showerror("Error", f"Script not found: {script_path}")
            return
        
        # Show parameters dialog
        dialog = ParametersDialog(self.root, script_name, self.params)
        self.root.wait_window(dialog)
        
        if dialog.result is None:
            return  # User cancelled
        
        # Extract parameters from dialog
        params = dialog.result
        
        # Always ensure console output is ready
        try:
            if self.output_text is None or not self.output_text.winfo_exists():
                self.setup_console_output(self.output_container)
        except:
            self.setup_console_output(self.output_container)
        
        self.clear_output()
        
        # Pretty header
        self.log_output("═" * 80, "separator")
        self.log_output(f"STARTING: {script_name.upper()}", "header")
        self.log_output("═" * 80, "separator")
        self.log_output(f"Script: {script_path}", "info")
        self.log_output("")
        
        # Build command based on script
        cmd = [sys.executable, str(script_path)]
        
        if script_name == "train_model.py":
            cmd.extend([
                "--epochs", params.get("epochs", "20"),
                "--imgsz", params.get("imgsz", "640"),
                "--batch", params.get("batch", "8")
            ])
            self.log_output(f"📈 Parameters: --epochs {params.get('epochs', '20')} --imgsz {params.get('imgsz', '640')} --batch {params.get('batch', '8')}", "info")
        
        elif script_name == "test_model.py":
            cmd.extend([
                "--imgsz", params.get("imgsz", "640"),
                "--conf", params.get("conf", "0.25"),
                "--device", params.get("device", "auto")
            ])
            if params.get("model") and params.get("model") != "best.pt":
                cmd.extend(["--model", params.get("model")])
            if params.get("save", True):
                cmd.append("--save")
            self.log_output(f"🔍 Parameters: --imgsz {params.get('imgsz', '640')} --conf {params.get('conf', '0.25')} --device {params.get('device', 'auto')}", "info")
        
        elif script_name == "upgrade_model.py":
            cmd.extend([
                "--epochs", params.get("epochs", "10"),
                "--imgsz", params.get("imgsz", "640"),
                "--batch", params.get("batch", "8"),
                "--device", params.get("device", "auto")
            ])
            if params.get("base_weights"):
                cmd.extend(["--base-weights", params.get("base_weights")])
            if params.get("auto_create", True):
                cmd.append("--auto-create")
            self.log_output(f"⬆️  Parameters: --epochs {params.get('epochs', '10')} --imgsz {params.get('imgsz', '640')} --batch {params.get('batch', '8')}", "info")
        
        elif script_name == "live_test.py":
            if params.get("model"):
                cmd.extend(["--model", params.get("model")])
            cmd.extend([
                "--imgsz", params.get("imgsz", "640"),
                "--conf", params.get("conf", "0.25"),
                "--device", params.get("device", "auto"),
                "--camera", params.get("camera", "0")
            ])
            self.log_output(f"📹 Parameters: --imgsz {params.get('imgsz', '640')} --conf {params.get('conf', '0.25')} --camera {params.get('camera', '0')}", "info")
            
            # Show camera preview instead of console for live detection
            self.setup_camera_preview(
                self.output_container,
                camera_id=int(params.get("camera", "0")),
                model_path=params.get("model")
            )

        elif script_name == "schedule_training.py":
            # Build schedule command
            # Expected params: time, repeat_daily (True/False), once (True/False)
            if not params.get('time'):
                messagebox.showerror("Error", "Please specify a time for scheduled training (HH:MM).")
                return

            cmd.extend(["--time", params.get('time')])
            if params.get('repeat_daily') in (True, 'True', 'true', '1'):
                cmd.append('--repeat-daily')
            if params.get('once') in (True, 'True', 'true', '1'):
                cmd.append('--once')

            # Forward training options
            if params.get('epochs'):
                cmd.extend(['--epochs', params.get('epochs')])
            if params.get('imgsz'):
                cmd.extend(['--imgsz', params.get('imgsz')])
            if params.get('batch'):
                cmd.extend(['--batch', params.get('batch')])
            if params.get('device'):
                cmd.extend(['--device', params.get('device')])
            if params.get('base_weights'):
                cmd.extend(['--base-weights', params.get('base_weights')])
            if params.get('auto_create') in (True, 'True', 'true', '1'):
                cmd.append('--auto-create')
            self.log_output(f"⏰ Scheduled: time={params.get('time')} repeat_daily={params.get('repeat_daily')} once={params.get('once')}", "info")
        
        elif script_name == "create_dataset.py":
            self.log_output("📁 Creating dataset from image folders...", "info")
        
        self.log_output("")
        self.log_output("─" * 80, "separator")
        self.log_output("")
        
        # Store params for next time
        self.params = params
        
        # Run in thread
        thread = threading.Thread(target=self._run_script_thread, args=(cmd, script_name))
        thread.daemon = True
        thread.start()
    
    def _run_script_thread(self, cmd, script_name):
        """Run script in separate thread"""
        self.is_running = True
        self.stop_btn.config(state=tk.NORMAL, bg="#F44336", activebackground="#d32f2f")
        self.update_status("Running...", "#FF9800")
        
        try:
            self.log_output(f"Executing: {' '.join(cmd)}")
            self.log_output("")
            
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace',
                cwd=str(self.yolo_root)
            )
            
            # Read output line by line
            for line in self.current_process.stdout:
                line = line.rstrip()
                if line:
                    # Detect log levels
                    if "error" in line.lower():
                        self.log_output(line, "error")
                    elif "success" in line.lower() or "completed" in line.lower():
                        self.log_output(line, "success")
                    elif "warning" in line.lower():
                        self.log_output(line, "warning")
                    else:
                        self.log_output(line)
            
            returncode = self.current_process.wait()
            
            self.log_output("")
            self.log_output("─" * 80, "separator")
            if returncode == 0:
                self.log_output(f"🎉 {script_name} COMPLETED SUCCESSFULLY!", "success")
                self.log_output("═" * 80, "separator")
                self.update_status("Completed", "#4CAF50")
            else:
                self.log_output(f"❌ {script_name} FAILED with exit code {returncode}", "error")
                self.log_output("═" * 80, "separator")
                self.update_status("Failed", "#F44336")
                
        except Exception as e:
            self.log_output(f"✕ Error running script: {str(e)}", "error")
            self.update_status("Error", "#F44336")
        
        finally:
            self.current_process = None
            self.is_running = False
            self.stop_btn.config(state=tk.DISABLED, bg=self.stop_btn_normal_color)
            self.update_status("Ready", "#4CAF50")
    
    def stop_process(self):
        """Stop the current process"""
        # Stop camera if active
        if hasattr(self, 'camera_active'):
            self.camera_active = False
            # Give thread time to stop
            self.root.after(500)
        
        if self.current_process and self.is_running:
            try:
                if hasattr(self, 'output_text') and self.output_text and self.output_text.winfo_exists():
                    self.log_output("Stopping process...", "warning")
                self.current_process.terminate()
                self.current_process.wait(timeout=5)
                if hasattr(self, 'output_text') and self.output_text and self.output_text.winfo_exists():
                    self.log_output("Process stopped.", "warning")
            except Exception as e:
                if hasattr(self, 'output_text') and self.output_text and self.output_text.winfo_exists():
                    self.log_output(f"Error stopping process: {e}", "error")
                try:
                    self.current_process.kill()
                except:
                    pass
            finally:
                self.current_process = None
                self.is_running = False
                self.stop_btn.config(state=tk.DISABLED)
    
    def _camera_capture_loop(self, camera_id, model_path):
        """Capture camera frames and display with YOLO detections"""
        import time
        cap = None
        try:
            # Import YOLO
            from ultralytics import YOLO
            
            # Load model
            if model_path and str(model_path) != "best.pt":
                model = YOLO(str(model_path))
            else:
                # Find latest model
                weights_dir = self.yolo_root / 'outputs'
                timestamped_models = sorted(weights_dir.glob('*_best.pt'), reverse=True)
                if timestamped_models:
                    model = YOLO(str(timestamped_models[0]))
                else:
                    default_model = weights_dir / 'yolo_training' / 'weights' / 'best.pt'
                    if default_model.exists():
                        model = YOLO(str(default_model))
                    else:
                        if hasattr(self, 'camera_info_label'):
                            self.camera_info_label.config(text="❌ No model found!")
                        return
            
            # Open camera
            cap = cv2.VideoCapture(camera_id)
            time.sleep(1)  # Wait for camera to initialize
            
            if not cap.isOpened():
                if hasattr(self, 'camera_info_label'):
                    self.camera_info_label.config(text=f"❌ Cannot open camera {camera_id}")
                return
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            frame_count = 0
            detection_count = 0
            display_frame_skip = 0  # Skip every other frame for performance
            
            # Store photo reference
            self._current_photo = None
            self._camera_image_id = None
            
            if hasattr(self, 'camera_info_label'):
                self.camera_info_label.config(text="📹 Camera: Initializing...")
            
            # Warm up camera with 5 frames
            for _ in range(5):
                ret, _ = cap.read()
                if ret:
                    break
            
            while self.camera_active and self.is_running:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    continue
                
                frame_count += 1
                
                try:
                    # Run YOLO detection on every frame
                    results = model(frame, verbose=False)
                    
                    # Count detections
                    if results[0].boxes:
                        detection_count += len(results[0].boxes)
                    
                    # Draw results
                    annotated_frame = results[0].plot()
                    
                    if annotated_frame is None or annotated_frame.size == 0:
                        continue
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Get valid canvas size
                    try:
                        canvas_w = self.camera_canvas.winfo_width()
                        canvas_h = self.camera_canvas.winfo_height()
                        if canvas_w < 50 or canvas_h < 50:
                            continue
                    except:
                        break
                    
                    # Resize frame to fit canvas
                    h, w = frame_rgb.shape[:2]
                    if h > 0 and w > 0:
                        scale = min(canvas_w / w, canvas_h / h, 1.0)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        if new_w > 0 and new_h > 0:
                            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Convert to PIL and display
                    try:
                        pil_image = Image.fromarray(frame_rgb)
                        self._current_photo = ImageTk.PhotoImage(pil_image)
                        
                        # Calculate center position
                        img_w = pil_image.width
                        img_h = pil_image.height
                        center_x = canvas_w // 2
                        center_y = canvas_h // 2
                        
                        # Create or update image on canvas
                        if self._camera_image_id is None:
                            self._camera_image_id = self.camera_canvas.create_image(
                                center_x, center_y, image=self._current_photo, anchor=tk.CENTER
                            )
                        else:
                            try:
                                self.camera_canvas.itemconfig(self._camera_image_id, image=self._current_photo)
                                self.camera_canvas.coords(self._camera_image_id, center_x, center_y)
                            except:
                                # Canvas item was deleted, recreate
                                self._camera_image_id = self.camera_canvas.create_image(
                                    center_x, center_y, image=self._current_photo, anchor=tk.CENTER
                                )
                        
                        # Update status
                        if hasattr(self, 'camera_info_label'):
                            try:
                                self.camera_info_label.config(
                                    text=f"📹 Camera: {frame_count} frames | 🎯 Detections: {detection_count}"
                                )
                            except:
                                pass
                        
                        # Force GUI update
                        try:
                            self.camera_canvas.update()
                        except:
                            break
                    
                    except Exception as e:
                        # Image error - skip frame
                        continue
                
                except Exception as e:
                    # Detection error - skip frame
                    continue
            
            # Cleanup
            if cap:
                cap.release()
            
            if hasattr(self, 'camera_info_label'):
                try:
                    self.camera_info_label.config(text="📹 Camera: Stopped")
                except:
                    pass
        
        except Exception as e:
            if hasattr(self, 'camera_info_label'):
                try:
                    self.camera_info_label.config(text=f"❌ Error: {str(e)[:35]}")
                except:
                    pass
        finally:
            if cap:
                cap.release()


def main():
    root = tk.Tk()
    app = DesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
