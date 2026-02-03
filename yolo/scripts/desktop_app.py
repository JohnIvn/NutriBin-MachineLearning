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
        self.root.geometry("1100x700")
        self.root.resizable(True, True)
        
        # Set color scheme
        self.bg_color = "#f0f0f0"
        self.fg_color = "#333333"
        self.accent_color = "#4CAF50"
        self.warning_color = "#FF9800"
        self.error_color = "#F44336"
        
        self.root.configure(bg=self.bg_color)
        
        # Paths
        self.script_dir = Path(__file__).resolve().parent
        self.yolo_root = self.script_dir.parent
        self.repo_root = self.yolo_root.parent
        
        # Process tracking
        self.current_process = None
        self.is_running = False
        
        # Parameters storage
        self.params = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI layout"""
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        title = ttk.Label(title_frame, text="NutriBin ML - YOLO Workflow Manager", 
                         font=("Arial", 16, "bold"))
        title.pack(side=tk.LEFT)
        
        # Status indicator
        self.status_label = ttk.Label(title_frame, text="● Ready", 
                                     font=("Arial", 10), foreground="#4CAF50")
        self.status_label.pack(side=tk.RIGHT)
        
        # Main content: left (buttons) and right (output)
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Script buttons
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        self.setup_button_panel(left_panel)
        
        # Right panel - Output console
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_output_panel(right_panel)
        
        # Footer
        footer_frame = ttk.Frame(main_container)
        footer_frame.pack(fill=tk.X, pady=(15, 0), side=tk.BOTTOM)
        
        ttk.Separator(main_container, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(10, 0))
        
        self.info_label = ttk.Label(footer_frame, text="Ready", 
                                   font=("Arial", 9), foreground="#666666")
        self.info_label.pack(side=tk.LEFT)
        
        clear_btn = ttk.Button(footer_frame, text="Clear Output", 
                             command=self.clear_output)
        clear_btn.pack(side=tk.RIGHT, padx=5)
    
    def setup_button_panel(self, parent):
        """Setup the left button panel"""
        # Instructions
        instr_frame = ttk.LabelFrame(parent, text="Instructions", padding=10)
        instr_frame.pack(fill=tk.X, pady=(0, 15))
        
        instructions = (
            "1. Create Dataset\n"
            "   Convert images to YOLO format\n\n"
            "2. Train Model\n"
            "   Train a new YOLO model\n\n"
            "3. Test Model\n"
            "   Run inference on test images\n\n"
            "4. Upgrade Model\n"
            "   Continue training existing model"
        )
        
        instr_label = ttk.Label(instr_frame, text=instructions, 
                              font=("Arial", 9), justify=tk.LEFT)
        instr_label.pack(fill=tk.X)
        
        # Script buttons
        buttons_frame = ttk.LabelFrame(parent, text="Scripts", padding=10)
        buttons_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Create Dataset Button
        btn1 = self.create_script_button(
            buttons_frame,
            "1. Create Dataset",
            "create_dataset.py",
            "Convert class folders to YOLO format"
        )
        btn1.pack(fill=tk.X, pady=5)
        
        # Train Model Button
        btn2 = self.create_script_button(
            buttons_frame,
            "2. Train Model",
            "train_model.py",
            "Train a new YOLO model"
        )
        btn2.pack(fill=tk.X, pady=5)
        
        # Test Model Button
        btn3 = self.create_script_button(
            buttons_frame,
            "3. Test Model",
            "test_model.py",
            "Run inference on test images"
        )
        btn3.pack(fill=tk.X, pady=5)
        
        # Upgrade Model Button
        btn4 = self.create_script_button(
            buttons_frame,
            "4. Upgrade Model",
            "upgrade_model.py",
            "Continue training from existing weights"
        )
        btn4.pack(fill=tk.X, pady=5)
        
        # Stop button
        stop_frame = ttk.Frame(parent)
        stop_frame.pack(fill=tk.X, pady=10)
        
        self.stop_btn = ttk.Button(stop_frame, text="⏹ Stop Process", 
                                  command=self.stop_process, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X)
    
    def create_script_button(self, parent, text, script, tooltip):
        """Create a styled button for a script"""
        btn = tk.Button(
            parent,
            text=text,
            command=lambda: self.run_script(script),
            bg=self.accent_color,
            fg="white",
            font=("Arial", 10, "bold"),
            relief=tk.RAISED,
            cursor="hand2",
            activebackground="#45a049",
            activeforeground="white"
        )
        
        def on_enter(e):
            btn.config(bg="#45a049")
        def on_leave(e):
            btn.config(bg=self.accent_color)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn
    
    def setup_output_panel(self, parent):
        """Setup the right output panel"""
        # Header with console title and stats
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 8))
        
        title = ttk.Label(header_frame, text="📊 Output Console", 
                         font=("Arial", 11, "bold"))
        title.pack(side=tk.LEFT)
        
        self.line_count_label = ttk.Label(header_frame, text="Lines: 0", 
                                         font=("Arial", 9), foreground="#666666")
        self.line_count_label.pack(side=tk.RIGHT, padx=10)
        
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
    
    def log_output(self, message, tag=""):
        """Log message to output panel with pretty formatting"""
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
        self.stop_btn.config(state=tk.NORMAL)
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
            self.stop_btn.config(state=tk.DISABLED)
            self.update_status("Ready", "#4CAF50")
    
    def stop_process(self):
        """Stop the current process"""
        if self.current_process and self.is_running:
            try:
                self.log_output("Stopping process...", "warning")
                self.current_process.terminate()
                self.current_process.wait(timeout=5)
                self.log_output("Process stopped.", "warning")
            except Exception as e:
                self.log_output(f"Error stopping process: {e}", "error")
                try:
                    self.current_process.kill()
                except:
                    pass
            finally:
                self.is_running = False
                self.stop_btn.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = DesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
