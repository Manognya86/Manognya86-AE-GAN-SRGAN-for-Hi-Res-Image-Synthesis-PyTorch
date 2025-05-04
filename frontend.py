import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageOps
import requests
import io
import os
import logging
from datetime import datetime
import base64
import threading
import time
import webbrowser
from tkinter.font import Font
import json


class AEGANApplication:
    def __init__(self):
        self.root = tk.Tk()
        self.setup_application()
        self.show_start_page()

    def setup_application(self):
        """Initialize application-wide settings"""
        self.root.title("Project: AEGAN")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.setup_styles()
        self.setup_logging()
        
        # Initialize all required attributes
        self.jwt_token = None
        self.current_user = None
        self.original_image = None
        self.enhanced_image = None
        self.processing = False
        self.loading_frames = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
        self.loading_idx = 0
        self.api_base_url = "http://localhost:5000"
        self.original_image_data = None
        self.status_var = None
        self.feedback_btn = None
        self.enhance_btn = None
        self.save_btn = None
        self.recent_images = []

    def setup_styles(self):
        """Configure application-wide styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Custom styles for Star Wars theme
        self.style.configure('StarWars.TFrame', background='black')
        self.style.configure('StarWars.TLabel', 
                           background='black', 
                           foreground='white', 
                           font=('Orbitron', 10))
        self.style.configure('StarWars.TButton', 
                           font=('Orbitron', 10),
                           padding=8,
                           background='#4a4e69',
                           foreground='white',
                           borderwidth=0)
        self.style.map('StarWars.TButton',
                      background=[('active', '#16213e'), ('!active', '#4a4e69')],
                      foreground=[('active', 'white'), ('!disabled', 'white')])
        
        # Special styles
        self.style.configure('Title.TLabel', font=('Orbitron', 16, 'bold'), foreground='#00ff00')
        self.style.configure('Subtitle.TLabel', font=('Orbitron', 12, 'bold'), foreground='#00ff00')
        self.style.configure('Error.TLabel', foreground='red')
        self.style.configure('Success.TLabel', foreground='green')
        self.style.configure('Admin.TButton', background='#6a040f')
        self.style.configure('DevOps.TButton', background='#3a0ca3')
        self.style.configure('DataScientist.TButton', background='#4cc9f0')

    def setup_logging(self):
        """Configure application logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('aegan_app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def show_start_page(self):
        """Display the start page (Use Case 1)"""
        self.clear_window()
        self.root.title("Team R2D2 - AEGAN")
        
        # Load background image
        try:
            bg_image = Image.open("static/starwarsback.jpg")
            bg_photo = ImageTk.PhotoImage(bg_image.resize((1200, 800), Image.LANCZOS))
        except Exception as e:
            self.logger.warning(f"Couldn't load background image: {str(e)}")
            bg_photo = None
        
        # Background canvas
        self.bg_canvas = tk.Canvas(self.root, highlightthickness=0)
        self.bg_canvas.pack(fill='both', expand=True)
        
        if bg_photo:
            self.bg_canvas.create_image(0, 0, image=bg_photo, anchor='nw')
            self.bg_canvas.image = bg_photo  # Keep reference
        
        # Main container with backdrop effect
        backdrop_frame = ttk.Frame(self.bg_canvas, style='StarWars.TFrame')
        backdrop_frame.place(relx=0.5, rely=0.5, anchor='center', width=600, height=400)
        
        # Title
        ttk.Label(
            backdrop_frame,
            text="R2-D2",
            style='Title.TLabel',
            foreground='#00ff00'
        ).pack(pady=(20, 5))
        
        ttk.Label(
            backdrop_frame,
            text="Project: AEGAN",
            style='Subtitle.TLabel'
        ).pack(pady=(0, 20))
        
        ttk.Label(
            backdrop_frame,
            text="Initializing rebel system access...",
            style='StarWars.TLabel',
            font=('Orbitron', 9)
        ).pack(pady=(0, 30))
        
        # Login button
        login_btn = ttk.Button(
            backdrop_frame,
            text="Proceed to Login",
            command=self.show_login_window,
            style='StarWars.TButton'
        )
        login_btn.pack(pady=20, ipadx=20, ipady=5)

    def show_login_window(self):
        """Display the login window (Use Case 1)"""
        self.clear_window()
        self.root.title("R2-D2 Login")
        
        # Create a black canvas for the background
        self.bg_canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.bg_canvas.pack(fill='both', expand=True)
        
        # Load and play background video (simulated with static image)
        try:
            bg_image = Image.open("static/flyingStars.jpg")  # Using image as video placeholder
            bg_photo = ImageTk.PhotoImage(bg_image.resize((1200, 800), Image.LANCZOS))
            self.bg_canvas.create_image(0, 0, image=bg_photo, anchor='nw')
            self.bg_canvas.image = bg_photo
        except Exception as e:
            self.logger.warning(f"Couldn't load background: {str(e)}")
        
        # Login container with blur effect simulation
        login_frame = ttk.Frame(self.bg_canvas, style='StarWars.TFrame')
        login_frame.place(relx=0.5, rely=0.5, anchor='center', width=400, height=450)
        
        # Title
        ttk.Label(
            login_frame,
            text="R2-D2",
            style='Title.TLabel',
            foreground='#00a8ff'
        ).pack(pady=(20, 5))
        
        ttk.Label(
            login_frame,
            text="Astromech Access Terminal",
            style='StarWars.TLabel',
            font=('Orbitron', 10)
        ).pack(pady=(0, 20))
        
        # Login form
        form_frame = ttk.Frame(login_frame, style='StarWars.TFrame')
        form_frame.pack(pady=10)
        
        # Username field
        ttk.Label(
            form_frame,
            text="Username:",
            style='StarWars.TLabel'
        ).pack(anchor='w', pady=(5, 0))
        
        self.username_entry = ttk.Entry(
            form_frame,
            font=('Orbitron', 10),
            width=30
        )
        self.username_entry.pack(pady=5, ipady=5)
        
        # Password field
        ttk.Label(
            form_frame,
            text="Password:",
            style='StarWars.TLabel'
        ).pack(anchor='w', pady=(5, 0))
        
        self.password_entry = ttk.Entry(
            form_frame,
            show="*",
            font=('Orbitron', 10),
            width=30
        )
        self.password_entry.pack(pady=5, ipady=5)
        
        # Login button
        login_btn = ttk.Button(
            form_frame,
            text="Login as R2 Unit",
            command=self.handle_login,
            style='StarWars.TButton'
        )
        login_btn.pack(pady=20, ipadx=20, ipady=5)
        
        # Register link
        register_frame = ttk.Frame(login_frame, style='StarWars.TFrame')
        register_frame.pack(pady=10)
        
        ttk.Label(
            register_frame,
            text="New user?",
            style='StarWars.TLabel',
            font=('Orbitron', 9)
        ).pack(side='left')
        
        register_btn = ttk.Button(
            register_frame,
            text="Register here",
            command=self.show_register_window,
            style='StarWars.TButton'
        )
        register_btn.pack(side='left', padx=5)
        
        # Status label
        self.login_status = ttk.Label(
            login_frame,
            text="",
            style='StarWars.TLabel'
        )
        self.login_status.pack(pady=10)
        
        # Footer
        ttk.Label(
            login_frame,
            text="Systems secured by the Rebel Alliance",
            style='StarWars.TLabel',
            font=('Orbitron', 8, 'italic')
        ).pack(side='bottom', pady=10)

    def show_register_window(self):
        """Display the registration window (Use Case 1)"""
        self.clear_window()
        self.root.title("R2-D2 Registration")
        
        # Load background image
        try:
            bg_image = Image.open("static/reg.jpg")
            bg_photo = ImageTk.PhotoImage(bg_image.resize((1200, 800), Image.LANCZOS))
        except Exception as e:
            self.logger.warning(f"Couldn't load background image: {str(e)}")
            bg_photo = None
        
        # Background canvas
        self.bg_canvas = tk.Canvas(self.root, highlightthickness=0)
        self.bg_canvas.pack(fill='both', expand=True)
        
        if bg_photo:
            self.bg_canvas.create_image(0, 0, image=bg_photo, anchor='nw')
            self.bg_canvas.image = bg_photo  # Keep reference
        
        # Register container with backdrop effect
        backdrop_frame = ttk.Frame(self.bg_canvas, style='StarWars.TFrame')
        backdrop_frame.place(relx=0.5, rely=0.5, anchor='center', width=500, height=500)
        
        # Title
        ttk.Label(
            backdrop_frame,
            text="R2-D2",
            style='Title.TLabel',
            foreground='yellow'
        ).pack(pady=(20, 5))
        
        ttk.Label(
            backdrop_frame,
            text="Register for AEGAN",
            style='Subtitle.TLabel'
        ).pack(pady=(0, 20))
        
        # Registration form
        form_frame = ttk.Frame(backdrop_frame, style='StarWars.TFrame')
        form_frame.pack(pady=10)
        
        # Username field
        ttk.Label(
            form_frame,
            text="Username:",
            style='StarWars.TLabel'
        ).pack(anchor='w', pady=(5, 0))
        
        self.reg_username_entry = ttk.Entry(
            form_frame,
            font=('Orbitron', 10),
            width=30
        )
        self.reg_username_entry.pack(pady=5, ipady=5)
        
        # Password field
        ttk.Label(
            form_frame,
            text="Password:",
            style='StarWars.TLabel'
        ).pack(anchor='w', pady=(5, 0))
        
        self.reg_password_entry = ttk.Entry(
            form_frame,
            show="*",
            font=('Orbitron', 10),
            width=30
        )
        self.reg_password_entry.pack(pady=5, ipady=5)
        
        # Confirm Password field
        ttk.Label(
            form_frame,
            text="Confirm Password:",
            style='StarWars.TLabel'
        ).pack(anchor='w', pady=(5, 0))
        
        self.reg_confirm_entry = ttk.Entry(
            form_frame,
            show="*",
            font=('Orbitron', 10),
            width=30
        )
        self.reg_confirm_entry.pack(pady=5, ipady=5)
        
        # Register button
        register_btn = ttk.Button(
            form_frame,
            text="Register",
            command=self.handle_register,
            style='StarWars.TButton'
        )
        register_btn.pack(pady=20, ipadx=20, ipady=5)
        
        # Back to login link
        login_frame = ttk.Frame(backdrop_frame, style='StarWars.TFrame')
        login_frame.pack(pady=10)
        
        ttk.Label(
            login_frame,
            text="Already have an account?",
            style='StarWars.TLabel',
            font=('Orbitron', 9)
        ).pack(side='left')
        
        login_btn = ttk.Button(
            login_frame,
            text="Login",
            command=self.show_login_window,
            style='StarWars.TButton'
        )
        login_btn.pack(side='left', padx=5)
        
        # Status label
        self.reg_status = ttk.Label(
            backdrop_frame,
            text="",
            style='StarWars.TLabel'
        )
        self.reg_status.pack(pady=10)

    def show_dashboard(self):
        """Display the main dashboard (Use Case 1)"""
        self.clear_window()
        self.root.title(f"Project: AEGAN - Welcome {self.current_user.get('username', 'User')}")
        
        # Load background image
        try:
            bg_image = Image.open("static/starwars.jpg")
            bg_photo = ImageTk.PhotoImage(bg_image.resize((1200, 800), Image.LANCZOS))
        except Exception as e:
            self.logger.warning(f"Couldn't load background image: {str(e)}")
            bg_photo = None
        
        # Background canvas
        self.bg_canvas = tk.Canvas(self.root, highlightthickness=0)
        self.bg_canvas.pack(fill='both', expand=True)
        
        if bg_photo:
            self.bg_canvas.create_image(0, 0, image=bg_photo, anchor='nw')
            self.bg_canvas.image = bg_photo  # Keep reference
        
        # Main container with semi-transparent effect
        main_frame = ttk.Frame(self.bg_canvas, style='StarWars.TFrame')
        main_frame.place(relx=0.5, rely=0.5, anchor='center', width=1100, height=700)
        
        # Header with role-based buttons
        self.setup_header(main_frame)
        
        # Image processing section
        self.setup_image_processing(main_frame)
        
        # Recent images section
        self.setup_recent_images(main_frame)
        
        # Status bar
        self.setup_status_bar(main_frame)
        
        # Load recent images
        self.load_recent_images()

    def setup_header(self, parent):
        """Setup the header with role-based buttons (Use Case 3)"""
        header_frame = ttk.Frame(parent, style='StarWars.TFrame')
        header_frame.pack(fill='x', padx=20, pady=15)
        
        ttk.Label(
            header_frame,
            text="Project: AEGAN",
            style='Title.TLabel'
        ).pack(side='left')
        
        self.user_label = ttk.Label(
            header_frame,
            text=f"Welcome, {self.current_user.get('username', 'User')} ({self.current_user.get('role', 'user')})",
            style='StarWars.TLabel'
        )
        self.user_label.pack(side='left', padx=20)
        
        # Role-specific buttons
        if self.current_user.get('role') == 'admin':
            ttk.Button(
                header_frame,
                text="User Management",
                command=self.show_admin_panel,
                style='Admin.TButton'
            ).pack(side='left', padx=5)
        
        if self.current_user.get('role') in ['data_scientist', 'admin']:
            ttk.Button(
                header_frame,
                text="Model Training",
                command=self.show_training_panel,
                style='DataScientist.TButton'
            ).pack(side='left', padx=5)
            
            ttk.Button(
                header_frame,
                text="Feedback Analysis",
                command=self.show_feedback_panel,
                style='DataScientist.TButton'
            ).pack(side='left', padx=5)
        
        if self.current_user.get('role') in ['devops', 'admin']:
            ttk.Button(
                header_frame,
                text="Model Deployment",
                command=self.show_deployment_panel,
                style='DevOps.TButton'
            ).pack(side='left', padx=5)
            
            ttk.Button(
                header_frame,
                text="System Monitoring",
                command=self.show_monitoring_panel,
                style='DevOps.TButton'
            ).pack(side='left', padx=5)
        
        # Logout button
        ttk.Button(
            header_frame,
            text="Logout",
            command=self.handle_logout,
            style='StarWars.TButton'
        ).pack(side='right')

    def setup_image_processing(self, parent):
        """Setup image processing controls (Use Case 1)"""
        control_frame = ttk.Frame(parent, style='StarWars.TFrame')
        control_frame.pack(fill='x', padx=20, pady=10)
        
        self.upload_btn = ttk.Button(
            control_frame,
            text="Upload Image",
            command=self.upload_image,
            style='StarWars.TButton'
        )
        self.upload_btn.pack(side='left', padx=5)
        
        self.enhance_btn = ttk.Button(
            control_frame,
            text="Enhance Image",
            command=self.enhance_image,
            state='disabled',
            style='StarWars.TButton'
        )
        self.enhance_btn.pack(side='left', padx=5)
        
        self.save_btn = ttk.Button(
            control_frame,
            text="Save Result",
            command=self.save_result,
            state='disabled',
            style='StarWars.TButton'
        )
        self.save_btn.pack(side='left', padx=5)
        
        self.feedback_btn = ttk.Button(
            control_frame,
            text="Provide Feedback",
            command=self.provide_feedback,
            state='disabled',
            style='StarWars.TButton'
        )
        self.feedback_btn.pack(side='left', padx=5)
        
        self.clear_btn = ttk.Button(
            control_frame,
            text="Clear All",
            command=self.clear_all,
            style='StarWars.TButton'
        )
        self.clear_btn.pack(side='left', padx=5)
        
        # Image display area
        image_frame = ttk.Frame(parent, style='StarWars.TFrame')
        image_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Original image box
        original_frame = ttk.Frame(image_frame, style='StarWars.TFrame')
        original_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        ttk.Label(
            original_frame,
            text="Original Image",
            style='Subtitle.TLabel'
        ).pack()
        
        self.original_canvas = tk.Canvas(
            original_frame,
            bg='#16213e',
            highlightthickness=0
        )
        self.original_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Enhanced image box
        enhanced_frame = ttk.Frame(image_frame, style='StarWars.TFrame')
        enhanced_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        ttk.Label(
            enhanced_frame,
            text="Enhanced Result",
            style='Subtitle.TLabel'
        ).pack()
        
        self.enhanced_canvas = tk.Canvas(
            enhanced_frame,
            bg='#16213e',
            highlightthickness=0
        )
        self.enhanced_canvas.pack(fill='both', expand=True, padx=10, pady=10)

    def setup_recent_images(self, parent):
        """Setup recent images section (Use Case 1)"""
        recent_frame = ttk.Frame(parent, style='StarWars.TFrame')
        recent_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(
            recent_frame,
            text="Recent Images",
            style='Subtitle.TLabel'
        ).pack(anchor='w')
        
        self.recent_images_container = ttk.Frame(recent_frame, style='StarWars.TFrame')
        self.recent_images_container.pack(fill='x', pady=5)

    def setup_status_bar(self, parent):
        """Setup status bar at bottom"""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        ttk.Label(
            parent,
            textvariable=self.status_var,
            relief='sunken',
            anchor='w',
            font=('Orbitron', 9),
            style='StarWars.TLabel'
        ).pack(fill='x', padx=20, pady=(0, 10))

    def handle_login(self):
        """Handle user login (Use Case 1)"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if not username or not password:
            self.login_status.config(text="Username and password required", style='Error.TLabel')
            return
            
        try:
            response = requests.post(
                f"{self.api_base_url}/api/login",
                json={"username": username, "password": password},
                timeout=5
            )
            
            if response.status_code == 200:
                jwt_token = response.cookies.get('access_token')
                if jwt_token:
                    self.jwt_token = jwt_token
                    self.current_user = response.json()
                    self.show_dashboard()
                else:
                    self.login_status.config(text="Login failed - no token received", style='Error.TLabel')
            else:
                error_msg = response.json().get('error', 'Login failed')
                self.login_status.config(text=error_msg, style='Error.TLabel')
        except requests.exceptions.ConnectionError:
            self.login_status.config(text="Connection failed", style='Error.TLabel')
        except Exception as e:
            self.login_status.config(text=f"Error: {str(e)}", style='Error.TLabel')

    def handle_register(self):
        """Handle user registration (Use Case 1)"""
        username = self.reg_username_entry.get()
        password = self.reg_password_entry.get()
        confirm = self.reg_confirm_entry.get()
        
        if not username or not password:
            self.reg_status.config(text="Username and password required", style='Error.TLabel')
            return
            
        if password != confirm:
            self.reg_status.config(text="Passwords do not match", style='Error.TLabel')
            return
            
        try:
            response = requests.post(
                f"{self.api_base_url}/api/register",
                json={"username": username, "password": password},
                timeout=5
            )
            
            if response.status_code == 201:
                self.reg_status.config(text="Registration successful! Please login.", style='Success.TLabel')
                self.root.after(2000, self.show_login_window)
            else:
                error_msg = response.json().get('error', 'Registration failed')
                self.reg_status.config(text=error_msg, style='Error.TLabel')
        except requests.exceptions.ConnectionError:
            self.reg_status.config(text="Connection failed", style='Error.TLabel')
        except Exception as e:
            self.reg_status.config(text=f"Error: {str(e)}", style='Error.TLabel')

    def handle_logout(self):
        """Handle user logout (Use Case 1)"""
        try:
            response = requests.post(
                f"{self.api_base_url}/api/logout",
                headers={"Authorization": f"Bearer {self.jwt_token}"},
                timeout=5
            )
            
            if response.status_code == 200:
                self.jwt_token = None
                self.current_user = None
                self.show_start_page()
            else:
                messagebox.showerror("Error", "Logout failed")
        except Exception as e:
            self.logger.error(f"Error during logout: {str(e)}")
            self.show_start_page()

    def upload_image(self):
        """Handle image upload (Use Case 1)"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.gif"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Image",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                with open(filepath, 'rb') as f:
                    self.original_image_data = f.read()
                    self.original_image = Image.open(io.BytesIO(self.original_image_data))
                    self.display_image(self.original_image, self.original_canvas)
                    
                    self.enhance_btn.config(state='normal')
                    self.save_btn.config(state='disabled')
                    self.feedback_btn.config(state='disabled')
                    self.status_var.set(f"Loaded: {os.path.basename(filepath)}")
            except Exception as e:
                self.logger.error(f"Error uploading image: {str(e)}")
                messagebox.showerror("Error", "Failed to load image")

    def enhance_image(self):
        """Enhance the uploaded image (Use Case 1)"""
        if not hasattr(self, 'original_image_data'):
            messagebox.showwarning("Warning", "Please upload an image first")
            return
            
        if self.processing:
            return
            
        self.processing = True
        self.enhance_btn.config(state='disabled')
        self.status_var.set("Processing image...")
        
        # Start enhancement in a separate thread
        threading.Thread(
            target=self.process_enhancement,
            daemon=True
        ).start()

    def process_enhancement(self):
        """Process image enhancement in background thread (Use Case 1)"""
        try:
            # Prepare the request with proper authentication
            headers = {
                "Authorization": f"Bearer {self.jwt_token}",
                "Accept": "application/json"
            }
            
            # Create a proper file upload
            files = {
                'file': ('image.png', self.original_image_data, 'image/png')
            }
            
            response = requests.post(
                f"{self.api_base_url}/api/enhance",
                files=files,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                enhanced_url = result.get('enhanced_url')
                
                if enhanced_url:
                    # Download enhanced image with authentication
                    enh_response = requests.get(
                        f"{self.api_base_url}{enhanced_url}",
                        headers=headers
                    )
                    if enh_response.status_code == 200:
                        self.enhanced_image = Image.open(io.BytesIO(enh_response.content))
                        
                        # Update UI in main thread
                        self.root.after(0, self.update_enhanced_display)
                        self.root.after(0, lambda: self.status_var.set("Enhancement complete!"))
                        self.root.after(0, self.load_recent_images)
                        
                        # Enable feedback button
                        self.feedback_btn.config(state='normal')
                    else:
                        self.root.after(0, lambda: self.status_var.set("Failed to load enhanced image"))
                        self.root.after(0, lambda: messagebox.showerror("Error", "Failed to load enhanced image"))
                else:
                    self.root.after(0, lambda: self.status_var.set("Invalid server response"))
            else:
                error_msg = response.json().get('error', 'Enhancement failed')
                self.root.after(0, lambda: self.status_var.set(f"Error: {error_msg}"))
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        except requests.exceptions.RequestException as e:
            self.root.after(0, lambda: self.status_var.set(f"Network error: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", "Network error occurred"))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", "An unexpected error occurred"))
        finally:
            self.processing = False
            self.root.after(0, lambda: self.enhance_btn.config(state='normal'))

    def update_enhanced_display(self):
        """Update UI with enhanced image (Use Case 1)"""
        if self.enhanced_image:
            self.display_image(self.enhanced_image, self.enhanced_canvas)
            self.save_btn.config(state='normal')

    def provide_feedback(self):
        """Provide feedback on enhanced image (Use Case 5)"""
        if not hasattr(self, 'enhanced_image'):
            messagebox.showwarning("Warning", "No enhanced image to provide feedback on")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Provide Feedback")
        dialog.geometry("400x300")
        dialog.resizable(False, False)
        
        ttk.Label(
            dialog,
            text="Rate the Enhancement",
            style='Title.TLabel'
        ).pack(pady=10)
        
        # Rating
        ttk.Label(
            dialog,
            text="Rating (1-5):",
            style='StarWars.TLabel'
        ).pack()
        
        self.rating_var = tk.IntVar(value=3)
        rating_frame = ttk.Frame(dialog, style='StarWars.TFrame')
        rating_frame.pack()
        
        for i in range(1, 6):
            ttk.Radiobutton(
                rating_frame,
                text=str(i),
                variable=self.rating_var,
                value=i,
                style='StarWars.TLabel'
            ).pack(side='left', padx=5)
        
        # Comments
        ttk.Label(
            dialog,
            text="Comments:",
            style='StarWars.TLabel'
        ).pack(pady=(10, 0))
        
        self.comment_text = tk.Text(
            dialog,
            height=5,
            width=40,
            bg='black',
            fg='white',
            insertbackground='white',
            font=('Orbitron', 10)
        )
        self.comment_text.pack(pady=5)
        
        # Status
        self.feedback_status = ttk.Label(
            dialog,
            text="",
            style='StarWars.TLabel'
        )
        self.feedback_status.pack()
        
        # Buttons
        button_frame = ttk.Frame(dialog, style='StarWars.TFrame')
        button_frame.pack(pady=10)
        
        ttk.Button(
            button_frame,
            text="Submit Feedback",
            command=self.submit_feedback,
            style='StarWars.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
            style='StarWars.TButton'
        ).pack(side='left', padx=5)

    def submit_feedback(self):
        """Submit feedback to the system (Use Case 5)"""
        rating = self.rating_var.get()
        comment = self.comment_text.get("1.0", "end-1c").strip()
    
        if not rating:
            self.feedback_status.config(text="Please provide a rating", style='Error.TLabel')
            return
        
        try:
            headers = {
                "Authorization": f"Bearer {self.jwt_token}",
                "Content-Type": "application/json"
            }
        
            data = {
                "rating": rating,
                "comment": comment
            }
        
            response = requests.post(
                f"{self.api_base_url}/api/feedback",
                headers=headers,
                data=json.dumps(data),
                timeout=5
            )
        
            if response.status_code == 201:
                self.feedback_status.config(text="Feedback submitted successfully!", style='Success.TLabel')
                self.root.after(1500, self.feedback_status.master.destroy)
            else:
                error_msg = response.json().get('error', 'Failed to submit feedback')
                self.feedback_status.config(text=error_msg, style='Error.TLabel')
        except requests.exceptions.ConnectionError:
            self.feedback_status.config(text="Connection failed", style='Error.TLabel')
        except Exception as e:
            self.feedback_status.config(text=f"Error: {str(e)}", style='Error.TLabel')
        
    def save_result(self):
        """Save enhanced image to file (Use Case 1)"""
        if not hasattr(self.enhanced_canvas, 'image'):
            messagebox.showwarning("Warning", "No enhanced image to save")
            return
            
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=filetypes,
            initialfile=f"enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
        if filepath:
            try:
                if self.enhanced_image:
                    self.enhanced_image.save(filepath)
                    self.status_var.set(f"Saved to {os.path.basename(filepath)}")
                else:
                    messagebox.showerror("Error", "No enhanced image data available")
            except Exception as e:
                self.logger.error(f"Error saving image: {str(e)}")
                messagebox.showerror("Error", "Failed to save image")

    def clear_all(self):
        """Clear all images and reset state (Use Case 1)"""
        self.original_canvas.delete("all")
        self.enhanced_canvas.delete("all")
        
        if hasattr(self, 'original_image_data'):
            del self.original_image_data
        if hasattr(self.original_canvas, 'image'):
            del self.original_canvas.image
        if hasattr(self.enhanced_canvas, 'image'):
            del self.enhanced_canvas.image
            
        self.original_image = None
        self.enhanced_image = None
        self.enhance_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        self.feedback_btn.config(state='disabled')
        self.status_var.set("Cleared all images")

    def display_image(self, image, canvas):
        """Display an image on the specified canvas (Use Case 1)"""
        canvas.delete("all")
        
        # Calculate aspect ratio and resize
        img_width, img_height = image.size
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_size = (int(img_width * ratio), int(img_height * ratio))
        
        # Resize and display
        resized_img = image.resize(new_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized_img)
        
        # Keep reference to prevent garbage collection
        canvas.image = photo
        canvas.create_image(
            (canvas_width - new_size[0]) // 2,
            (canvas_height - new_size[1]) // 2,
            anchor='nw',
            image=photo
        )

    def load_recent_images(self):
        """Load recent images from API (Use Case 1)"""
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            response = requests.get(
                f"{self.api_base_url}/api/recent-images",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                self.recent_images = response.json().get("recent_images", [])
                self.display_recent_images()
            else:
                self.logger.warning(f"Failed to load recent images: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Error loading recent images: {str(e)}")

    def display_recent_images(self):
        """Display recent images in the UI (Use Case 1)"""
        # Clear existing images
        for widget in self.recent_images_container.winfo_children():
            widget.destroy()
            
        if not self.recent_images:
            no_images_label = ttk.Label(
                self.recent_images_container,
                text="No recent images found",
                style='StarWars.TLabel'
            )
            no_images_label.pack()
            return
            
        for image in self.recent_images:
            img_frame = ttk.Frame(self.recent_images_container, style='StarWars.TFrame')
            img_frame.pack(side='left', padx=5)
            
            # Create thumbnail (simplified - would normally fetch thumbnail from server)
            thumb_canvas = tk.Canvas(
                img_frame,
                width=100,
                height=100,
                bg='#16213e',
                highlightthickness=0
            )
            thumb_canvas.pack()
            
            # Placeholder for thumbnail - would normally load actual image
            thumb_canvas.create_text(
                50, 50,
                text=f"Image {image['id']}",
                fill="white",
                font=('Orbitron', 8)
            )
            
            thumb_canvas.bind("<Button-1>", lambda e, img=image: self.load_selected_image(img))
            
            date_label = ttk.Label(
                img_frame,
                text=datetime.fromisoformat(image['created_at']).strftime("%m/%d %H:%M"),
                style='StarWars.TLabel'
            )
            date_label.pack()

    def load_selected_image(self, image_data):
        """Load a selected recent image into the viewer (Use Case 1)"""
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            
            # Load original image
            orig_response = requests.get(
                f"{self.api_base_url}{image_data['original_url']}",
                headers=headers
            )
            orig_img = Image.open(io.BytesIO(orig_response.content))
            self.original_image = orig_img
            self.display_image(orig_img, self.original_canvas)
            
            # Load enhanced image
            enh_response = requests.get(
                f"{self.api_base_url}{image_data['enhanced_url']}",
                headers=headers
            )
            enh_img = Image.open(io.BytesIO(enh_response.content))
            self.enhanced_image = enh_img
            self.display_image(enh_img, self.enhanced_canvas)
            
            self.enhance_btn.config(state='disabled')
            self.save_btn.config(state='normal')
            self.status_var.set(f"Loaded image from {image_data['created_at']}")
        except Exception as e:
            self.logger.error(f"Error loading selected image: {str(e)}")
            messagebox.showerror("Error", "Failed to load image")

    def show_admin_panel(self):
        """Display the admin management panel (Use Case 3)"""
        self.clear_window()
        self.root.title("AEGAN - Admin Panel")
        
        # Main container
        main_frame = ttk.Frame(self.root, style='StarWars.TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(main_frame, style='StarWars.TFrame')
        header_frame.pack(fill='x', pady=10)
        
        ttk.Label(
            header_frame,
            text="User Management",
            style='Title.TLabel'
        ).pack(side='left')
        
        ttk.Button(
            header_frame,
            text="Back to Dashboard",
            command=self.show_dashboard,
            style='StarWars.TButton'
        ).pack(side='right')
        
        # User list
        user_list_frame = ttk.Frame(main_frame, style='StarWars.TFrame')
        user_list_frame.pack(fill='both', expand=True)
        
        # Treeview for users
        columns = ('username', 'role', 'last_login')
        self.user_tree = ttk.Treeview(
            user_list_frame,
            columns=columns,
            show='headings',
            selectmode='browse'
        )
        
        # Define headings
        self.user_tree.heading('username', text='Username')
        self.user_tree.heading('role', text='Role')
        self.user_tree.heading('last_login', text='Last Login')
        
        # Set column widths
        self.user_tree.column('username', width=200, anchor='w')
        self.user_tree.column('role', width=150, anchor='w')
        self.user_tree.column('last_login', width=200, anchor='w')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            user_list_frame,
            orient='vertical',
            command=self.user_tree.yview
        )
        self.user_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        self.user_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Control buttons
        control_frame = ttk.Frame(main_frame, style='StarWars.TFrame')
        control_frame.pack(fill='x', pady=10)
        
        ttk.Button(
            control_frame,
            text="Add User",
            command=self.show_add_user_dialog,
            style='StarWars.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            control_frame,
            text="Edit User",
            command=self.show_edit_user_dialog,
            style='StarWars.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            control_frame,
            text="Delete User",
            command=self.delete_user,
            style='StarWars.TButton'
        ).pack(side='left', padx=5)
        
        # Load users
        self.load_users()

    def load_users(self):
        """Load users for admin panel (Use Case 3)"""
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            response = requests.get(
                f"{self.api_base_url}/api/users",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                # Clear existing items
                for item in self.user_tree.get_children():
                    self.user_tree.delete(item)
                
                # Add new items
                for user in response.json().get('users', []):
                    self.user_tree.insert('', 'end', values=(
                        user.get('username'),
                        user.get('role'),
                        user.get('last_login', 'Never')
                    ))
            else:
                messagebox.showerror("Error", "Failed to load users")
        except Exception as e:
            self.logger.error(f"Error loading users: {str(e)}")
            messagebox.showerror("Error", "Connection failed")

    def show_add_user_dialog(self):
        """Show dialog to add a new user (Use Case 3)"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New User")
        dialog.geometry("400x300")
        dialog.resizable(False, False)
        
        ttk.Label(
            dialog,
            text="Add New User",
            style='Title.TLabel'
        ).pack(pady=10)
        
        # Username
        ttk.Label(
            dialog,
            text="Username:",
            style='StarWars.TLabel'
        ).pack()
        
        username_entry = ttk.Entry(
            dialog,
            font=('Orbitron', 10)
        )
        username_entry.pack(pady=5)
        
        # Password
        ttk.Label(
            dialog,
            text="Password:",
            style='StarWars.TLabel'
        ).pack()
        
        password_entry = ttk.Entry(
            dialog,
            show="*",
            font=('Orbitron', 10)
        )
        password_entry.pack(pady=5)
        
        # Role
        ttk.Label(
            dialog,
            text="Role:",
            style='StarWars.TLabel'
        ).pack()
        
        role_var = tk.StringVar(value='user')
        role_menu = ttk.OptionMenu(
            dialog,
            role_var,
            'user',
            'user',
            'data_scientist',
            'devops',
            'admin'
        )
        role_menu.pack(pady=5)
        
        # Status
        status_label = ttk.Label(
            dialog,
            text="",
            style='StarWars.TLabel'
        )
        status_label.pack(pady=5)
        
        # Buttons
        button_frame = ttk.Frame(dialog, style='StarWars.TFrame')
        button_frame.pack(pady=10)
        
        ttk.Button(
            button_frame,
            text="Add User",
            command=lambda: self.add_user(
                username_entry.get(),
                password_entry.get(),
                role_var.get(),
                status_label,
                dialog
            ),
            style='StarWars.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
            style='StarWars.TButton'
        ).pack(side='left', padx=5)

    def add_user(self, username, password, role, status_label, dialog):
        """Add a new user (Use Case 3)"""
        if not username or not password:
            status_label.config(text="Username and password required", style='Error.TLabel')
            return
            
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            response = requests.post(
                f"{self.api_base_url}/api/users",
                json={
                    "username": username,
                    "password": password,
                    "role": role
                },
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 201:
                status_label.config(text="User added successfully!", style='Success.TLabel')
                self.load_users()
                dialog.after(1500, dialog.destroy)
            else:
                error_msg = response.json().get('error', 'Failed to add user')
                status_label.config(text=error_msg, style='Error.TLabel')
        except requests.exceptions.ConnectionError:
            status_label.config(text="Connection failed", style='Error.TLabel')
        except Exception as e:
            status_label.config(text=f"Error: {str(e)}", style='Error.TLabel')

    def show_edit_user_dialog(self):
        """Show dialog to edit a user (Use Case 3)"""
        selected = self.user_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a user first")
            return
            
        user_data = self.user_tree.item(selected)['values']
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit User")
        dialog.geometry("400x300")
        dialog.resizable(False, False)
        
        ttk.Label(
            dialog,
            text="Edit User",
            style='Title.TLabel'
        ).pack(pady=10)
        
        # Username
        ttk.Label(
            dialog,
            text="Username:",
            style='StarWars.TLabel'
        ).pack()
        
        username_entry = ttk.Entry(
            dialog,
            font=('Orbitron', 10)
        )
        username_entry.insert(0, user_data[0])
        username_entry.config(state='readonly')
        username_entry.pack(pady=5)
        
        # Password
        ttk.Label(
            dialog,
            text="New Password (leave blank to keep current):",
            style='StarWars.TLabel'
        ).pack()
        
        password_entry = ttk.Entry(
            dialog,
            show="*",
            font=('Orbitron', 10)
        )
        password_entry.pack(pady=5)
        
        # Role
        ttk.Label(
            dialog,
            text="Role:",
            style='StarWars.TLabel'
        ).pack()
        
        role_var = tk.StringVar(value=user_data[1])
        role_menu = ttk.OptionMenu(
            dialog,
            role_var,
            user_data[1],
            'user',
            'data_scientist',
            'devops',
            'admin'
        )
        role_menu.pack(pady=5)
        
        # Status
        status_label = ttk.Label(
            dialog,
            text="",
            style='StarWars.TLabel'
        )
        status_label.pack(pady=5)
        
        # Buttons
        button_frame = ttk.Frame(dialog, style='StarWars.TFrame')
        button_frame.pack(pady=10)
        
        ttk.Button(
            button_frame,
            text="Update User",
            command=lambda: self.update_user(
                user_data[0],
                password_entry.get(),
                role_var.get(),
                status_label,
                dialog
            ),
            style='StarWars.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
            style='StarWars.TButton'
        ).pack(side='left', padx=5)

    def update_user(self, username, password, role, status_label, dialog):
        """Update user information (Use Case 3)"""
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            data = {"role": role}
            if password:
                data["password"] = password
                
            response = requests.put(
                f"{self.api_base_url}/api/users/{username}",
                json=data,
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                status_label.config(text="User updated successfully!", style='Success.TLabel')
                self.load_users()
                dialog.after(1500, dialog.destroy)
            else:
                error_msg = response.json().get('error', 'Failed to update user')
                status_label.config(text=error_msg, style='Error.TLabel')
        except requests.exceptions.ConnectionError:
            status_label.config(text="Connection failed", style='Error.TLabel')
        except Exception as e:
            status_label.config(text=f"Error: {str(e)}", style='Error.TLabel')

    def delete_user(self):
        """Delete selected user (Use Case 3)"""
        selected = self.user_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a user first")
            return
            
        user_data = self.user_tree.item(selected)['values']
        
        if messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete user {user_data[0]}?"
        ):
            try:
                headers = {"Authorization": f"Bearer {self.jwt_token}"}
                response = requests.delete(
                    f"{self.api_base_url}/api/users/{user_data[0]}",
                    headers=headers,
                    timeout=5
                )
                
                if response.status_code == 200:
                    messagebox.showinfo("Success", "User deleted successfully")
                    self.load_users()
                else:
                    error_msg = response.json().get('error', 'Failed to delete user')
                    messagebox.showerror("Error", error_msg)
            except requests.exceptions.ConnectionError:
                messagebox.showerror("Error", "Connection failed")
            except Exception as e:
                messagebox.showerror("Error", f"Error: {str(e)}")

    def show_training_panel(self):
        """Display the training interface (Use Case 2)"""
        self.clear_window()
        self.root.title("AEGAN - Training Panel")
        
        # Main container
        main_frame = ttk.Frame(self.root, style='StarWars.TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(main_frame, style='StarWars.TFrame')
        header_frame.pack(fill='x', pady=10)
        
        ttk.Label(
            header_frame,
            text="Model Training",
            style='Title.TLabel'
        ).pack(side='left')
        
        ttk.Button(
            header_frame,
            text="Back to Dashboard",
            command=self.show_dashboard,
            style='StarWars.TButton'
        ).pack(side='right')
        
        # Training configuration
        config_frame = ttk.LabelFrame(
            main_frame,
            text="Training Configuration",
            style='StarWars.TFrame'
        )
        config_frame.pack(fill='x', pady=10)
        
        # Dataset path
        ttk.Label(
            config_frame,
            text="Dataset Path:",
            style='StarWars.TLabel'
        ).grid(row=0, column=0, padx=5, pady=5, sticky='e')
        
        self.dataset_path = tk.StringVar()
        ttk.Entry(
            config_frame,
            textvariable=self.dataset_path,
            width=50
        ).grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Button(
            config_frame,
            text="Browse",
            command=self.browse_dataset,
            style='StarWars.TButton'
        ).grid(row=0, column=2, padx=5, pady=5)
        
        # Epochs
        ttk.Label(
            config_frame,
            text="Epochs:",
            style='StarWars.TLabel'
        ).grid(row=1, column=0, padx=5, pady=5, sticky='e')
        
        self.epochs = tk.IntVar(value=50)
        ttk.Entry(
            config_frame,
            textvariable=self.epochs,
            width=10
        ).grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        # Batch size
        ttk.Label(
            config_frame,
            text="Batch Size:",
            style='StarWars.TLabel'
        ).grid(row=2, column=0, padx=5, pady=5, sticky='e')
        
        self.batch_size = tk.IntVar(value=16)
        ttk.Entry(
            config_frame,
            textvariable=self.batch_size,
            width=10
        ).grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        # Learning rate
        ttk.Label(
            config_frame,
            text="Learning Rate:",
            style='StarWars.TLabel'
        ).grid(row=3, column=0, padx=5, pady=5, sticky='e')
        
        self.learning_rate = tk.StringVar(value="0.0002")
        ttk.Entry(
            config_frame,
            textvariable=self.learning_rate,
            width=10
        ).grid(row=3, column=1, padx=5, pady=5, sticky='w')
        
        # Training controls
        control_frame = ttk.Frame(main_frame, style='StarWars.TFrame')
        control_frame.pack(fill='x', pady=10)
        
        self.start_train_btn = ttk.Button(
            control_frame,
            text="Start Training",
            command=self.start_training,
            style='StarWars.TButton'
        )
        self.start_train_btn.pack(side='left', padx=5)
        
        ttk.Button(
            control_frame,
            text="Stop Training",
            command=self.stop_training,
            style='StarWars.TButton'
        ).pack(side='left', padx=5)
        
        # Training output
        output_frame = ttk.LabelFrame(
            main_frame,
            text="Training Output",
            style='StarWars.TFrame'
        )
        output_frame.pack(fill='both', expand=True, pady=10)
        
        self.training_output = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            width=80,
            height=20,
            bg='black',
            fg='white',
            insertbackground='white',
            font=('Courier', 10)
        )
        self.training_output.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill='x', pady=10)
        
        # Status
        self.train_status = ttk.Label(
            main_frame,
            text="Ready",
            style='StarWars.TLabel'
        )
        self.train_status.pack()

    def browse_dataset(self):
        """Browse for dataset directory (Use Case 2)"""
        dirpath = filedialog.askdirectory(title="Select Dataset Directory")
        if dirpath:
            self.dataset_path.set(dirpath)

    def start_training(self):
        """Start model training (Use Case 2)"""
        if not self.dataset_path.get():
            messagebox.showwarning("Warning", "Please select a dataset directory")
            return
            
        # Disable start button during training
        self.start_train_btn.config(state='disabled')
        self.train_status.config(text="Training started...", style='StarWars.TLabel')
        
        # Simulate training (in a real app, this would call the training API)
        self.training_output.insert('end', "Starting training session...\n")
        self.training_output.insert('end', f"Dataset: {self.dataset_path.get()}\n")
        self.training_output.insert('end', f"Epochs: {self.epochs.get()}\n")
        self.training_output.insert('end', f"Batch size: {self.batch_size.get()}\n")
        self.training_output.insert('end', f"Learning rate: {self.learning_rate.get()}\n\n")
        
        # Start simulated training progress
        self.simulate_training_progress()

    def simulate_training_progress(self):
        """Simulate training progress (for demo purposes) (Use Case 2)"""
        if not hasattr(self, 'training_in_progress'):
            self.training_in_progress = True
            self.current_epoch = 0
            self.progress_var.set(0)
            
        if self.current_epoch < self.epochs.get():
            self.current_epoch += 1
            progress = (self.current_epoch / self.epochs.get()) * 100
            self.progress_var.set(progress)
            
            # Add fake training output
            self.training_output.insert('end', 
                f"Epoch {self.current_epoch}/{self.epochs.get()} - " +
                f"Loss: {0.5 + (0.3 * (1 - self.current_epoch/self.epochs.get())):.4f} - " +
                f"Accuracy: {(0.2 + (0.7 * self.current_epoch/self.epochs.get())):.4f}\n"
            )
            self.training_output.see('end')
            
            # Schedule next update
            self.root.after(500, self.simulate_training_progress)
        else:
            self.training_output.insert('end', "\nTraining completed!\n")
            self.training_output.see('end')
            self.train_status.config(text="Training completed", style='Success.TLabel')
            self.start_train_btn.config(state='normal')
            del self.training_in_progress

    def stop_training(self):
        """Stop model training (Use Case 2)"""
        if hasattr(self, 'training_in_progress'):
            del self.training_in_progress
            self.training_output.insert('end', "\nTraining stopped by user\n")
            self.training_output.see('end')
            self.train_status.config(text="Training stopped", style='Error.TLabel')
            self.start_train_btn.config(state='normal')

    def show_feedback_panel(self):
        """Display feedback analysis panel (Use Case 5)"""
        self.clear_window()
        self.root.title("AEGAN - Feedback Analysis")
        
        # Main container
        main_frame = ttk.Frame(self.root, style='StarWars.TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(main_frame, style='StarWars.TFrame')
        header_frame.pack(fill='x', pady=10)
        
        ttk.Label(
            header_frame,
            text="Feedback Analysis",
            style='Title.TLabel'
        ).pack(side='left')
        
        ttk.Button(
            header_frame,
            text="Back to Dashboard",
            command=self.show_dashboard,
            style='StarWars.TButton'
        ).pack(side='right')
        
        # Feedback statistics
        stats_frame = ttk.LabelFrame(
            main_frame,
            text="Feedback Statistics",
            style='StarWars.TFrame'
        )
        stats_frame.pack(fill='x', pady=10)
        
        # Add charts/visualizations here (would use matplotlib or custom widgets)
        ttk.Label(
            stats_frame,
            text="Feedback ratings distribution and trends would be displayed here",
            style='StarWars.TLabel'
        ).pack(pady=20)
        
        # Feedback list
        list_frame = ttk.LabelFrame(
            main_frame,
            text="Recent Feedback",
            style='StarWars.TFrame'
        )
        list_frame.pack(fill='both', expand=True, pady=10)
        
        # Treeview for feedback
        columns = ('user', 'rating', 'comment', 'date')
        self.feedback_tree = ttk.Treeview(
            list_frame,
            columns=columns,
            show='headings',
            selectmode='browse'
        )
        
        # Define headings
        self.feedback_tree.heading('user', text='User')
        self.feedback_tree.heading('rating', text='Rating')
        self.feedback_tree.heading('comment', text='Comment')
        self.feedback_tree.heading('date', text='Date')
        
        # Set column widths
        self.feedback_tree.column('user', width=150, anchor='w')
        self.feedback_tree.column('rating', width=80, anchor='center')
        self.feedback_tree.column('comment', width=300, anchor='w')
        self.feedback_tree.column('date', width=150, anchor='w')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            list_frame,
            orient='vertical',
            command=self.feedback_tree.yview
        )
        self.feedback_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        self.feedback_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Load feedback
        self.load_feedback()

    def load_feedback(self):
        """Load feedback data for analysis (Use Case 5)"""
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            response = requests.get(
                f"{self.api_base_url}/api/feedback",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                # Clear existing items
                for item in self.feedback_tree.get_children():
                    self.feedback_tree.delete(item)
                
                # Add new items
                for feedback in response.json().get('feedback', []):
                    self.feedback_tree.insert('', 'end', values=(
                        feedback.get('username'),
                        feedback.get('rating'),
                        feedback.get('comment', 'No comment'),
                        feedback.get('created_at', 'Unknown')
                    ))
            else:
                messagebox.showerror("Error", "Failed to load feedback data")
        except Exception as e:
            messagebox.showerror("Error", f"Connection failed: {str(e)}")

    def show_deployment_panel(self):
        """Display the deployment interface (Use Case 4)"""
        self.clear_window()
        self.root.title("AEGAN - Deployment Panel")
        
        # Main container
        main_frame = ttk.Frame(self.root, style='StarWars.TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(main_frame, style='StarWars.TFrame')
        header_frame.pack(fill='x', pady=10)
        
        ttk.Label(
            header_frame,
            text="Model Deployment",
            style='Title.TLabel'
        ).pack(side='left')
        
        ttk.Button(
            header_frame,
            text="Back to Dashboard",
            command=self.show_dashboard,
            style='StarWars.TButton'
        ).pack(side='right')
        
        # Deployment controls
        control_frame = ttk.Frame(main_frame, style='StarWars.TFrame')
        control_frame.pack(fill='x', pady=10)
        
        ttk.Button(
            control_frame,
            text="Deploy Latest Model",
            command=self.deploy_model,
            style='StarWars.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            control_frame,
            text="Rollback to Previous",
            command=self.rollback_model,
            style='StarWars.TButton'
        ).pack(side='left', padx=5)
        
        # Model versions
        versions_frame = ttk.LabelFrame(
            main_frame,
            text="Model Versions",
            style='StarWars.TFrame'
        )
        versions_frame.pack(fill='both', expand=True, pady=10)
        
        # Treeview for versions
        columns = ('version', 'date', 'status')
        self.version_tree = ttk.Treeview(
            versions_frame,
            columns=columns,
            show='headings',
            selectmode='browse'
        )
        
        # Define headings
        self.version_tree.heading('version', text='Version')
        self.version_tree.heading('date', text='Deployment Date')
        self.version_tree.heading('status', text='Status')
        
        # Set column widths
        self.version_tree.column('version', width=200, anchor='w')
        self.version_tree.column('date', width=200, anchor='w')
        self.version_tree.column('status', width=150, anchor='w')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            versions_frame,
            orient='vertical',
            command=self.version_tree.yview
        )
        self.version_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        self.version_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Performance metrics
        metrics_frame = ttk.LabelFrame(
            main_frame,
            text="Performance Metrics",
            style='StarWars.TFrame'
        )
        metrics_frame.pack(fill='x', pady=10)
        
        # Add metrics visualization (placeholder)
        ttk.Label(
            metrics_frame,
            text="Performance metrics visualization would go here",
            style='StarWars.TLabel'
        ).pack(pady=20)
        
        # Load versions
        self.load_model_versions()

    def load_model_versions(self):
        """Load model versions for deployment panel (Use Case 4)"""
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            response = requests.get(
                f"{self.api_base_url}/api/models",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                # Clear existing items
                for item in self.version_tree.get_children():
                    self.version_tree.delete(item)
                
                # Add new items
                for model in response.json().get('models', []):
                    self.version_tree.insert('', 'end', values=(
                        model.get('version'),
                        model.get('date'),
                        model.get('status', 'unknown')
                    ))
            else:
                messagebox.showerror("Error", "Failed to load model versions")
        except Exception as e:
            self.logger.error(f"Error loading model versions: {str(e)}")
            messagebox.showerror("Error", "Connection failed")

    def deploy_model(self):
        """Deploy the latest model version (Use Case 4)"""
        if messagebox.askyesno(
            "Confirm Deployment",
            "Are you sure you want to deploy the latest model version?"
        ):
            try:
                headers = {"Authorization": f"Bearer {self.jwt_token}"}
                response = requests.post(
                    f"{self.api_base_url}/api/deploy",
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    messagebox.showinfo("Success", "Model deployed successfully")
                    self.load_model_versions()
                else:
                    error_msg = response.json().get('error', 'Failed to deploy model')
                    messagebox.showerror("Error", error_msg)
            except requests.exceptions.ConnectionError:
                messagebox.showerror("Error", "Connection failed")
            except Exception as e:
                messagebox.showerror("Error", f"Error: {str(e)}")

    def rollback_model(self):
        """Rollback to previous model version (Use Case 4)"""
        if messagebox.askyesno(
            "Confirm Rollback",
            "Are you sure you want to rollback to the previous model version?"
        ):
            try:
                headers = {"Authorization": f"Bearer {self.jwt_token}"}
                response = requests.post(
                    f"{self.api_base_url}/api/rollback",
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    messagebox.showinfo("Success", "Model rolled back successfully")
                    self.load_model_versions()
                else:
                    error_msg = response.json().get('error', 'Failed to rollback model')
                    messagebox.showerror("Error", error_msg)
            except requests.exceptions.ConnectionError:
                messagebox.showerror("Error", "Connection failed")
            except Exception as e:
                messagebox.showerror("Error", f"Error: {str(e)}")

    def show_monitoring_panel(self):
        """Display system monitoring panel (Use Case 6)"""
        self.clear_window()
        self.root.title("AEGAN - System Monitoring")
        
        # Main container
        main_frame = ttk.Frame(self.root, style='StarWars.TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(main_frame, style='StarWars.TFrame')
        header_frame.pack(fill='x', pady=10)
        
        ttk.Label(
            header_frame,
            text="System Monitoring",
            style='Title.TLabel'
        ).pack(side='left')
        
        ttk.Button(
            header_frame,
            text="Back to Dashboard",
            command=self.show_dashboard,
            style='StarWars.TButton'
        ).pack(side='right')
        
        # System metrics
        metrics_frame = ttk.LabelFrame(
            main_frame,
            text="System Metrics",
            style='StarWars.TFrame'
        )
        metrics_frame.pack(fill='both', expand=True, pady=10)
        
        # CPU Usage
        ttk.Label(
            metrics_frame,
            text="CPU Usage:",
            style='StarWars.TLabel'
        ).pack(anchor='w', padx=10, pady=(10, 0))
        
        self.cpu_usage = ttk.Progressbar(
            metrics_frame,
            orient='horizontal',
            length=200,
            mode='determinate'
        )
        self.cpu_usage.pack(anchor='w', padx=10, pady=(0, 10))
        
        # Memory Usage
        ttk.Label(
            metrics_frame,
            text="Memory Usage:",
            style='StarWars.TLabel'
        ).pack(anchor='w', padx=10, pady=(10, 0))
        
        self.memory_usage = ttk.Progressbar(
            metrics_frame,
            orient='horizontal',
            length=200,
            mode='determinate'
        )
        self.memory_usage.pack(anchor='w', padx=10, pady=(0, 10))
        
        # Disk Usage
        ttk.Label(
            metrics_frame,
            text="Disk Usage:",
            style='StarWars.TLabel'
        ).pack(anchor='w', padx=10, pady=(10, 0))
        
        self.disk_usage = ttk.Progressbar(
            metrics_frame,
            orient='horizontal',
            length=200,
            mode='determinate'
        )
        self.disk_usage.pack(anchor='w', padx=10, pady=(0, 10))
        
        # Recent logs
        logs_frame = ttk.LabelFrame(
            main_frame,
            text="Recent System Logs",
            style='StarWars.TFrame'
        )
        logs_frame.pack(fill='both', expand=True, pady=10)
        
        self.logs_text = scrolledtext.ScrolledText(
            logs_frame,
            wrap=tk.WORD,
            width=80,
            height=10,
            bg='black',
            fg='white',
            insertbackground='white',
            font=('Courier', 10)
        )
        self.logs_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Refresh button
        ttk.Button(
            main_frame,
            text="Refresh Metrics",
            command=self.refresh_monitoring,
            style='StarWars.TButton'
        ).pack(pady=10)
        
        # Initial load
        self.refresh_monitoring()

    def refresh_monitoring(self):
        """Refresh monitoring data (Use Case 6)"""
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            response = requests.get(
                f"{self.api_base_url}/api/monitoring",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                metrics = response.json()
                
                # Update progress bars
                self.cpu_usage['value'] = metrics.get('cpu_usage', 0)
                self.memory_usage['value'] = metrics.get('memory_usage', 0)
                self.disk_usage['value'] = metrics.get('disk_usage', 0)
                
                # Update logs
                self.logs_text.delete(1.0, tk.END)
                for log in metrics.get('logs', []):
                    self.logs_text.insert(tk.END, f"{log}\n")
            else:
                messagebox.showerror("Error", "Failed to load monitoring data")
        except Exception as e:
            messagebox.showerror("Error", f"Connection failed: {str(e)}")

    def clear_window(self):
        """Clear all widgets from the window"""
        for widget in self.root.winfo_children():
            widget.destroy()

    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = AEGANApplication()
    app.run()