import os
import glob
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Налаштування глобального стилю
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

class SmartLabDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("SmartLab Analytics - Pro Dashboard")
        self.geometry("1200x800")
        self.minimum_size = (1000, 700)
        
        # Колірна палітра
        self.colors = {
            "bg": "#f8f9fa",         
            "sidebar": "#1e293b",    
            "card": "#ffffff",       
            "text_dark": "#0f172a",  
            "text_light": "#f1f5f9", 
            "accent": "#0ea5e9",     
            "border": "#e2e8f0"      
        }
        
        self.configure(bg=self.colors["bg"])
        self.base_data_path = os.path.join("data", "Projects")
        
        self.selected_project = tk.StringVar()
        self.selected_sensor = tk.StringVar()
        self.selected_file = tk.StringVar()
        
        self.kpi_avg = tk.StringVar(value="--")
        self.kpi_max = tk.StringVar(value="--")
        self.kpi_min = tk.StringVar(value="--")
        
        self.setup_styles()
        self.setup_ui()
        self.load_projects()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TCombobox", 
                             fieldbackground="white", 
                             background=self.colors["border"],
                             foreground=self.colors["text_dark"],
                             padding=5)

    def setup_ui(self):
        main_container = tk.Frame(self, bg=self.colors["bg"])
        main_container.pack(fill="both", expand=True)
        
        # --- SIDEBAR ---
        sidebar = tk.Frame(main_container, bg=self.colors["sidebar"], width=280)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        
        logo_label = tk.Label(sidebar, text="📊 SmartLab", font=("Segoe UI", 18, "bold"), 
                              bg=self.colors["sidebar"], fg=self.colors["accent"])
        logo_label.pack(pady=(25, 5), padx=20, anchor="w")
        
        subtitle_label = tk.Label(sidebar, text="Analytics Platform v1.1", font=("Segoe UI", 9), 
                                 bg=self.colors["sidebar"], fg="#64748b")
        subtitle_label.pack(pady=(0, 30), padx=20, anchor="w")
        
        def create_sidebar_label(text):
            lbl = tk.Label(sidebar, text=text, font=("Segoe UI", 10, "bold"), 
                           bg=self.colors["sidebar"], fg=self.colors["text_light"])
            lbl.pack(anchor="w", padx=20, pady=(15, 5))
            return lbl

        create_sidebar_label("📁 Project:")
        self.project_cb = ttk.Combobox(sidebar, textvariable=self.selected_project, state="readonly")
        self.project_cb.pack(fill="x", padx=20, pady=5)
        self.project_cb.bind("<<ComboboxSelected>>", self.on_project_select)
        
        create_sidebar_label("📟 Sensor:")
        self.sensor_cb = ttk.Combobox(sidebar, textvariable=self.selected_sensor, state="readonly")
        self.sensor_cb.pack(fill="x", padx=20, pady=5)
        self.sensor_cb.bind("<<ComboboxSelected>>", self.on_sensor_select)
        
        create_sidebar_label("📄 Data File:")
        self.file_cb = ttk.Combobox(sidebar, textvariable=self.selected_file, state="readonly")
        self.file_cb.pack(fill="x", padx=20, pady=5)
        
        self.plot_btn = tk.Button(
            sidebar, text="Generate Analytics", command=self.plot_data, 
            bg=self.colors["accent"], fg="white", font=("Segoe UI", 11, "bold"),
            activebackground="#0284c7", activeforeground="white",
            relief="flat", bd=0, cursor="hand2"
        )
        self.plot_btn.pack(fill="x", padx=20, pady=(35, 10))

        # --- MAIN CONTENT AREA ---
        content_area = tk.Frame(main_container, bg=self.colors["bg"])
        content_area.pack(side="right", fill="both", expand=True, padx=25, pady=20)
        
        self.header_lbl = tk.Label(content_area, text="Dashboard Overview", font=("Segoe UI", 18, "bold"), 
                                   bg=self.colors["bg"], fg=self.colors["text_dark"])
        self.header_lbl.pack(anchor="w", pady=(0, 15))
        
        # KPI Cards
        kpi_frame = tk.Frame(content_area, bg=self.colors["bg"])
        kpi_frame.pack(fill="x", pady=(0, 20))
        
        def create_kpi_card(parent, title, variable, accent_color):
            card = tk.Frame(parent, bg=self.colors["card"], highlightbackground=self.colors["border"], highlightthickness=1)
            card.pack(side="left", fill="both", expand=True, padx=(0, 15))
            
            stripe = tk.Frame(card, bg=accent_color, width=4)
            stripe.pack(side="left", fill="y")
            
            inner = tk.Frame(card, bg=self.colors["card"], padx=15, pady=10)
            inner.pack(side="left", fill="both", expand=True)
            
            lbl_title = tk.Label(inner, text=title, font=("Segoe UI", 9, "bold"), bg=self.colors["card"], fg="#64748b")
            lbl_title.pack(anchor="w")
            
            lbl_val = tk.Label(inner, textvariable=variable, font=("Segoe UI", 16, "bold"), bg=self.colors["card"], fg=self.colors["text_dark"])
            lbl_val.pack(anchor="w", pady=(4, 0))
            return card

        create_kpi_card(kpi_frame, "AVERAGE VALUE", self.kpi_avg, self.colors["accent"])
        create_kpi_card(kpi_frame, "MAXIMUM RECORDED", self.kpi_max, "#ef4444")
        create_kpi_card(kpi_frame, "MINIMUM RECORDED", self.kpi_min, "#22c55e")
        
        # Plot Frame Card (Контейнер-обгортка для графіка)
        self.plot_card = tk.Frame(content_area, bg=self.colors["card"], highlightbackground=self.colors["border"], highlightthickness=1)
        self.plot_card.pack(fill="both", expand=True)
        
        # Текстовий плейсхолдер при запуску програми
        self.placeholder_label = tk.Label(
            self.plot_card, 
            text="Please configure your analytics targets on the left sidebar\nand click 'Generate Analytics' to map out the sensor data.", 
            font=("Segoe UI", 11), bg=self.colors["card"], fg="#64748b", justify="center"
        )
        self.placeholder_label.pack(expand=True, fill="both")

    def load_projects(self):
        if not os.path.exists(self.base_data_path):
            messagebox.showwarning("Warning", f"Data path '{self.base_data_path}' not found.\nPlease run mock data generation first.")
            return
            
        projects = [d for d in os.listdir(self.base_data_path) if os.path.isdir(os.path.join(self.base_data_path, d))]
        self.project_cb['values'] = projects
        if projects:
            self.project_cb.current(0)
            self.on_project_select(None)

    def on_project_select(self, event):
        project = self.selected_project.get()
        sensors_path = os.path.join(self.base_data_path, project, "Sensors")
        
        if os.path.exists(sensors_path):
            sensors = [d for d in os.listdir(sensors_path) if os.path.isdir(os.path.join(sensors_path, d))]
            self.sensor_cb['values'] = sensors
            if sensors:
                self.sensor_cb.current(0)
                self.on_sensor_select(None)
            else:
                self.sensor_cb['values'] = []
                self.file_cb['values'] = []
        else:
            self.sensor_cb['values'] = []
            self.file_cb['values'] = []

    def on_sensor_select(self, event):
        project = self.selected_project.get()
        sensor = self.selected_sensor.get()
        csv_path = os.path.join(self.base_data_path, project, "Sensors", sensor)
        
        if os.path.exists(csv_path):
            csv_files = [os.path.basename(f) for f in glob.glob(os.path.join(csv_path, "*.csv"))]
            self.file_cb['values'] = csv_files
            if csv_files:
                self.file_cb.current(0)
            else:
                self.file_cb['values'] = []

    def plot_data(self):
        project = self.selected_project.get()
        sensor = self.selected_sensor.get()
        filename = self.selected_file.get()
        
        if not project or not sensor or not filename:
            messagebox.showerror("Error", "Please select Project, Sensor, and Data File.")
            return
            
        full_path = os.path.join(self.base_data_path, project, "Sensors", sensor, filename)
        
        try:
            # Clear previous plots from the frame
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
                
            # Load data (assuming CSV has columns like 'timestamp' and 'value' or index-based)
            df = pd.read_csv(full_path)
            
            if df.empty:
                raise ValueError("The selected CSV file is empty.")
                
            # Determine columns to plot
            # If standard columns exist, use them; otherwise plot the first two columns
            if 'timestamp' in df.columns and 'value' in df.columns:
                x_data = df['timestamp']
                y_data = df['value']
            else:
                x_data = df.index
                y_data = df.iloc[:, 0] # first column as values
            
            # Create Matplotlib Figure
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            ax.plot(x_data, y_data, marker='o', linestyle='-', color='#0078d4', label=sensor)
            ax.set_title(f"{project} - {sensor} Data Analysis", fontsize=12, fontweight='bold')
            ax.set_xlabel("Time / Step")
            ax.set_ylabel("Sensor Reading")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            
            plt.xticks(rotation=15)
            fig.tight_layout()
            
            # Embed Figure into Tkinter Canvas
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill="both", expand=True)
            
            # Add Matplotlib Navigation Toolbar (Zoom, Pan, Save)
            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()
            canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Data Error", f"Failed to read or plot data:\n{str(e)}")
            # Restore placeholder if failed
            self.placeholder_label = tk.Label(self.plot_frame, text="Error loading data.", font=("Arial", 12), bg="white", fg="red")
            self.placeholder_label.pack(expand=True)

if __name__ == "__main__":
    app = SmartLabDashboard()
    app.mainloop()
