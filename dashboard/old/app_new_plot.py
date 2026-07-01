import os
import glob
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class SmartLabDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("SmartLab Analytics - MVP Dashboard")
        self.geometry("1000x700")
        self.configure(bg="#f0f0f0")
        
        # Base path for data
        self.base_data_path = os.path.join("data", "Projects")
        
        # UI Elements Variables
        self.selected_project = tk.StringVar()
        self.selected_sensor = tk.StringVar()
        self.selected_file = tk.StringVar()
        
        self.setup_ui()
        self.load_projects()

    def setup_ui(self):
        # --- Top Control Panel ---
        control_frame = tk.LabelFrame(self, text=" Experiment Settings ", font=("Arial", 11, "bold"), bg="#f0f0f0", fg="#333333")
        control_frame.pack(fill="x", padx=15, pady=10)
        
        # Project Selection
        tk.Label(control_frame, text="Select Project:", bg="#f0f0f0").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.project_cb = ttk.Combobox(control_frame, textvariable=self.selected_project, width=30, state="readonly")
        self.project_cb.grid(row=0, column=1, padx=10, pady=10)
        self.project_cb.bind("<<ComboboxSelected>>", self.on_project_select)
        
        # Sensor Selection
        tk.Label(control_frame, text="Select Sensor:", bg="#f0f0f0").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        self.sensor_cb = ttk.Combobox(control_frame, textvariable=self.selected_sensor, width=25, state="readonly")
        self.sensor_cb.grid(row=0, column=3, padx=10, pady=10)
        self.sensor_cb.bind("<<ComboboxSelected>>", self.on_sensor_select)
        
        # File/Date Selection
        tk.Label(control_frame, text="Select Data File:", bg="#f0f0f0").grid(row=0, column=4, padx=10, pady=10, sticky="w")
        self.file_cb = ttk.Combobox(control_frame, textvariable=self.selected_file, width=20, state="readonly")
        self.file_cb.grid(row=0, column=5, padx=10, pady=10)
        
        # Plot Button
        self.plot_btn = tk.Button(control_frame, text="Plot Data", command=self.plot_data, bg="#0078d4", fg="white", font=("Arial", 10, "bold"), padx=15)
        self.plot_btn.grid(row=0, column=6, padx=20, pady=10)
        
        # --- Bottom Plot Panel ---
        self.plot_frame = tk.Frame(self, bg="white", bd=1, relief="sunken")
        self.plot_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Placeholder Label inside Plot Frame
        self.placeholder_label = tk.Label(self.plot_frame, text="Select parameters above and click 'Plot Data'", font=("Arial", 12), bg="white", fg="#666666")
        self.placeholder_label.pack(expand=True)

    def load_projects(self):
        """Scans data/Projects directory for subfolders"""
        if not os.path.exists(self.base_data_path):
            messagebox.showwarning("Warning", f"Data path '{self.base_data_path}' not found.\nPlease run mock data generation first.")
            return
            
        projects = [d for d in os.listdir(self.base_data_path) if os.path.isdir(os.path.join(self.base_data_path, d))]
        self.project_cb['values'] = projects
        if projects:
            self.project_cb.current(0)
            self.on_project_select(None)

    def on_project_select(self, event):
        """Triggered when user selects a project. Finds available sensors."""
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
        """Triggered when user selects a sensor. Finds available CSV data files."""
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
        """Reads the selected CSV and renders a Matplotlib graph inside the Tkinter frame."""
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