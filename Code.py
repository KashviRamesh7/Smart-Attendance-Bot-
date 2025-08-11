
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import os
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime, timedelta
import face_recognition
from PIL import Image, ImageTk
import threading

class KashviSmartFaceAttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Face Recognition Attendance System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')

        self.faces_dir = "registered_faces"
        self.attendance_dir = "attendance_photos"
        self.report_dir = "reports"
        self.attendance_file = "attendance.csv"
        self.face_data_file = "face_encodings.pkl"
        self.config_file = "config.json"

        os.makedirs(self.faces_dir, exist_ok=True)
        os.makedirs(self.attendance_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

        self.load_config()
        self.load_faces()

        if not os.path.exists(self.attendance_file):
            pd.DataFrame(columns=['Name', 'Student_ID', 'Date', 'Time', 'Status', 'Photo_Path']).to_csv(self.attendance_file, index=False)

        self.cap = None
        self.is_camera_running = False
        self.camera_thread = None

        self.setup_gui()

    def load_config(self):
        default = {
            "work_start": "09:00",
            "work_end": "17:00",
            "late_threshold": 15,
            "recognition_tolerance": 0.6,
            "location": "Main Office"
        }
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except json.JSONDecodeError:
                self.config = default
        else:
            self.config = default

        updated = False
        for key, value in default.items():
            if key not in self.config:
                self.config[key] = value
                updated = True

        if updated or not os.path.exists(self.config_file):
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)

    # The rest of your code remains unchanged...

    
    def load_faces(self):
        if os.path.exists(self.face_data_file):
            with open(self.face_data_file, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data['encodings']
                self.known_names = data['names']
                self.known_ids = data['ids']
        else:
            self.known_encodings = []
            self.known_names = []
            self.known_ids = []
    
    def save_faces(self):
        data = {
            "encodings": self.known_encodings,
            "names": self.known_names,
            "ids": self.known_ids
        }
        with open(self.face_data_file, 'wb') as f:
            pickle.dump(data, f)
    
    def setup_gui(self):
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="Smart Face Recognition Attendance System", 
                              font=('Arial', 18, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create tabs
        self.create_registration_tab()
        self.create_attendance_tab()
        self.create_records_tab()
        self.create_settings_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, bg='#ecf0f1')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_registration_tab(self):
        reg_frame = ttk.Frame(self.notebook)
        self.notebook.add(reg_frame, text="Registration")
        
        # Registration form
        form_frame = tk.Frame(reg_frame, bg='white', relief=tk.RAISED, bd=2)
        form_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(form_frame, text="Register New Face", font=('Arial', 14, 'bold'), bg='white').pack(pady=10)
        
        # Name entry
        name_frame = tk.Frame(form_frame, bg='white')
        name_frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(name_frame, text="Name:", font=('Arial', 10), bg='white').pack(side=tk.LEFT)
        self.name_entry = tk.Entry(name_frame, font=('Arial', 10), width=30)
        self.name_entry.pack(side=tk.RIGHT, padx=10)
        
        # Student ID entry
        id_frame = tk.Frame(form_frame, bg='white')
        id_frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(id_frame, text="Student ID:", font=('Arial', 10), bg='white').pack(side=tk.LEFT)
        self.id_entry = tk.Entry(id_frame, font=('Arial', 10), width=30)
        self.id_entry.pack(side=tk.RIGHT, padx=10)
        
        # Buttons
        button_frame = tk.Frame(form_frame, bg='white')
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Register from Camera", command=self.register_from_camera,
                 bg='#3498db', fg='white', font=('Arial', 10, 'bold'), width=20).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Register from Image", command=self.register_from_image,
                 bg='#2ecc71', fg='white', font=('Arial', 10, 'bold'), width=20).pack(side=tk.LEFT, padx=5)
        
        # Camera preview frame
        self.camera_frame = tk.Frame(reg_frame, bg='black', relief=tk.SUNKEN, bd=2)
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.camera_label = tk.Label(self.camera_frame, text="Camera Preview", bg='black', fg='white', 
                                    font=('Arial', 12))
        self.camera_label.pack(expand=True)
        
        # Registered faces list
        list_frame = tk.Frame(reg_frame, bg='white', relief=tk.RAISED, bd=2)
        list_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(list_frame, text="Registered Faces", font=('Arial', 12, 'bold'), bg='white').pack(pady=5)
        
        # Listbox with scrollbar
        list_container = tk.Frame(list_frame, bg='white')
        list_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.faces_listbox = tk.Listbox(list_container, height=6, font=('Arial', 10))
        scrollbar = tk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.faces_listbox.yview)
        self.faces_listbox.config(yscrollcommand=scrollbar.set)
        
        self.faces_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Delete button
        tk.Button(list_frame, text="Delete Selected", command=self.delete_selected_face,
                 bg='#e74c3c', fg='white', font=('Arial', 10, 'bold')).pack(pady=5)
        
        self.update_faces_list()
    
    def create_attendance_tab(self):
        att_frame = ttk.Frame(self.notebook)
        self.notebook.add(att_frame, text="Attendance")
        
        # Control buttons
        control_frame = tk.Frame(att_frame, bg='white', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(control_frame, text="Attendance Control", font=('Arial', 14, 'bold'), bg='white').pack(pady=10)
        
        button_frame = tk.Frame(control_frame, bg='white')
        button_frame.pack(pady=10)
        
        self.start_button = tk.Button(button_frame, text="Start Attendance", command=self.start_attendance,
                                     bg='#27ae60', fg='white', font=('Arial', 12, 'bold'), width=15)
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        self.stop_button = tk.Button(button_frame, text="Stop Attendance", command=self.stop_attendance,
                                    bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'), width=15, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # Attendance camera frame
        self.attendance_camera_frame = tk.Frame(att_frame, bg='black', relief=tk.SUNKEN, bd=2)
        self.attendance_camera_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.attendance_camera_label = tk.Label(self.attendance_camera_frame, text="Attendance Camera View", 
                                               bg='black', fg='white', font=('Arial', 12))
        self.attendance_camera_label.pack(expand=True)
        
        # Today's attendance summary
        summary_frame = tk.Frame(att_frame, bg='white', relief=tk.RAISED, bd=2)
        summary_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(summary_frame, text="Today's Attendance Summary", font=('Arial', 12, 'bold'), bg='white').pack(pady=5)
        
        self.summary_text = tk.Text(summary_frame, height=4, font=('Arial', 9))
        self.summary_text.pack(fill=tk.X, padx=10, pady=5)
        
        self.update_summary()
    
    def create_records_tab(self):
        records_frame = ttk.Frame(self.notebook)
        self.notebook.add(records_frame, text="Records")
        
        # Control buttons
        control_frame = tk.Frame(records_frame, bg='white', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(control_frame, text="Attendance Records", font=('Arial', 14, 'bold'), bg='white').pack(pady=5)
        
        button_frame = tk.Frame(control_frame, bg='white')
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Refresh Records", command=self.refresh_records,
                 bg='#3498db', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Export Report", command=self.export_report,
                 bg='#f39c12', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Records table
        table_frame = tk.Frame(records_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Treeview for records
        columns = ('Name', 'Student_ID', 'Date', 'Time', 'Status')
        self.records_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Define column headings
        for col in columns:
            self.records_tree.heading(col, text=col)
            self.records_tree.column(col, width=150)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.records_tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.records_tree.xview)
        
        self.records_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.records_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.refresh_records()
    
    def create_settings_tab(self):
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Settings form
        form_frame = tk.Frame(settings_frame, bg='white', relief=tk.RAISED, bd=2)
        form_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(form_frame, text="System Settings", font=('Arial', 14, 'bold'), bg='white').pack(pady=10)
        
        # Work start time
        start_frame = tk.Frame(form_frame, bg='white')
        start_frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(start_frame, text="Work Start Time:", font=('Arial', 10), bg='white').pack(side=tk.LEFT)
        self.start_time_var = tk.StringVar(value=self.config['work_start'])
        tk.Entry(start_frame, textvariable=self.start_time_var, font=('Arial', 10), width=10).pack(side=tk.RIGHT)
        
        # Work end time
        end_frame = tk.Frame(form_frame, bg='white')
        end_frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(end_frame, text="Work End Time:", font=('Arial', 10), bg='white').pack(side=tk.LEFT)
        self.end_time_var = tk.StringVar(value=self.config['work_end'])
        tk.Entry(end_frame, textvariable=self.end_time_var, font=('Arial', 10), width=10).pack(side=tk.RIGHT)
        
        # Late threshold
        late_frame = tk.Frame(form_frame, bg='white')
        late_frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(late_frame, text="Late Threshold (minutes):", font=('Arial', 10), bg='white').pack(side=tk.LEFT)
        self.late_threshold_var = tk.StringVar(value=str(self.config['late_threshold']))
        tk.Entry(late_frame, textvariable=self.late_threshold_var, font=('Arial', 10), width=10).pack(side=tk.RIGHT)
        
        # Recognition tolerance
        tolerance_frame = tk.Frame(form_frame, bg='white')
        tolerance_frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(tolerance_frame, text="Recognition Tolerance:", font=('Arial', 10), bg='white').pack(side=tk.LEFT)
        self.tolerance_var = tk.StringVar(value=str(self.config['recognition_tolerance']))
        tk.Entry(tolerance_frame, textvariable=self.tolerance_var, font=('Arial', 10), width=10).pack(side=tk.RIGHT)
        
        # Location
        location_frame = tk.Frame(form_frame, bg='white')
        location_frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(location_frame, text="Location:", font=('Arial', 10), bg='white').pack(side=tk.LEFT)
        self.location_var = tk.StringVar(value=self.config['location'])
        tk.Entry(location_frame, textvariable=self.location_var, font=('Arial', 10), width=30).pack(side=tk.RIGHT)
        
        # Save button
        tk.Button(form_frame, text="Save Settings", command=self.save_settings,
                 bg='#27ae60', fg='white', font=('Arial', 10, 'bold')).pack(pady=20)
    
    def register_from_camera(self):
        if not self.name_entry.get().strip() or not self.id_entry.get().strip():
            messagebox.showerror("Error", "Please enter both name and student ID.")
            return
        
        self.status_var.set("Starting camera for registration...")
        self.is_registering = True
        self.registration_thread = threading.Thread(target=self._register_camera_thread)
        self.registration_thread.daemon = True
        self.registration_thread.start()
    
    def _register_camera_thread(self):
        cap = cv2.VideoCapture(0)
        name = self.name_entry.get().strip()
        student_id = self.id_entry.get().strip()
        
        while self.is_registering:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb)
            
            # Draw rectangles around faces
            for (top, right, bottom, left) in faces:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Convert frame for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((640, 480), Image.Resampling.LANCZOS)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Update GUI in main thread
            self.root.after(0, self._update_camera_preview, frame_tk)
            
            # Check for face capture
            if faces and len(faces) > 0:
                # Auto-capture after 3 seconds of stable face detection
                encodings = face_recognition.face_encodings(rgb, faces)
                if encodings:
                    self.known_encodings.append(encodings[0])
                    self.known_names.append(name)
                    self.known_ids.append(student_id)
                    
                    # Save photo
                    photo_name = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                    cv2.imwrite(os.path.join(self.faces_dir, photo_name), frame)
                    
                    self.save_faces()
                    self.root.after(0, self._registration_complete, name)
                    break
        
        cap.release()
        self.is_registering = False
    
    def _update_camera_preview(self, frame_tk):
        self.camera_label.configure(image=frame_tk, text="")
        self.camera_label.image = frame_tk
    
    def _registration_complete(self, name):
        self.status_var.set(f"Registration completed for {name}")
        messagebox.showinfo("Success", f"{name} registered successfully!")
        self.name_entry.delete(0, tk.END)
        self.id_entry.delete(0, tk.END)
        self.update_faces_list()
        self.camera_label.configure(image="", text="Camera Preview")
        self.camera_label.image = None
    
    def register_from_image(self):
        if not self.name_entry.get().strip() or not self.id_entry.get().strip():
            messagebox.showerror("Error", "Please enter both name and student ID.")
            return
        
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not image_path:
            return
        
        try:
            image = face_recognition.load_image_file(image_path)
            faces = face_recognition.face_locations(image)
            
            if not faces:
                messagebox.showerror("Error", "No face detected in the image.")
                return
            
            encodings = face_recognition.face_encodings(image, faces)
            name = self.name_entry.get().strip()
            student_id = self.id_entry.get().strip()
            
            self.known_encodings.append(encodings[0])
            self.known_names.append(name)
            self.known_ids.append(student_id)
            
            # Save image
            img = Image.open(image_path)
            img.save(os.path.join(self.faces_dir, f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"))
            
            self.save_faces()
            self.update_faces_list()
            
            messagebox.showinfo("Success", f"{name} registered successfully from image!")
            self.name_entry.delete(0, tk.END)
            self.id_entry.delete(0, tk.END)
            self.status_var.set(f"Registration completed for {name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Registration failed: {str(e)}")
    
    def update_faces_list(self):
        self.faces_listbox.delete(0, tk.END)
        for i, name in enumerate(self.known_names):
            self.faces_listbox.insert(tk.END, f"{name} ({self.known_ids[i]})")
    
    def delete_selected_face(self):
        selection = self.faces_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a face to delete.")
            return
        
        index = selection[0]
        name = self.known_names[index]
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {name}?"):
            self.known_names.pop(index)
            self.known_encodings.pop(index)
            self.known_ids.pop(index)
            self.save_faces()
            self.update_faces_list()
            messagebox.showinfo("Success", f"{name} deleted successfully!")
    
    def start_attendance(self):
        self.is_camera_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Attendance system started")
        
        self.camera_thread = threading.Thread(target=self._attendance_camera_thread)
        self.camera_thread.daemon = True
        self.camera_thread.start()
    
    def stop_attendance(self):
        self.is_camera_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Attendance system stopped")
        
        # Clear camera display
        self.attendance_camera_label.configure(image="", text="Attendance Camera View")
        self.attendance_camera_label.image = None
    
    def _attendance_camera_thread(self):
        cap = cv2.VideoCapture(0)
        
        while self.is_camera_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Face recognition
            rgb_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB)
            
            locations = face_recognition.face_locations(rgb_small)
            encodings = face_recognition.face_encodings(rgb_small, locations)
            
            for encoding, (top, right, bottom, left) in zip(encodings, locations):
                matches = face_recognition.compare_faces(self.known_encodings, encoding, 
                                                       float(self.config['recognition_tolerance']))
                name = "Unknown"
                student_id = ""
                
                if True in matches:
                    idx = matches.index(True)
                    name = self.known_names[idx]
                    student_id = self.known_ids[idx]
                    
                    if not self.is_already_marked(name):
                        photo_name = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                        photo_path = os.path.join(self.attendance_dir, photo_name)
                        cv2.imwrite(photo_path, frame)
                        self.mark_attendance(name, student_id, photo_path)
                        
                        # Update summary
                        self.root.after(0, self.update_summary)
                
                # Draw rectangle and label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                scale = 4
                cv2.rectangle(frame, (left * scale, top * scale), (right * scale, bottom * scale), color, 2)
                cv2.putText(frame, name, (left * scale, top * scale - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add title
            cv2.putText(frame, "Attendance System", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert frame for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((640, 480), Image.Resampling.LANCZOS)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Update GUI
            self.root.after(0, self._update_attendance_camera, frame_tk)
        
        cap.release()
    
    def _update_attendance_camera(self, frame_tk):
        self.attendance_camera_label.configure(image=frame_tk, text="")
        self.attendance_camera_label.image = frame_tk
    
    def is_late(self, current_time):
        work_start = datetime.strptime(self.config['work_start'], '%H:%M')
        late_time = work_start + timedelta(minutes=int(self.config['late_threshold']))
        return current_time > late_time.time()
    
    def is_already_marked(self, name):
        df = pd.read_csv(self.attendance_file)
        today = datetime.now().strftime("%Y-%m-%d")
        return ((df['Name'] == name) & (df['Date'] == today)).any()
    
    def mark_attendance(self, name, student_id, photo_path):
        now = datetime.now()
        status = "Late" if self.is_late(now.time()) else "Present"
        
        df = pd.read_csv(self.attendance_file)
        new_record = {
            'Name': name,
            'Student_ID': student_id,
            'Date': now.strftime("%Y-%m-%d"),
            'Time': now.strftime("%H:%M:%S"),
            'Status': status,
            'Photo_Path': photo_path
        }
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        df.to_csv(self.attendance_file, index=False)
        
        self.status_var.set(f"{name} marked {status} at {now.strftime('%H:%M:%S')}")
    
    def update_summary(self):
        df = pd.read_csv(self.attendance_file)
        today = datetime.now().strftime("%Y-%m-%d")
        today_records = df[df['Date'] == today]
        
        present_count = len(today_records[today_records['Status'] == 'Present'])
        late_count = len(today_records[today_records['Status'] == 'Late'])
        total_count = len(today_records)
        
        summary_text = f"""Today's Attendance Summary ({today}):
Present: {present_count}
Late: {late_count}
Total: {total_count}"""
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, summary_text)
    
    def refresh_records(self):
        # Clear existing records
        for item in self.records_tree.get_children():
            self.records_tree.delete(item)
        
        # Load and display records
        df = pd.read_csv(self.attendance_file)
        for _, row in df.iterrows():
            self.records_tree.insert('', tk.END, values=(
                row['Name'], row['Student_ID'], row['Date'], row['Time'], row['Status']
            ))
    
    def export_report(self):
        df = pd.read_csv(self.attendance_file)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.report_dir, f"report_{timestamp}.csv")
        df.to_csv(report_file, index=False)
        messagebox.showinfo("Success", f"Report exported to {report_file}")
        self.status_var.set(f"Report exported successfully")
    
    def save_settings(self):
        try:
            self.config['work_start'] = self.start_time_var.get()
            self.config['work_end'] = self.end_time_var.get()
            self.config['late_threshold'] = int(self.late_threshold_var.get())
            self.config['recognition_tolerance'] = float(self.tolerance_var.get())
            self.config['location'] = self.location_var.get()
            
            self.save_config()
            messagebox.showinfo("Success", "Settings saved successfully!")
            self.status_var.set("Settings updated")
        except ValueError:
            messagebox.showerror("Error", "Invalid values in settings. Please check your input.")
    
    def on_closing(self):
        if self.is_camera_running:
            self.stop_attendance()
        if hasattr(self, 'is_registering') and self.is_registering:
            self.is_registering = False
        self.root.destroy()

def main():
    root = tk.Tk()
    app = KashviSmartFaceAttendanceGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
    
