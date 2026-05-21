import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import os
import shutil
from datetime import datetime

class ImageCollectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Thu thập ảnh ký hiệu tay")
        self.root.geometry("1000x800")
        
        # Tạo frame chính
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame bên trái - Danh sách thư mục
        left_frame = ttk.LabelFrame(main_frame, text="Danh sách ký tự", padding="5")
        left_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Tạo scrollbar cho danh sách
        scrollbar = ttk.Scrollbar(left_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Listbox hiển thị các ký tự
        self.char_listbox = tk.Listbox(left_frame, yscrollcommand=scrollbar.set, width=20, height=30)
        self.char_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.char_listbox.yview)
        
        # Frame bên phải - Hiển thị ảnh
        right_frame = ttk.LabelFrame(main_frame, text="Ảnh đã thu thập", padding="5")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Canvas để hiển thị ảnh
        self.canvas = tk.Canvas(right_frame, width=600, height=400, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Frame chứa các nút điều khiển
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=10)
        
        # Nút thêm ảnh
        ttk.Button(control_frame, text="Thêm ảnh", command=self.add_images).pack(side=tk.LEFT, padx=5)
        
        # Nút xóa ảnh
        ttk.Button(control_frame, text="Xóa ảnh", command=self.delete_image).pack(side=tk.LEFT, padx=5)
        
        # Nút tạo thư mục mới
        ttk.Button(control_frame, text="Tạo thư mục mới", command=self.create_new_folder).pack(side=tk.LEFT, padx=5)
        
        # Label hiển thị thông tin
        self.info_label = ttk.Label(main_frame, text="")
        self.info_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Khởi tạo biến
        self.current_char = None
        self.current_images = []
        self.current_image_index = 0
        
        # Tải danh sách thư mục
        self.load_folders()
        
        # Bind sự kiện
        self.char_listbox.bind('<<ListboxSelect>>', self.on_char_select)
        
    def load_folders(self):
        """Tải danh sách các thư mục ký tự"""
        self.char_listbox.delete(0, tk.END)
        if os.path.exists('data/real_images'):
            for char in sorted(os.listdir('data/real_images')):
                if os.path.isdir(os.path.join('data/real_images', char)):
                    self.char_listbox.insert(tk.END, char)
    
    def on_char_select(self, event):
        """Xử lý khi chọn một ký tự"""
        selection = self.char_listbox.curselection()
        if selection:
            self.current_char = self.char_listbox.get(selection[0])
            self.load_images()
    
    def load_images(self):
        """Tải ảnh của ký tự được chọn"""
        self.current_images = []
        self.current_image_index = 0
        
        if self.current_char:
            char_dir = os.path.join('data/real_images', self.current_char)
            if os.path.exists(char_dir):
                self.current_images = [f for f in os.listdir(char_dir) 
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                self.current_images.sort()
                
                if self.current_images:
                    self.show_current_image()
                else:
                    self.canvas.delete("all")
                    self.canvas.create_text(300, 200, text="Không có ảnh", font=('Arial', 20))
                
                self.update_info()
    
    def show_current_image(self):
        """Hiển thị ảnh hiện tại"""
        if 0 <= self.current_image_index < len(self.current_images):
            img_path = os.path.join('data/real_images', self.current_char, 
                                  self.current_images[self.current_image_index])
            try:
                img = Image.open(img_path)
                # Tính toán kích thước để fit vào canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                img.thumbnail((canvas_width, canvas_height))
                
                photo = ImageTk.PhotoImage(img)
                self.canvas.delete("all")
                self.canvas.create_image(canvas_width//2, canvas_height//2, 
                                      image=photo, anchor=tk.CENTER)
                self.canvas.image = photo
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể hiển thị ảnh: {e}")
    
    def update_info(self):
        """Cập nhật thông tin hiển thị"""
        if self.current_char:
            total = len(self.current_images)
            current = self.current_image_index + 1 if total > 0 else 0
            self.info_label.config(text=f"Ký tự: {self.current_char} | Ảnh: {current}/{total}")
    
    def add_images(self):
        """Thêm ảnh mới vào thư mục hiện tại"""
        if not self.current_char:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một ký tự trước!")
            return
            
        files = filedialog.askopenfilenames(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if files:
            char_dir = os.path.join('data/real_images', self.current_char)
            os.makedirs(char_dir, exist_ok=True)
            
            for file in files:
                try:
                    # Tạo tên file mới với timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ext = os.path.splitext(file)[1]
                    new_name = f"{timestamp}{ext}"
                    
                    # Copy file vào thư mục
                    shutil.copy2(file, os.path.join(char_dir, new_name))
                except Exception as e:
                    messagebox.showerror("Lỗi", f"Không thể thêm ảnh {file}: {e}")
            
            self.load_images()
            messagebox.showinfo("Thành công", f"Đã thêm {len(files)} ảnh!")
    
    def delete_image(self):
        """Xóa ảnh hiện tại"""
        if not self.current_char or not self.current_images:
            return
            
        if messagebox.askyesno("Xác nhận", "Bạn có chắc muốn xóa ảnh này?"):
            try:
                img_path = os.path.join('data/real_images', self.current_char,
                                      self.current_images[self.current_image_index])
                os.remove(img_path)
                self.load_images()
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể xóa ảnh: {e}")
    
    def create_new_folder(self):
        """Tạo thư mục mới cho ký tự"""
        char = tk.simpledialog.askstring("Tạo thư mục mới", "Nhập ký tự:")
        if char:
            char_dir = os.path.join('data/real_images', char)
            if os.path.exists(char_dir):
                messagebox.showwarning("Cảnh báo", "Thư mục này đã tồn tại!")
            else:
                os.makedirs(char_dir)
                self.load_folders()
                # Chọn thư mục mới tạo
                index = self.char_listbox.get(0, tk.END).index(char)
                self.char_listbox.selection_set(index)
                self.on_char_select(None)

if __name__ == '__main__':
    root = tk.Tk()
    app = ImageCollectorGUI(root)
    root.mainloop() 