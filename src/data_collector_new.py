import cv2
import os
import time
from datetime import datetime
from PIL import Image
import customtkinter as ctk
from tkinter import messagebox

LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]


class DataCollectorModern(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Công cụ thu thập dữ liệu ký hiệu tay (Modern UI)")
        self.geometry("1100完整x700")
        self.geometry("1100x700")
        self.resizable(False, False)
        self.attributes("-topmost", True)

        self.cap = None
        self.is_capturing = False
        self.current_frame = None
        self._after_id = None

        self.crop_x = 80
        self.crop_y = 40
        self.crop_size = 200

        self.save_dir = "data/real_images"
        os.makedirs(self.save_dir, exist_ok=True)

        self.setup_ui()
        self.start_webcam()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # ---- CỘT TRÁI ----
        left_frame = ctk.CTkFrame(self, corner_radius=15)
        left_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        ctk.CTkLabel(left_frame, text="BẢNG ĐIỀU KHIỂN", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=15)

        ctk.CTkLabel(left_frame, text="Chọn ký tự cần thu thập:", font=ctk.CTkFont(size=13)).pack(anchor="w", padx=20,
                                                                                                  pady=2)
        self.char_var = ctk.StringVar(value=LABELS[0])
        self.char_menu = ctk.CTkOptionMenu(left_frame, variable=self.char_var, values=LABELS, width=180)
        self.char_menu.pack(pady=10)

        slider_card = ctk.CTkFrame(left_frame, fg_color=("#EAEAEA", "#2B2B2B"), corner_radius=10)
        slider_card.pack(fill="x", padx=15, pady=15)

        ctk.CTkLabel(slider_card, text="ĐIỀU CHỈNH KHUNG CẮT (ROI)", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=5)

        ctk.CTkLabel(slider_card, text="Tọa độ Ngang (X):").pack(anchor="w", padx=15)
        self.x_slider = ctk.CTkSlider(slider_card, from_=0, to=400, command=self.on_slider_change)
        self.x_slider.set(self.crop_x)
        self.x_slider.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(slider_card, text="Tọa độ Dọc (Y):").pack(anchor="w", padx=15)
        self.y_slider = ctk.CTkSlider(slider_card, from_=0, to=300, command=self.on_slider_change)
        self.y_slider.set(self.crop_y)
        self.y_slider.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(slider_card, text="Kích thước khung:").pack(anchor="w", padx=15)
        self.size_slider = ctk.CTkSlider(slider_card, from_=100, to=350, command=self.on_slider_change)
        self.size_slider.set(self.crop_size)
        self.size_slider.pack(fill="x", padx=15, pady=5)

        self.btn_capture = ctk.CTkButton(left_frame, text="📸 CHỤP ẢNH (Space)",
                                         font=ctk.CTkFont(size=14, weight="bold"),
                                         fg_color="#2ecc71", hover_color="#27ae60", height=45,
                                         command=self.capture_image)
        self.btn_capture.pack(fill="x", padx=20, pady=20)
        self.bind("<space>", lambda event: self.capture_image())

        self.status_label = ctk.CTkLabel(left_frame, text="Sẵn sàng thu thập...", text_color="gray",
                                         font=ctk.CTkFont(size=12))
        self.status_label.pack(pady=10, padx=10)

        # ---- CỘT PHẢI ----
        right_frame = ctk.CTkFrame(self, fg_color="transparent")
        right_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_columnconfigure(1, weight=1)
        right_frame.grid_rowconfigure(0, weight=1)

        webcam_card = ctk.CTkFrame(right_frame, corner_radius=12)
        webcam_card.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(webcam_card, text="CAMERA TRỰC TIẾP", font=ctk.CTkFont(weight="bold")).pack(pady=10)

        self.video_label = ctk.CTkLabel(webcam_card, text="Đang kết nối camera...", fg_color="black", width=440,
                                        height=440, corner_radius=10)
        self.video_label.pack(expand=True, padx=15, pady=15)

        preview_card = ctk.CTkFrame(right_frame, corner_radius=12)
        preview_card.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(preview_card, text="ẢNH CẮT THỰC TẾ", font=ctk.CTkFont(weight="bold")).pack(pady=10)

        self.preview_label = ctk.CTkLabel(preview_card, text="Vùng tay sau cắt", fg_color="#1D1E1E", width=220,
                                          height=220, corner_radius=10)
        self.preview_label.pack(expand=True, padx=15, pady=15)

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.video_label.configure(
                text="❌ Không thể kết nối với Webcam!\nĐảm bảo cửa sổ webcam cũ đã đóng hoàn toàn.")
            return
        self.is_capturing = True
        self.update_frame()

    def on_slider_change(self, *args):
        self.crop_x = int(self.x_slider.get())
        self.crop_y = int(self.y_slider.get())
        self.crop_size = int(self.size_slider.get())

    def update_frame(self):
        if not self.is_capturing or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            self.crop_x = min(self.crop_x, w - self.crop_size)
            self.crop_y = min(self.crop_y, h - self.crop_size)

            # Lưu trữ vùng cắt thực tế để lưu file ảnh
            self.current_frame = frame_rgb[
                self.crop_y:self.crop_y + self.crop_size, self.crop_x:self.crop_x + self.crop_size]

            # Vẽ hình vuông định vị trên khung camera luồng
            cv2.rectangle(frame_rgb, (self.crop_x, self.crop_y),
                          (self.crop_x + self.crop_size, self.crop_y + self.crop_size), (46, 204, 113), 3)

            img_main = Image.fromarray(frame_rgb)
            ctk_main = ctk.CTkImage(light_image=img_main, dark_image=img_main, size=(440, 330))
            self.video_label.configure(image=ctk_main, text="")

            img_crop = Image.fromarray(self.current_frame)
            ctk_crop = ctk.CTkImage(light_image=img_crop, dark_image=img_crop, size=(220, 220))
            self.preview_label.configure(image=ctk_crop, text="")

        self._after_id = self.after(20, self.update_frame)

    def capture_image(self):
        if self.current_frame is None:
            return
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            char = self.char_var.get()
            filename = f"{char}_{timestamp}.png"
            char_dir = os.path.join(self.save_dir, char)
            os.makedirs(char_dir, exist_ok=True)

            filepath = os.path.join(char_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
            self.status_label.configure(text=f"✅ Đã lưu: {filename}", text_color="#2ecc71")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không lưu được ảnh: {str(e)}")

    def on_close(self):
        self.is_capturing = False
        if self._after_id:
            self.after_cancel(self._after_id)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.destroy()