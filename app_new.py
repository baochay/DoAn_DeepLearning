import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image
import torch
import os
import cv2
import time
from src.model import get_model
from src.data_loader import load_data
from src.data_collector_new import DataCollectorModern
from torchvision import transforms

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

REAL_IMAGE_DIR = 'data/real_images'
MODELS_DIR = 'models'


def preprocess_for_predict(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    return img


def predict_image(image_path, model, device):
    img = preprocess_for_predict(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    _, predicted = output.max(1)
    return predicted.item()


def save_image_with_label(image_path, label):
    img = preprocess_for_predict(image_path)
    label_dir = os.path.join(REAL_IMAGE_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    filename = f'{int(time.time() * 1000)}.png'
    img.save(os.path.join(label_dir, filename))


class AppModern(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title('Ứng Dụng Nhận Diện Ngôn Ngữ Ký Hiệu Tay Bằng CNN')
        self.geometry('480x700')
        self.resizable(False, False)

        self.image_path = None
        self.last_label = None

        _, _, _, num_classes = load_data()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.model = None

        # --- UI DESIGN ---
        self.title_label = ctk.CTkLabel(self, text="Nhận Diện Ngôn Ngữ Ký Hiệu Tay Bằng CNN",
                                        font=ctk.CTkFont(size=18, weight="bold"))
        self.title_label.pack(pady=20)

        self.image_frame = ctk.CTkFrame(self, width=220, height=220, corner_radius=12)
        self.image_frame.pack_propagate(False)
        self.image_frame.pack(pady=10)

        self.img_label = ctk.CTkLabel(self.image_frame, text="Chưa tải tệp ảnh lên", font=ctk.CTkFont(size=13))
        self.img_label.pack(expand=True, fill="both")

        self.result_card = ctk.CTkFrame(self, width=400, height=65, fg_color=("#EAEAEA", "#2B2B2B"), corner_radius=10)
        self.result_card.pack_propagate(False)
        self.result_card.pack(pady=15)

        self.result_label = ctk.CTkLabel(self.result_card, text='Trạng thái: Sẵn sàng thực thi',
                                         font=ctk.CTkFont(size=15, weight="bold"))
        self.result_label.pack(expand=True)

        self.model_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.model_frame.pack(pady=5)
        ctk.CTkLabel(self.model_frame, text='Chọn models (.pth):', font=ctk.CTkFont(size=13)).pack(side=tk.LEFT,
                                                                                                          padx=5)

        self.model_files = self.get_model_files()
        self.model_var = ctk.StringVar(value=self.model_files[0] if self.model_files else "Trống")
        self.model_menu = ctk.CTkOptionMenu(self.model_frame, variable=self.model_var,
                                            values=self.model_files if self.model_files else ["Trống"],
                                            command=self.change_model, width=180)
        self.model_menu.pack(side=tk.LEFT, padx=5)

        if self.model_files:
            self.load_model(self.model_var.get())

        self.btn_grid = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_grid.pack(pady=15)

        self.btn_choose = ctk.CTkButton(self.btn_grid, text='Chọn Ảnh Tĩnh', command=self.choose_image, width=120,
                                        corner_radius=6)
        self.btn_choose.grid(row=0, column=0, padx=8, pady=5)

        self.btn_predict = ctk.CTkButton(self.btn_grid, text='Chạy Dự Đoán', command=self.recognize, fg_color="#2ecc71",
                                         hover_color="#27ae60", width=120, corner_radius=6)
        self.btn_predict.grid(row=0, column=1, padx=8, pady=5)

        self.btn_fix = ctk.CTkButton(self.btn_grid, text='Sửa Nhãn Ảnh', command=self.fix_label, fg_color="#e67e22",
                                     hover_color="#d35400", width=120, corner_radius=6)
        self.btn_fix.grid(row=0, column=2, padx=8, pady=5)

        self.btn_collect = ctk.CTkButton(self, text='⚙️ Thu Thập Dữ Liệu', fg_color="#34495e",
                                         hover_color="#2c3e50", height=42, font=ctk.CTkFont(size=13, weight="bold"),
                                         corner_radius=8, command=self.open_data_collector)
        self.btn_collect.pack(fill="x", padx=40, pady=6)

        self.btn_webcam = ctk.CTkButton(self, text='📷 Dự Đoán Webcam Real-time',
                                        command=self.predict_from_webcam, height=42,
                                        font=ctk.CTkFont(size=13, weight="bold"), corner_radius=8)
        self.btn_webcam.pack(fill="x", padx=40, pady=6)

        self.btn_train = ctk.CTkButton(self, text='🚀 Bắt Đầu Train Lại Mạng', command=self.retrain,
                                       fg_color="#9b59b6", hover_color="#8e44ad", height=42,
                                       font=ctk.CTkFont(size=13, weight="bold"), corner_radius=8)
        self.btn_train.pack(fill="x", padx=40, pady=6)

    def get_model_files(self):
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
        files.sort()
        return files

    def load_model(self, model_file):
        if not model_file or model_file == "Trống":
            return
        self.model = get_model(self.num_classes, self.device)
        self.model.load_state_dict(torch.load(os.path.join(MODELS_DIR, model_file), map_location=self.device))
        self.model.eval()

    def change_model(self, value):
        self.load_model(value)
        messagebox.showinfo('Trọng số mạng', f'Đã nạp thành công tệp mạng: {value}')
        self.result_label.configure(text='Đã cập nhật mô hình tính toán!')

    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[('Image Files', '*.png;*.jpg;*.jpeg;*.bmp')])
        if path:
            self.image_path = path
            img = Image.open(path)
            self.ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 200))
            self.img_label.configure(image=self.ctk_img, text="")
            self.result_label.configure(text='Nạp ảnh tĩnh thành công.')
            self.last_label = None

    def recognize(self):
        if not self.image_path:
            messagebox.showwarning('Cảnh báo', 'Vui lòng chọn tệp hình ảnh đầu vào!')
            return
        if not self.model:
            messagebox.showwarning('Cảnh báo', 'Hệ thống chưa tìm thấy tệp trọng số mạng phù hợp!')
            return
        label = predict_image(self.image_path, self.model, self.device)
        char = LABELS[label] if 0 <= label < len(LABELS) else '?'
        self.result_label.configure(text=f'KẾT QUẢ DỰ ĐOÁN ➔ KÝ TỰ: {char}  (Mã lớp: {label})')
        self.last_label = label

    def fix_label(self):
        if not self.image_path:
            messagebox.showwarning('Thông báo', 'Cần chọn ảnh tĩnh để cấu trúc nhãn dữ liệu.')
            return

        label_win = ctk.CTkToplevel(self)
        label_win.title('Cấu trúc nhãn đúng')
        label_win.geometry('260x140')
        label_win.attributes("-topmost", True)

        ctk.CTkLabel(label_win, text='Chọn chính xác ký tự chữ cái:').pack(pady=10)
        var = ctk.StringVar(value=LABELS[self.last_label] if self.last_label is not None else LABELS[0])
        option = ctk.CTkOptionMenu(label_win, variable=var, values=LABELS)
        option.pack(pady=5)

        def save_and_close():
            label = var.get()
            save_image_with_label(self.image_path, label)
            messagebox.showinfo('Hệ thống dữ liệu', f'Ảnh lưu vào nhóm {label}! Nhấn Train lại để cập nhật mô hình.')
            label_win.destroy()

        ctk.CTkButton(label_win, text='Xác Nhận Lưu', command=save_and_close).pack(pady=10)

    def open_data_collector(self):
        DataCollectorModern(self)

    def predict_from_webcam(self):
        win = ctk.CTkToplevel(self)
        win.title("Dự đoán Webcam thời gian thực")
        win.geometry("500x680")
        win.attributes("-topmost", True)
        win.resizable(False, False)

        model_select_frame = ctk.CTkFrame(win, fg_color="transparent")
        model_select_frame.pack(pady=10)
        ctk.CTkLabel(model_select_frame, text='Sử dụng mô hình:', font=ctk.CTkFont(size=12, weight="bold")).pack(
            side=tk.LEFT, padx=5)

        w_model_var = ctk.StringVar(value=self.model_var.get())
        w_model_menu = ctk.CTkOptionMenu(model_select_frame, variable=w_model_var,
                                         values=self.model_files if self.model_files else ["Trống"], width=160)
        w_model_menu.pack(side=tk.LEFT, padx=5)

        btn_frame = ctk.CTkFrame(win, fg_color="transparent")
        btn_frame.pack(pady=5)

        capture_btn = ctk.CTkButton(btn_frame, text="📸 Chụp Dự Đoán", fg_color="#3498db", width=120, corner_radius=6)
        capture_btn.pack(side=tk.LEFT, padx=5)
        retry_btn = ctk.CTkButton(btn_frame, text="🔄 Tiếp Tục Quét", width=120, state="disabled", corner_radius=6)
        retry_btn.pack(side=tk.LEFT, padx=5)
        fix_btn = ctk.CTkButton(btn_frame, text="🛠️ Sửa & Gán Nhãn", fg_color="#e67e22", hover_color="#d35400",
                                width=120, state="disabled", corner_radius=6)
        fix_btn.pack(side=tk.LEFT, padx=5)

        lmain = ctk.CTkLabel(win, text="Đang mở luồng Camera...")
        lmain.pack(pady=5)

        res_card = ctk.CTkFrame(win, width=420, height=50, fg_color=("#D5F5E3", "#196F3D"), corner_radius=8)
        res_card.pack_propagate(False)
        res_card.pack(pady=5)
        result_label = ctk.CTkLabel(res_card, text="Sẵn sàng nhận diện!", font=ctk.CTkFont(size=15, weight="bold"))
        result_label.pack(expand=True)

        slider_frame = ctk.CTkFrame(win)
        slider_frame.pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(slider_frame, text='Căn lề hộp cắt trục Ngang (X):').pack(anchor="w", padx=10)
        x1_slider = ctk.CTkSlider(slider_frame, from_=0, to=200)
        x1_slider.set(80)
        x1_slider.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(slider_frame, text='Căn lề hộp cắt trục Dọc (Y):').pack(anchor="w", padx=10)
        y1_slider = ctk.CTkSlider(slider_frame, from_=0, to=200)
        y1_slider.set(40)
        y1_slider.pack(fill="x", padx=10, pady=2)

        cap = cv2.VideoCapture(0)
        webcam_model = get_model(self.num_classes, self.device)

        def load_selected_webcam_model(val=None):
            selected = w_model_var.get()
            if selected and selected != "Trống":
                try:
                    webcam_model.load_state_dict(
                        torch.load(os.path.join(MODELS_DIR, selected), map_location=self.device))
                    webcam_model.eval()
                except Exception as e:
                    print("Lỗi nạp model phụ:", e)

        load_selected_webcam_model()
        w_model_menu.configure(command=load_selected_webcam_model)

        def show_frame():
            if not cap or not cap.isOpened():
                return
            ret, frame = cap.read()
            if not ret:
                win.after(15, show_frame)
                return

            frame = cv2.flip(frame, 1)
            x1, y1 = int(x1_slider.get()), int(y1_slider.get())
            x2, y2 = x1 + 160, y1 + 160

            cv2.rectangle(frame, (x1, y1), (x2, y2), (46, 204, 113), 2)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(360, 270))
            lmain.configure(image=imgtk, text="")
            win._after_id = win.after(15, show_frame)

        def capture():
            ret, frame = cap.read()
            if not ret:
                return
            frame = cv2.flip(frame, 1)
            x1, y1 = int(x1_slider.get()), int(y1_slider.get())
            x2, y2 = x1 + 160, y1 + 160
            crop = frame[y1:y2, x1:x2]

            temp_path = "temp_webcam.png"
            cv2.imwrite(temp_path, crop)
            self.image_path = temp_path

            img = Image.open(temp_path)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 200))
            lmain.configure(image=imgtk)

            try:
                label = predict_image(temp_path, webcam_model, self.device)
                char = LABELS[label] if 0 <= label < len(LABELS) else '?'
                result_label.configure(text=f'🎯 KẾT QUẢ DỰ ĐOÁN WEBCAM: CHỮ {char}')
                self.last_label = label
                fix_btn.configure(state="normal")
            except Exception as e:
                result_label.configure(text="Lỗi xử lý dự đoán.")
                print(e)

            capture_btn.configure(state="disabled")
            retry_btn.configure(state="normal")

            if hasattr(win, '_after_id'):
                win.after_cancel(win._after_id)

        def retry():
            result_label.configure(text='Đang nhận diện trực tiếp...')
            capture_btn.configure(state="normal")
            retry_btn.configure(state="disabled")
            fix_btn.configure(state="disabled")
            show_frame()

        def fix_label_webcam():
            if self.last_label is None:
                return
            fix_win = ctk.CTkToplevel(win)
            fix_win.title('Sửa nhãn dữ liệu lỗi')
            fix_win.geometry('240x140')
            fix_win.attributes("-topmost", True)
            ctk.CTkLabel(fix_win, text='Chọn nhãn chuẩn thực tế:').pack(pady=8)
            w_fix_var = ctk.StringVar(value=LABELS[self.last_label])
            ctk.CTkOptionMenu(fix_win, variable=w_fix_var, values=LABELS).pack(pady=5)

            def save_w():
                save_image_with_label(self.image_path, w_fix_var.get())
                messagebox.showinfo('Hệ thống', 'Đã lưu ảnh lỗi vào tập chờ train!')
                fix_win.destroy()

            ctk.CTkButton(fix_win, text='Xác nhận lưu', command=save_w).pack(pady=5)

        capture_btn.configure(command=capture)
        retry_btn.configure(command=retry)
        fix_btn.configure(command=fix_label_webcam)

        show_frame()

        def on_close():
            if hasattr(win, '_after_id'):
                win.after_cancel(win._after_id)
            cap.release()
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    def retrain(self):
        train_win = ctk.CTkToplevel(self)
        train_win.title("Huấn luyện tối ưu hóa")
        train_win.geometry("340x120")
        train_win.attributes("-topmost", True)

        lbl = ctk.CTkLabel(train_win, text="Đang chạy tiến trình gộp tập ảnh và huấn luyện CNN...",
                           font=ctk.CTkFont(size=12))
        lbl.pack(pady=25)
        train_win.update()

        from preprocess_real_images import preprocess_images
        preprocess_images('data/real_images', 'data/real_images_processed')

        try:
            from src.train_finetune import main as train_main
            train_main()
        except Exception:
            from src.train import main as train_main
            train_main()

        self.model_files = self.get_model_files()
        if self.model_files:
            self.model_var.set(self.model_files[-1])
            self.load_model(self.model_files[-1])

        train_win.destroy()
        messagebox.showinfo("Thành công", "Mô hình CNN đã cập nhật trọng số mới ưu việt nhất thành công!")


if __name__ == '__main__':
    app = AppModern()
    app.mainloop()