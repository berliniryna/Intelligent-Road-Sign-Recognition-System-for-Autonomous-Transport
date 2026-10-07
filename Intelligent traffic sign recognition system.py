import os
import cv2
import numpy as np
import tkinter as tk
import tensorflow as tf
from tkinter import filedialog, messagebox
from scipy.signal import convolve2d
from PIL import Image, ImageTk, ImageDraw
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical


class SVM:
    def __init__(self, C, gamma, max_iter):
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.alphas = None
        self.b = 0.0
        self.X = None
        self.y = None

    def rbf_kernel(self, x1, x2):
        distance_sq = np.sum((x1 - x2) ** 2)
        return np.exp(-self.gamma * distance_sq)

    def compute_kernel_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.rbf_kernel(X[i], X[j])
        return K

    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples = X.shape[0]
        self.alphas = np.zeros(n_samples)
        self.b = 0.0
        self.errors = -self.y.astype(float)
        K = self.compute_kernel_matrix(X)
        tol = 1e-3
        for it in range(self.max_iter):
            alpha_pairs_changed = 0
            for i in range(n_samples):
                i_ = self.errors[i]
                E_i = i_
                r_i = self.y[i] * E_i
                if (r_i < -tol and self.alphas[i] < self.C) or (r_i > tol and self.alphas[i] > 0):
                    j = self._select_second_alpha(i, E_i, n_samples)
                    if j == -1:
                        continue
                    alphas_i_old = self.alphas[i].copy()
                    alphas_j_old = self.alphas[j].copy()
                    E_j = self.errors[j]
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    if L == H: continue
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0: continue
                    self.alphas[j] -= self.y[j] * (E_i - E_j) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    if abs(self.alphas[j] - alphas_j_old) < 1e-5:
                        continue
                    self.alphas[i] += self.y[i] * self.y[j] * (alphas_j_old - self.alphas[j])
                    b1 = self.b - E_i - self.y[i] * (self.alphas[i] - alphas_i_old) * K[i, i] \
                         - self.y[j] * (self.alphas[j] - alphas_j_old) * K[i, j]
                    b2 = self.b - E_j - self.y[i] * (self.alphas[i] - alphas_i_old) * K[i, j] \
                         - self.y[j] * (self.alphas[j] - alphas_j_old) * K[j, j]
                    if 0 < self.alphas[i] < self.C:
                        delta_b = b1 - self.b
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        delta_b = b2 - self.b
                        self.b = b2
                    else:
                        delta_b = ((b1 + b2) / 2.0) - self.b
                        self.b = (b1 + b2) / 2.0
                    delta_alpha_i = self.alphas[i] - alphas_i_old
                    delta_alpha_j = self.alphas[j] - alphas_j_old
                    self.errors += (self.y[i] * delta_alpha_i * K[:, i] +
                                    self.y[j] * delta_alpha_j * K[:, j] + delta_b)
                    alpha_pairs_changed += 1
            if alpha_pairs_changed == 0:
                break

    def _select_second_alpha(self, i, E_i, n_samples):
        best_j = -1
        max_delta_E = 0
        valid_indices = np.where((self.alphas > 0) & (self.alphas < self.C))[0]
        if len(valid_indices) > 1:
            for j in valid_indices:
                if i == j: continue
                delta_E = abs(E_i - self.errors[j])
                if delta_E > max_delta_E:
                    max_delta_E = delta_E
                    best_j = j
        if best_j == -1:
            best_j = i
            while best_j == i:
                best_j = np.random.randint(0, n_samples)
        return best_j

    def predict(self, X_test):
        y_pred = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            prediction = 0.0
            for j in range(self.alphas.shape[0]):
                if self.alphas[j] > 0:
                    prediction += self.alphas[j] * self.y[j] * self.rbf_kernel(self.X[j], X_test[i])
            y_pred[i] = np.sign(prediction + self.b)
        return y_pred


class MainInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Settings")
        self.root.geometry("500x500")
        self.root.configure(bg="#f5f5f5")
        self.current_data_path = None
        self.training_data_path = None
        mode_frame = tk.LabelFrame(root, text="Operating mode", bg="#f5f5f5", font=("Arial", 10, "bold"), padx=10,
                                   pady=10)
        mode_frame.pack(fill=tk.X, padx=20, pady=5)
        self.mode_var = tk.StringVar(value="single")
        tk.Radiobutton(mode_frame, text="Single photo", variable=self.mode_var, value="single",
                       bg="#f5f5f5", command=self.update_ui).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="Photo folder", variable=self.mode_var, value="folder",
                       bg="#f5f5f5", command=self.update_ui).pack(anchor=tk.W)
        self.btn_load = tk.Button(root, text="Select photo", command=self.load_data,
            font=("Arial", 10, "bold"), bg="#2196F3", fg="white", height=1)
        self.btn_load.pack(fill=tk.X, padx=20, pady=5)
        self.btn_train = tk.Button(root, text="Select training photos", command=self.load_training_data,
            font=("Arial", 10, "bold"), bg="#009688", fg="white", height=1)
        self.btn_train.pack(fill=tk.X, padx=20, pady=5)
        toggle_frame = tk.LabelFrame(root, text="Image preparation", bg="#f5f5f5", font=("Arial", 10, "bold"),
                                     padx=10,pady=10)
        toggle_frame.pack(fill=tk.X, padx=20, pady=10)
        self.var_contrast = tk.BooleanVar(value=True)
        self.var_sobel = tk.BooleanVar(value=True)
        self.var_morph = tk.BooleanVar(value=True)
        self.var_hu = tk.BooleanVar(value=True)
        tk.Checkbutton(toggle_frame, text="1. Image enhancement", variable=self.var_contrast,
                       bg="#f5f5f5").pack(anchor=tk.W)
        tk.Checkbutton(toggle_frame, text="2. Edge detection", variable=self.var_sobel,
                       bg="#f5f5f5").pack(anchor=tk.W)
        tk.Checkbutton(toggle_frame, text="3. Contour search", variable=self.var_morph,
                       bg="#f5f5f5").pack(anchor=tk.W)
        tk.Checkbutton(toggle_frame, text="4. Sign extraction", variable=self.var_hu,
                       bg="#f5f5f5").pack(anchor=tk.W)
        algo_frame = tk.LabelFrame(root, text="Run classification", bg="#f5f5f5", font=("Arial", 10, "bold"), padx=10,
                                   pady=10)
        algo_frame.pack(fill=tk.X, padx=20, pady=5)
        btn_container = tk.Frame(algo_frame, bg="#f5f5f5")
        btn_container.pack(fill=tk.X, pady=5)
        self.btn_svm = tk.Button(btn_container, text="SVM", command=self.run_svm, font=("Arial", 10, "bold"),
                                 bg="#FF9800", fg="white")
        self.btn_svm.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.btn_cnn = tk.Button(btn_container, text="CNN", command=self.run_cnn, font=("Arial", 10, "bold"),
                                 bg="#E91E63", fg="white")
        self.btn_cnn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.lbl_status = tk.Label(root, text="Status: Waiting...", font=("Arial", 10, "italic"), bg="#f5f5f5",
                                   fg="#555")
        self.lbl_status.pack(pady=10)

    def update_ui(self):
        if self.mode_var.get() == "single":
            self.btn_load.config(text="Select photo")
            self.btn_train.config(text="Select training photos")
        else:
            self.btn_load.config(text="Select photo folder")
            self.btn_train.config(text="Select training photos")

    def load_data(self):
        if self.mode_var.get() == "single":
            path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.ppm")])
        else:
            path = filedialog.askdirectory()
        if path:
            self.current_data_path = path
            self.lbl_status.config(text=f"Data selected: {os.path.basename(path)}")

    def load_training_data(self):
        if self.mode_var.get():
            path = filedialog.askdirectory()
        if path:
            self.training_data_path = path
            self.lbl_status.config(text=f"Training data selected: {os.path.basename(path)}")

    def clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        h, w = image.shape
        grid_y, grid_x = tile_grid_size
        tile_h = h // grid_y
        tile_w = w // grid_x
        mappings = np.zeros((grid_y, grid_x, 256), dtype=np.uint8)
        n_clip = max(1, int(clip_limit * (tile_h * tile_w) / 256))
        for i in range(grid_y):
            for j in range(grid_x):
                tile = image[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w]
                hist, _ = np.histogram(tile.flatten(), bins=256, range=(0, 256))
                excess = 0
                for k in range(256):
                    if hist[k] > n_clip:
                        excess += hist[k] - n_clip
                        hist[k] = n_clip
                l_avg = excess // 256
                residual = excess % 256
                hist += l_avg
                if residual > 0:
                    step = max(1, 256 // residual)
                    for k in range(0, 256, step):
                        if residual > 0:
                            hist[k] += 1
                            residual -= 1
                cdf = hist.cumsum()
                if cdf[-1] > 0:
                    cdf_normalized = (cdf * 255) / cdf[-1]
                else:
                    cdf_normalized = np.arange(256)
                mappings[i, j] = np.round(cdf_normalized).astype(np.uint8)
        result = np.zeros_like(image, dtype=np.uint8)
        center_y = (np.arange(grid_y) + 0.5) * tile_h
        center_x = (np.arange(grid_x) + 0.5) * tile_w
        for y in range(h):
            for x in range(w):
                s = image[y, x]
                idx_y = np.searchsorted(center_y, y) - 1
                idx_x = np.searchsorted(center_x, x) - 1
                y1 = max(0, min(idx_y, grid_y - 1))
                y2 = max(0, min(idx_y + 1, grid_y - 1))
                x1 = max(0, min(idx_x, grid_x - 1))
                x2 = max(0, min(idx_x + 1, grid_x - 1))
                v_TL = mappings[y1, x1, s]
                v_TR = mappings[y1, x2, s]
                v_BL = mappings[y2, x1, s]
                v_BR = mappings[y2, x2, s]
                cy1 = center_y[y1] if y1 == y2 else center_y[y1]
                cx1 = center_x[x1] if x1 == x2 else center_x[x1]
                dy = (y - cy1) / tile_h if y1 != y2 else 0
                dx = (x - cx1) / tile_w if x1 != x2 else 0
                new_val = (1 - dx) * (1 - dy) * v_TL + \
                          dx * (1 - dy) * v_TR + \
                          (1 - dx) * dy * v_BL + \
                          dx * dy * v_BR
                result[y, x] = np.clip(new_val, 0, 255)
        return result

    def calculate_hu_moments(self, cnt, img_shape):
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0: return [0.0] * 7
        m00 = float(len(x_indices))
        m10 = float(np.sum(x_indices))
        m01 = float(np.sum(y_indices))
        cx = m10 / m00
        cy = m01 / m00
        dx = x_indices - cx
        dy = y_indices - cy
        mu20 = float(np.sum(dx ** 2))
        mu02 = float(np.sum(dy ** 2))
        mu11 = float(np.sum(dx * dy))
        mu30 = float(np.sum(dx ** 3))
        mu03 = float(np.sum(dy ** 3))
        mu21 = float(np.sum((dx ** 2) * dy))
        mu12 = float(np.sum(dx * (dy ** 2)))
        eta20 = mu20 / (m00 ** 2)
        eta02 = mu02 / (m00 ** 2)
        eta11 = mu11 / (m00 ** 2)
        eta30 = mu30 / (m00 ** 2.5)
        eta03 = mu03 / (m00 ** 2.5)
        eta21 = mu21 / (m00 ** 2.5)
        eta12 = mu12 / (m00 ** 2.5)
        h1 = eta20 + eta02
        h2 = (eta20 - eta02) ** 2 + 4 * (eta11 ** 2)
        h3 = (eta30 - 3 * eta12) ** 2 + (3 * eta21 - eta03) ** 2
        h4 = (eta30 + eta12) ** 2 + (eta21 + eta03) ** 2
        h5 = (eta30 - 3 * eta12) * (eta30 + eta12) * ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2) + \
             (3 * eta21 - eta03) * (eta21 + eta03) * (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2)
        h6 = (eta20 - eta02) * ((eta30 + eta12) ** 2 - (eta21 + eta03) ** 2) + \
             4 * eta11 * (eta30 + eta12) * (eta21 + eta03)
        h7 = (3 * eta21 - eta03) * (eta30 + eta12) * ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2) - \
             (eta30 - 3 * eta12) * (eta21 + eta03) * (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2)
        return [h1, h2, h3, h4, h5, h6, h7]

    def train_svm(self, dataset_folder):
        target_class = '00014'
        negative_class = 'triangle'
        folders_to_load = {target_class: 1, negative_class: -1}
        X_data, y_data = [], []
        self.lbl_status.config(text="Extracting Hu moments from the dataset...")
        self.root.update()

        for folder_name, label in folders_to_load.items():
            folder_path = os.path.join(dataset_folder, folder_name)
            if not os.path.exists(folder_path):
                messagebox.showerror("Error", f"Folder {folder_path} not found.")
                return
            images_in_folder = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp',
                                                                                  '.ppm'))]
            for idx, img_name in enumerate(images_in_folder[:780]):
                img_path = os.path.join(folder_path, img_name)
                try:
                    img_pil = Image.open(img_path).convert("RGB")
                    img_np = np.array(img_pil)
                    results = self.process_pipeline(img_np)
                    hu_vector = results["hu_moments"]
                    if sum(hu_vector) != 0.0:
                        log_hu = []
                        for h in hu_vector:
                            if h == 0:
                                log_hu.append(0.0)
                            else:
                                log_hu.append(-1.0 * np.sign(h) * np.log10(np.abs(h)))
                        X_data.append(log_hu)
                        y_data.append(label)
                except Exception as e:
                    print(f"Error reading {img_name}: {e}")
                if idx % 10 == 0:
                    self.lbl_status.config(text=f"Processing folder {folder_name}: {(idx+1)/len(images_in_folder)*100:.2f}"
                                                f"%...")
                    self.root.update()
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        if len(X_data) == 0:
            messagebox.showerror("Error", "Failed to extract features from the dataset.")
            return
        self.lbl_status.config(text="Scaling data...")
        self.root.update()
        self.svm_mean = np.mean(X_data, axis=0)
        self.svm_std = np.std(X_data, axis=0)
        self.svm_std[self.svm_std == 0] = 1e-10
        X_data = (X_data - self.svm_mean) / self.svm_std
        self.lbl_status.config(text="Splitting data into training and test sets...")
        self.root.update()
        indices = np.arange(X_data.shape[0])
        np.random.shuffle(indices)
        X_data = X_data[indices]
        y_data = y_data[indices]
        split_idx = int(0.85 * len(X_data))
        X_train, X_test = X_data[:split_idx], X_data[split_idx:]
        y_train, y_test = y_data[:split_idx], y_data[split_idx:]
        self.lbl_status.config(text=f"Training SVM on {len(X_train)} samples...")
        self.root.update()
        self.svm_model = SVM(C=5, gamma=0.05, max_iter=2000)
        self.svm_model.fit(X_train, y_train)
        self.lbl_status.config(text="Testing the model and calculating metrics...")
        self.root.update()
        predictions = self.svm_model.predict(X_test)
        TP = np.sum((y_test == 1) & (predictions == 1))
        FP = np.sum((y_test == -1) & (predictions == 1))
        FN = np.sum((y_test == 1) & (predictions == -1))
        TN = np.sum((y_test == -1) & (predictions == -1))
        accuracy = ((TP + TN) / len(y_test)) * 100
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        print("\n SVM performance evaluation on the test set:")
        print(f"\n Confusion matrix:")
        print(f" [ Actual '+' ]  True Positives (TP): {TP}  | False Negatives (FN): {FN}")
        print(f" [ Actual '-' ]  False Positives (FP): {FP} | True Negatives (TN): {TN}")
        print(f"\n Accuracy:            {accuracy:.2f}%")
        print(f" Precision:           {precision * 100:.2f}%")
        print(f" Recall:              {recall * 100:.2f}%")
        print(f" F1-Score:            {f1 * 100:.2f}%")
        messagebox.showinfo("SVM Results",
                            f"SVM trained successfully.\nAccuracy: {accuracy:.2f}%")
        self.lbl_status.config(text=f"SVM ready. Accuracy: {accuracy:.2f}%")

    def run_svm(self):
        if not self.current_data_path:
            messagebox.showwarning("Warning", "First select input photos or a folder!")
            return
        if not hasattr(self, 'svm_model') or self.svm_model is None:
            if hasattr(self, 'training_data_path') and self.training_data_path:
                self.train_svm(self.training_data_path)
            else:
                messagebox.showwarning("Warning", "The SVM model has not been trained yet, and the dataset path is not "
                                                  "specified.")
                return
        self.lbl_status.config(text="Running SVM algorithm...")
        self.root.update()
        try:
            if self.mode_var.get() == "single" and self.current_data_path:
                img_pil = Image.open(self.current_data_path).convert('RGB')
                img_np = np.array(img_pil)
                results = self.process_pipeline(img_np)
                current_hu = results.get("hu_moments", [0.0] * 7)
                if sum(current_hu) == 0.0:
                    self.lbl_status.config(text="SVM analysis: No signs detected (zero moments)")
                else:
                    current_log_hu = []
                    for h in current_hu:
                        if h == 0:
                            current_log_hu.append(0.0)
                        else:
                            current_log_hu.append(-1.0 * np.sign(h) * np.log10(np.abs(h)))
                    current_log_hu = np.array(current_log_hu)
                    scaled_current_hu = (current_log_hu - self.svm_mean) / self.svm_std
                    current_X = np.array([scaled_current_hu])
                    prediction = self.svm_model.predict(current_X)
                    print(f"File: {os.path.basename(self.current_data_path)}")
                    result_text = "Target sign recognized" if prediction[0] > 0 else "Other sign"
                    print(f"Current Hu moments: {current_hu}")
                    print(f"SVM prediction: {result_text} (Class: {prediction[0]})")
                    self.lbl_status.config(text=f"SVM analysis completed: {result_text}. Information printed to the "
                                                f"console.")
        except Exception as e:
            messagebox.showerror("SVM Error", f"Critical prediction error:\n{str(e)}")
        self._execute_pipeline_based_on_mode()

    def process_single_image(self, file_path):
        try:
            orig_pil = Image.open(file_path).convert('RGB')
            img_np = np.array(orig_pil)
            results = self.process_pipeline(img_np)
            VisualizationWindow(self.root, orig_pil, results)
            self.lbl_status.config(text="Image preparation completed successfully.")
        except Exception as e:
            messagebox.showerror("Processing error", f"Failed to process the image:\n{str(e)}")

    def process_folder(self, folder_path):
        if not hasattr(self, 'svm_model') or self.svm_model is None:
            messagebox.showwarning("Warning", "The SVM model has not been trained yet for folder classification!")
            return
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.ppm')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        if not files:
            messagebox.showinfo("Information", "There are no image files in the selected folder.")
            return
        report_data = []
        for idx, filename in enumerate(files):
            file_path = os.path.join(folder_path, filename)
            try:
                orig_pil = Image.open(file_path).convert("RGB")
                img_np = np.array(orig_pil)
                results = self.process_pipeline(img_np)
                hu_vector = results.get("hu_moments", [0.0] * 7)
                cropped_np = results.get("result", img_np)
                cropped_pil = Image.fromarray(cropped_np)
                if sum(hu_vector) == 0.0:
                    prediction_text = "Sign not detected (0 moments)"
                else:
                    log_hu = []
                    for h in hu_vector:
                        if h == 0:
                            log_hu.append(0.0)
                        else:
                            log_hu.append(-1.0 * np.sign(h) * np.log10(np.abs(h)))
                    log_hu = np.array(log_hu)
                    scaled_hu = (log_hu - self.svm_mean) / self.svm_std
                    prediction = self.svm_model.predict(np.array([scaled_hu]))
                    pred_val = prediction[0] if hasattr(prediction, '__getitem__') else prediction
                    prediction_text = "Target sign recognized" if pred_val > 0 else "Other sign"
                report_data.append({
                    "filename": filename,
                    "cropped_pil": cropped_pil,
                    "prediction_text": prediction_text
                })
                self.lbl_status.config(text=f"Processing folder: {(idx + 1)/len(files)*100:.2f}%...")
                self.root.update()
            except Exception as e:
                print(f"Processing error {filename}: {e}")
        self.lbl_status.config(text="Folder processing completed. Creating report...")
        self.root.update()
        FolderReportWindow(self.root, report_data)
        self.lbl_status.config(text="SVM report created successfully.")

    def train_cnn_model(self, dataset_folder):
        target_class = '00014'
        negative_class = 'triangle'
        folders_to_load = {negative_class: 0, target_class: 1}
        image_data = []
        image_labels = []
        self.lbl_status.config(text="Loading and preparing images...")
        self.root.update()
        for folder_name, label_idx in folders_to_load.items():
            folder_path = os.path.join(dataset_folder, folder_name)
            if not os.path.exists(folder_path):
                messagebox.showerror("Error", f"Folder {folder_path} not found.")
                return
            images_in_folder = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp',
                                                                                  '.ppm'))]
            for img_name in images_in_folder[:780]:
                img_path = os.path.join(folder_path, img_name)
                try:
                    orig_pil = Image.open(img_path)
                    orig_pil.load()
                    resize_image = orig_pil.convert("RGB").resize((30, 30))
                    image_data.append(np.array(resize_image))
                    image_labels.append(label_idx)
                except Exception as e:
                    print(f"Error reading {img_name}: {e}")
        X_data = np.array(image_data, dtype=np.float32) / 255.0
        y_data = np.array(image_labels)
        if len(X_data) == 0:
            messagebox.showerror("Error", "Failed to load data for CNN.")
            return
        y_data = to_categorical(y_data, 2)
        indices = np.arange(X_data.shape[0])
        np.random.shuffle(indices)
        X_data, y_data = X_data[indices], y_data[indices]
        split = int(0.75 * len(X_data))
        X_train, X_val = X_data[:split], X_data[split:]
        y_train, y_val = y_data[:split], y_data[split:]
        self.lbl_status.config(text="Building CNN...")
        self.root.update()
        model = Sequential()
        model.add(Input(shape=(30, 30, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, name='logits_layer'))
        model.add(Activation('softmax'))
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.lbl_status.config(text="Training neural network...")
        self.root.update()
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, verbose=0)
        self.cnn_model = model
        val_predictions = model.predict(X_val, verbose=0)
        pred_classes = np.argmax(val_predictions, axis=1)
        y_val_raw_labels = np.argmax(y_val, axis=1)
        CNN_TP = np.sum((y_val_raw_labels == 1) & (pred_classes == 1))
        CNN_FP = np.sum((y_val_raw_labels == 0) & (pred_classes == 1))
        CNN_FN = np.sum((y_val_raw_labels == 1) & (pred_classes == 0))
        CNN_TN = np.sum((y_val_raw_labels == 0) & (pred_classes == 0))
        cnn_accuracy = ((CNN_TP + CNN_TN) / len(y_val_raw_labels)) * 100
        cnn_precision = CNN_TP / (CNN_TP + CNN_FP) if (CNN_TP + CNN_FP) > 0 else 0.0
        cnn_recall = CNN_TP / (CNN_TP + CNN_FN) if (CNN_TP + CNN_FN) > 0 else 0.0
        cnn_f1 = 2 * (cnn_precision * cnn_recall) / (cnn_precision + cnn_recall) if (cnn_precision + cnn_recall) > 0 \
            else 0.0
        print("\n CNN performance evaluation on the test set:")
        print(f"\n Confusion matrix:")
        print(f" [ Actual '+' ]  True Positives (TP): {CNN_TP}  | False Negatives (FN): {CNN_FN}")
        print(f" [ Actual '-' ]  False Positives (FP): {CNN_FP} | True Negatives (TN): {CNN_TN}")
        print(f"\n Accuracy:            {cnn_accuracy:.2f}%")
        print(f" Precision:           {cnn_precision * 100:.2f}%")
        print(f" Recall:              {cnn_recall * 100:.2f}%")
        print(f" F1-Score:            {cnn_f1 * 100:.2f}%")
        messagebox.showinfo("CNN Results",
                            f"CNN trained successfully.\nAccuracy: {cnn_accuracy:.2f}%.")

    def run_cnn(self):
        if not self.current_data_path:
            messagebox.showwarning("Warning", "First select input photos or a folder.")
            return
        if not hasattr(self, 'cnn_model') or self.cnn_model is None:
            if hasattr(self, 'training_data_path') and self.training_data_path:
                self.train_cnn_model(self.training_data_path)
            else:
                messagebox.showwarning("Warning", "The CNN model has not been trained yet, specify the path to the "
                                                  "training photos.")
                return
        if self.mode_var.get() == "single":
            self.process_single_image_cnn(self.current_data_path)
        else:
            self.process_folder_cnn(self.current_data_path)

    def process_single_image_cnn(self, file_path):
        try:
            orig_pil = Image.open(file_path).convert('RGB')
            orig_pil.load()
            img_np = np.array(orig_pil.convert("RGB"))

            results = self.process_pipeline(img_np)
            cropped_np = results.get("result", img_np)
            cnn_input_img = Image.fromarray(cropped_np).resize((30, 30))
            cnn_input_arr = np.array(cnn_input_img, dtype=np.float32) / 255.0
            cnn_input_arr = np.expand_dims(cnn_input_arr, axis=0)
            logits_extractor = tf.keras.Model(inputs=self.cnn_model.input,
                                              outputs=self.cnn_model.get_layer('logits_layer').output)
            raw_logits = logits_extractor.predict(cnn_input_arr, verbose=0)[0]
            probabilities = self.cnn_model.predict(cnn_input_arr, verbose=0)[0]
            predicted_class = np.argmax(probabilities)
            print(f"File: {os.path.basename(file_path)}")
            print(f"Vector before Softmax: {raw_logits}")
            print(f"Probability vector: {probabilities}")
            print(f"CNN prediction: {'Target sign recognized' if predicted_class == 1 else 'Other sign'}")
            VisualizationWindow(self.root, orig_pil, results)
            self.lbl_status.config(text=f"CNN: Class {predicted_class}. Information printed to the console.")

        except Exception as e:
            messagebox.showerror("CNN Error", f"Failed to process the photo:\n{str(e)}")

    def process_folder_cnn(self, folder_path):
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.ppm')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        if not files:
            messagebox.showinfo("Information", "There are no image files in the folder.")
            return
        self.lbl_status.config(text="Splitting data...")
        self.root.update()
        preprocessed_images_list = []
        report_data = []
        for idx, filename in enumerate(files):
            file_path = os.path.join(folder_path, filename)
            try:
                orig_pil = Image.open(file_path).convert('RGB')
                orig_pil.load()
                img_np = np.array(orig_pil)
                results = self.process_pipeline(img_np)
                cropped_np = results.get("result", img_np)
                cropped_pil = Image.fromarray(cropped_np)
                cnn_img = cropped_pil.resize((30, 30))
                preprocessed_images_list.append(np.array(cnn_img, dtype=np.float32) / 255.0)
                report_data.append({
                    "filename": filename,
                    "cropped_pil": cropped_pil,
                    "prediction_text": "Processing..."
                })
            except Exception as e:
                print(f"Skipped file {filename}: {e}")
        if not preprocessed_images_list:
            messagebox.showwarning("Warning", "Failed to prepare any image.")
            return
        self.lbl_status.config(text="Testing the model...")
        self.root.update()
        X_batch = np.array(preprocessed_images_list)
        batch_predictions = self.cnn_model.predict(X_batch, verbose=0)
        predicted_classes = np.argmax(batch_predictions, axis=1)
        for idx, class_idx in enumerate(predicted_classes):
            if class_idx == 1:
                report_data[idx]["prediction_text"] = "Target sign recognized"
            else:
                report_data[idx]["prediction_text"] = "Other sign"
        self.lbl_status.config(text="Folder processing completed. Creating report...")
        self.root.update()
        CNNFolderReportWindow(self.root, report_data)
        self.lbl_status.config(text="CNN report created successfully.")

    def _execute_pipeline_based_on_mode(self):
        if self.mode_var.get() == "single" and self.current_data_path:
            self.process_single_image(self.current_data_path)
        elif self.mode_var.get() == "folder" and self.current_data_path:
            self.process_folder(self.current_data_path)

    def process_pipeline(self, img):
        blank = np.zeros_like(img[:, :, 0])
        gray = blank
        contrast_img = blank
        smoothed_img = blank
        sobel_img = blank
        binary_sobel = blank
        all_filled_masks = blank
        all_opened_masks = blank
        cropped_img = img.copy()
        method_name = "Skipped"
        extracted_hu = [0.0] * 7
        current_processing_target = img.copy()
        if self.var_contrast.get():
            R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)
            mean_brightness = np.mean(gray)
            std_contrast = np.std(gray)
            if mean_brightness < 55:
                method_name = "Logarithmic contrasting"
                max_val = np.max(gray)
                contrast_img = (255 / np.log(1 + max_val) * np.log(1.0 + gray)).astype(
                    np.uint8) if max_val > 0 else gray.copy()
            elif std_contrast < 25:
                method_name = "Linear contrasting"
                g_min, g_max = np.min(gray), np.max(gray)
                contrast_img = (255 / (g_max - g_min) * (gray - g_min)).astype(
                    np.uint8) if g_max > g_min else gray.copy()
            elif std_contrast < 45:
                method_name = "Global equalization"
                hist, _ = np.histogram(gray.flatten(), 256, [0, 256])
                cdf = hist.cumsum()
                cdf_m = np.ma.masked_equal(cdf, 0)
                cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
                contrast_img = np.ma.filled(cdf_m, 0).astype(np.uint8)[gray]
            else:
                method_name = "CLAHE"
                contrast_img = self.clahe(gray)
            current_processing_target = contrast_img
        else:
            method_name = "Skipped"
            gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)
            contrast_img = gray.copy()
            current_processing_target = gray
        if self.var_sobel.get():
            sigma = 1.0
            kernel_size = 5
            x, y = np.mgrid[-(kernel_size // 2):(kernel_size // 2) + 1, -(kernel_size // 2):(kernel_size // 2) + 1]
            gaussian_kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / np.exp(
                -(x ** 2 + y ** 2) / (2 * sigma ** 2)).sum()
            smoothed_img = convolve2d(current_processing_target, gaussian_kernel, mode='same', boundary='symm')
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
            grad_x = convolve2d(smoothed_img, sobel_x, mode='same', boundary='symm')
            grad_y = convolve2d(smoothed_img, sobel_y, mode='same', boundary='symm')
            sobel_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            max_magnitude = np.max(sobel_magnitude)
            sobel_img = (sobel_magnitude / max_magnitude * 255).astype(
                np.uint8) if max_magnitude > 0 else sobel_magnitude.astype(np.uint8)
            morph_target = sobel_img
        else:
            morph_target = current_processing_target
        if self.var_morph.get():
            morph_target = morph_target.astype(np.uint8)
            _, binary_sobel = cv2.threshold(morph_target, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            refined_sobel = cv2.morphologyEx(binary_sobel, cv2.MORPH_CLOSE, kernel_small)
            contours, _ = cv2.findContours(refined_sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_filled_masks = np.zeros(img.shape[:2], dtype=np.uint8)
            all_opened_masks = np.zeros(img.shape[:2], dtype=np.uint8)
            hu_candidates = contours
        else:
            hu_candidates = []
        if self.var_morph.get():
            best_cnt = None
            max_area = 0
            for cnt in hu_candidates:
                area = cv2.contourArea(cnt)
                mask_single = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_single, [cnt], -1, 255, -1)
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
                mask_opened_single = cv2.morphologyEx(mask_single, cv2.MORPH_OPEN, kernel_open)
                cleaned_cnts, _ = cv2.findContours(mask_opened_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cleaned_cnts:
                    continue
                sub_cnt = cleaned_cnts[0]
                hull = cv2.convexHull(sub_cnt)
                x, y, w, h = cv2.boundingRect(hull)
                if not (0.5 <= float(w) / h <= 1.5):
                    continue
                hu_moments = self.calculate_hu_moments(hull, img.shape)
                if area > max_area:
                    max_area = area
                    best_cnt = hull
                    extracted_hu = hu_moments
                if self.var_hu.get():
                    all_filled_masks = cv2.bitwise_or(all_filled_masks, mask_single)
                    all_opened_masks = cv2.bitwise_or(all_opened_masks, mask_opened_single)
            if self.var_hu.get() and best_cnt is not None:
                x, y, w, h = cv2.boundingRect(best_cnt)
                pad = 15
                y1, y2 = max(0, y - pad), min(img.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(img.shape[1], x + w + pad)
                cropped_img = img[y1:y2, x1:x2].copy()
            if best_cnt is None and len(hu_candidates) > 0:
                largest_cnt = max(hu_candidates, key=cv2.contourArea)
                if cv2.contourArea(largest_cnt) > 100:
                    extracted_hu = self.calculate_hu_moments(largest_cnt, img.shape)
        return {
            "gray": gray, "contrast": contrast_img,
            "smooth": smoothed_img.astype(np.uint8) if type(smoothed_img) is np.ndarray else smoothed_img,
            "sobel": sobel_img, "binary": binary_sobel, "filled": all_filled_masks,
            "opened": all_opened_masks, "result": cropped_img, "method_name": method_name,
            "hu_moments": extracted_hu
        }


class FolderReportWindow(tk.Toplevel):
    def __init__(self, parent, report_data):
        super().__init__(parent)
        self.title("Folder processing report — SVM results")
        self.geometry("650x700")
        self.configure(bg="#f5f5f5")
        lbl_title = tk.Label(self, text="SVM classification results for the folder",
                             font=("Arial", 12, "bold"), bg="#f5f5f5", fg="#333")
        lbl_title.pack(pady=10)
        canvas_container = tk.Frame(self, bg="white", bd=2, relief=tk.SUNKEN)
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        canvas = tk.Canvas(canvas_container, bg="white", highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_container, orient="vertical", command=canvas.yview)
        self.table_frame = tk.Frame(canvas, bg="white")
        self.table_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.table_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        header_font = ("Arial", 10, "bold")
        tk.Label(self.table_frame, text="№", font=header_font, bg="#e0e0e0", width=4, relief=tk.RIDGE).grid(
            row=0, column=0,sticky="nsew")
        tk.Label(self.table_frame, text="File name", font=header_font, bg="#e0e0e0", width=20, relief=tk.RIDGE).grid(
            row=0, column=1, sticky="nsew")
        tk.Label(self.table_frame, text="Sign preview", font=header_font, bg="#e0e0e0", width=15, relief=tk.RIDGE).grid(
            row=0, column=2, sticky="nsew")
        tk.Label(self.table_frame, text="SVM result", font=header_font, bg="#e0e0e0", width=25,
                 relief=tk.RIDGE).grid(row=0, column=3, sticky="nsew")
        self.tk_previews = []
        for idx, item in enumerate(report_data):
            row_idx = idx + 1
            bg_color = "#f9f9f9" if idx % 2 == 0 else "white"
            tk.Label(self.table_frame, text=str(row_idx), font=("Arial", 9), bg=bg_color).grid(
                row=row_idx, column=0, sticky="nsew")
            tk.Label(self.table_frame, text=item["filename"], font=("Arial", 9), bg=bg_color, anchor="w", padx=5
                     ).grid(row=row_idx, column=1, sticky="nsew")
            img_pil = item["cropped_pil"].copy()
            img_pil.thumbnail((60, 45))
            tk_preview = ImageTk.PhotoImage(img_pil)
            self.tk_previews.append(tk_preview)
            img_label = tk.Label(self.table_frame, image=tk_preview, bg=bg_color)
            img_label.grid(row=row_idx, column=2, pady=2, padx=2)
            res_text = item["prediction_text"]
            text_color = "green" if "recognized" in res_text else "#b71c1c"
            tk.Label(self.table_frame, text=res_text, font=("Arial", 9, "bold"), fg=text_color, bg=bg_color
                     ).grid(row=row_idx, column=3, sticky="nsew")
        btn_close = tk.Button(self, text="Close report", command=self.destroy, font=("Arial", 10, "bold"), bg="#2196F3",
                              fg="white")
        btn_close.pack(pady=10)


class CNNFolderReportWindow(tk.Toplevel):
    def __init__(self, parent, report_data):
        super().__init__(parent)
        self.title("Folder processing report — CNN results")
        self.geometry("650x700")
        self.configure(bg="#f5f5f5")
        lbl_title = tk.Label(self, text="CNN classification results for the folder",
                             font=("Arial", 12, "bold"), bg="#f5f5f5", fg="#333")
        lbl_title.pack(pady=10)
        canvas_container = tk.Frame(self, bg="white", bd=2, relief=tk.SUNKEN)
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        canvas = tk.Canvas(canvas_container, bg="white", highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_container, orient="vertical", command=canvas.yview)
        self.table_frame = tk.Frame(canvas, bg="white")
        self.table_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.table_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        header_font = ("Arial", 10, "bold")
        tk.Label(self.table_frame, text="№", font=header_font, bg="#e0e0e0", width=4, relief=tk.RIDGE)\
            .grid(row=0, column=0, sticky="nsew")
        tk.Label(self.table_frame, text="File name", font=header_font, bg="#e0e0e0", width=20, relief=tk.RIDGE)\
            .grid(row=0, column=1, sticky="nsew")
        tk.Label(self.table_frame, text="Sign preview", font=header_font, bg="#e0e0e0", width=15, relief=tk.RIDGE)\
            .grid(row=0, column=2, sticky="nsew")
        tk.Label(self.table_frame, text="CNN result", font=header_font, bg="#e0e0e0", width=25, relief=tk.RIDGE)\
            .grid(row=0, column=3, sticky="nsew")
        self.tk_previews = []
        for idx, item in enumerate(report_data):
            row_idx = idx + 1
            bg_color = "#f9f9f9" if idx % 2 == 0 else "white"
            tk.Label(self.table_frame, text=str(row_idx), font=("Arial", 9), bg=bg_color, relief=tk.RIDGE)\
                .grid(row=row_idx, column=0, sticky="nsew")
            tk.Label(self.table_frame, text=item["filename"], font=("Arial", 9), bg=bg_color, anchor="w",
                     padx=5, relief=tk.RIDGE).grid(row=row_idx, column=1, sticky="nsew")
            img_pil = item["cropped_pil"].copy()
            img_pil.thumbnail((60, 45))
            tk_preview = ImageTk.PhotoImage(img_pil)
            self.tk_previews.append(tk_preview)
            img_label = tk.Label(self.table_frame, image=tk_preview, bg=bg_color, relief=tk.RIDGE)
            img_label.grid(row=row_idx, column=2, pady=2, padx=2)
            res_text = item["prediction_text"]
            text_color = "green" if "recognized" in res_text.lower() else "#b71c1c"
            tk.Label(self.table_frame, text=res_text, font=("Arial", 9, "bold"), fg=text_color, bg=bg_color,
                     relief=tk.RIDGE).grid(row=row_idx, column=3, sticky="nsew")
        btn_close = tk.Button(self, text="Close report", command=self.destroy, font=("Arial", 10, "bold"),
                              bg="#E91E63", fg="white")
        btn_close.pack(pady=10)


class VisualizationWindow(tk.Toplevel):
    def __init__(self, parent, orig_pil, res):
        super().__init__(parent)
        self.title("Intermediate image preparation stages")
        self.geometry("1250x950")
        self.configure(bg="#f5f5f5")
        grid_frame = tk.Frame(self, bg="#f5f5f5")
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        for i in range(3):
            grid_frame.columnconfigure(i, weight=1)
            grid_frame.rowconfigure(i, weight=1)
        images = [
            orig_pil,
            self._to_pil(res["gray"], " "),
            self._to_pil(res["contrast"], " "),
            self._to_pil(res["smooth"], " "),
            self._to_pil(res["sobel"], " "),
            self._to_pil(res["binary"], " "),
            self._to_pil(res["filled"], " "),
            self._to_pil(res["opened"], " "),
            self._to_pil(res["result"], " ")
        ]
        titles = [
            "Original image", "1. Grayscale (Luma)", f"2. {res['method_name']}",
            "3. Gaussian smoothing", "4. Edge detection (Sobel)", "5. Binarization (Otsu)",
            "6. Filling", "7. Trimming", "8. Sign"
        ]
        self.tk_images = []
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        for i, img in enumerate(images):
            img.thumbnail((300, 220))
            tk_img = ImageTk.PhotoImage(img)
            self.tk_images.append(tk_img)
            frame = tk.Frame(grid_frame, bg="white", bd=2, relief=tk.GROOVE)
            frame.grid(row=positions[i][0], column=positions[i][1], padx=8, pady=8, sticky="nsew")
            tk.Label(frame, text=titles[i], font=("Arial", 10, "bold"), bg="white").pack(side=tk.TOP, pady=3)
            tk.Label(frame, image=tk_img, bg="#e0e0e0").pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _to_pil(self, arr, placeholder_text):
        if np.sum(arr) == 0:
            img = Image.new('RGB', (300, 220), color=(220, 220, 220))
            d = ImageDraw.Draw(img)
            d.text((100, 100), placeholder_text, fill=(100, 100, 100))
            return img
        return Image.fromarray(arr)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainInterface(root)
    root.mainloop()