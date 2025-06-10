import os
import sys
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import fitz  # PyMuPDF
from PIL import Image, ImageTk

sys.path.append('./tools')

import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="int32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect[0], rect[1], rect[2], rect[3]

def per_tran(image, rect):
    tl, tr, br, bl = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype='float32')
    rect = np.array(rect, dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def get_standard_table_image(gray, table):
    sorted_rect = get_sorted_rect(np.array(table))
    gray_z = per_tran(gray, sorted_rect)
    binary_z = cv2.adaptiveThreshold(~gray_z, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, -5)
    return gray_z, binary_z


def get_sorted_rect(rect):
    try:
        mid_x = (max(p[1] for p in rect) - min(p[1] for p in rect)) * 0.5 + min(p[1] for p in rect)
        left = [p for p in rect if p[1] < mid_x]
        left.sort(key=lambda x: (x[0], x[1]))
        right = [p for p in rect if p[1] > mid_x]
        right.sort(key=lambda x: (x[0], x[1]))
        sorted_rect = left[0], left[1], right[1], right[0]
    except Exception:
        rect = np.asarray(rect)
        s = rect.sum(axis=1)
        diff = np.diff(rect, axis=1)
        sorted_rect = (
            rect[np.argmin(s)],
            rect[np.argmin(diff)],
            rect[np.argmax(s)],
            rect[np.argmax(diff)],
        )
    return sorted_rect


def get_y_sorted_contours(contours):
    boxes = [cv2.boxPoints(cv2.minAreaRect(c)).astype(int) for c in contours]
    boxes.sort(key=lambda b: b[:, 1].min())
    return boxes


def detect_tables(gray, max_box_ratio=10, min_table_area=0):
    gray_copy = gray.copy()
    gray_copy[:50, :] = 255
    gray_copy[-50:, :] = 255
    gray_copy[:, :50] = 255
    gray_copy[:, -50:] = 255

    canny = cv2.Canny(gray_copy, 200, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    canny = cv2.dilate(canny, kernel)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not min_table_area:
        min_table_area = gray.shape[0] * gray.shape[1] * 0.01

    candidates = [c for c in contours if cv2.contourArea(c) > min_table_area]
    candidates = sorted(candidates, key=cv2.contourArea, reverse=True)
    candidates = get_y_sorted_contours(candidates)

    tables = []
    for cnt in candidates:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        sorted_box = get_sorted_rect(box)
        ratio = (sorted_box[2][0] - sorted_box[3][0]) / (sorted_box[2][1] - sorted_box[1][1])
        if ratio > max_box_ratio or ratio < 1 / max_box_ratio:
            continue
        result = [sorted_box[2].tolist(), sorted_box[3].tolist(), sorted_box[0].tolist(), sorted_box[1].tolist()]
        tables.append(result)

    return tables


class TableCropperGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Table Cropper")

        self.pages = []
        self.current_page = 0
        self.display_image = None
        self.output_dir = os.path.join(os.getcwd(), "table_image")

        tk.Label(master, text="max_box_ratio").grid(row=0, column=0, sticky="e")
        self.box_ratio_var = tk.DoubleVar(value=10.0)
        tk.Entry(master, textvariable=self.box_ratio_var).grid(row=0, column=1, padx=5, pady=2)

        tk.Label(master, text="min_table_area").grid(row=1, column=0, sticky="e")
        self.min_area_var = tk.DoubleVar(value=0)
        tk.Entry(master, textvariable=self.min_area_var).grid(row=1, column=1, padx=5, pady=2)

        tk.Label(master, text="Output Dir").grid(row=2, column=0, sticky="e")
        self.save_dir_var = tk.StringVar(value=self.output_dir)
        tk.Entry(master, textvariable=self.save_dir_var, width=30).grid(row=2, column=1, padx=5, pady=2)
        tk.Button(master, text="Select", command=self.select_dir).grid(row=2, column=2, padx=2)

        tk.Button(master, text="Open File", command=self.open_file).grid(row=3, column=0, columnspan=3, sticky="we", pady=5)

        nav_frame = tk.Frame(master)
        nav_frame.grid(row=4, column=0, columnspan=3, pady=2)
        tk.Button(nav_frame, text="Prev Page", command=self.prev_page).pack(side="left", padx=5)
        tk.Button(nav_frame, text="Next Page", command=self.next_page).pack(side="left", padx=5)
        tk.Button(nav_frame, text="Detect && Save", command=self.detect_tables_action).pack(side="left", padx=5)

        self.canvas = tk.Canvas(master, width=600, height=400, bg="gray")
        self.canvas.grid(row=5, column=0, columnspan=3, padx=5, pady=5)

    def select_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.save_dir_var.set(path)

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("Image/PDF", "*.png *.jpg *.jpeg *.bmp *.pdf")])
        if not path:
            return
        self.pages = []
        self.current_page = 0
        ext = os.path.splitext(path)[1].lower()
        if ext == '.pdf':
            doc = fitz.open(path)
            for page in doc:
                pix = page.get_pixmap()
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.pages.append(img)
        else:
            img = cv2.imread(path)
            if img is None:
                messagebox.showerror("Error", "Failed to load image")
                return
            self.pages.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not self.pages:
            messagebox.showerror("Error", "No pages loaded")
            return
        self.show_page()

    def show_page(self):
        img = self.pages[self.current_page]
        self.show_image(img)

    def prev_page(self):
        if not self.pages:
            return
        self.current_page = max(0, self.current_page - 1)
        self.show_page()

    def next_page(self):
        if not self.pages:
            return
        self.current_page = min(len(self.pages) - 1, self.current_page + 1)
        self.show_page()

    def show_image(self, img, boxes=None):
        draw_img = img.copy()
        if boxes:
            for b in boxes:
                pts = np.array(b).reshape(-1, 2)
                cv2.polylines(draw_img, [pts], True, (255, 0, 0), 2)
        img_pil = Image.fromarray(draw_img)
        img_pil.thumbnail((600, 400))
        self.display_image = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.display_image)

    def detect_tables_action(self):
        if not self.pages:
            messagebox.showinfo("Info", "Please open a file first")
            return
        img = self.pages[self.current_page]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        tables = detect_tables(gray, max_box_ratio=self.box_ratio_var.get(), min_table_area=self.min_area_var.get())
        if not tables:
            messagebox.showinfo("Result", "No tables found")
            self.show_image(img)
            return
        save_dir = self.save_dir_var.get()
        os.makedirs(save_dir, exist_ok=True)
        for i, t in enumerate(tables):
            table_img, _ = get_standard_table_image(gray, t)
            save_name = f"{self.current_page+1}-{i}.jpg"
            cv2.imwrite(os.path.join(save_dir, save_name), table_img)
        self.show_image(img, boxes=tables)
        messagebox.showinfo("Result", f"Saved {len(tables)} table image(s) to {save_dir}")


if __name__ == '__main__':
    root = tk.Tk()
    TableCropperGUI(root)
    root.mainloop()
