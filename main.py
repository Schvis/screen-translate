from PIL import Image, ImageDraw
import requests
import easyocr
import pyautogui
import keyboard 
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QComboBox, QPushButton, QWidget, QColorDialog, QSpinBox, QCheckBox
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt
import sys
import torch
import tkinter as tk
from tkinter import simpledialog, Canvas
import logging
import json
import time
from skimage.metrics import structural_similarity as compare_ssim
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

reader = None

def get_easyocr_reader(languages):
    try:
        return easyocr.Reader(languages, gpu=torch.cuda.is_available())
    except ValueError as e:
        logging.error(f"Error initializing EasyOCR reader: {e}")
        raise

def load_api_key():
    try:
        with open("cfg.json", "r") as file:
            config = json.load(file)
            return config.get("deepl_api_key", None)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading API key from cfg.json: {e}")
        return None

def deepl_translate_request(data):
    api_key = load_api_key()
    if not api_key:
        logging.error("DeepL API key not found in cfg.json.")
        return None

    url = "https://api-free.deepl.com/v2/translate"
    headers = {
        "Authorization": f"DeepL-Auth-Key {api_key}"
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        try:
            return response.json()["translations"]
        except (ValueError, KeyError) as e:
            logging.error(f"Error parsing translation response: {e}")
            return None
    else:
        logging.error(f"Translation request failed: {response.status_code} - {response.text}")
        return None

def google_translate_request(text, source_lang, target_lang):
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={requests.utils.quote(text)}"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                return response.json()[0][0][0]
            except (ValueError, KeyError, IndexError) as e:
                logging.error(f"Error parsing Google Translate response: {e}")
                return None
        else:
            logging.error(f"Google Translate request failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logging.error(f"Error in Google Translate request: {e}")
        return None

def translate_request(data, use_google=False):
    if use_google:
        return google_translate_request(data["text"], data["source_lang"], data["target_lang"])
    else:
        return deepl_translate_request(data)

def translate_text(text, source_lang, target_lang):
    data = {
        "text": text,
        "source_lang": source_lang.lower(),
        "target_lang": target_lang.lower()
    }
    translations = translate_request(data, use_google=not load_api_key())
    translated_text = translations if isinstance(translations, str) else translations[0]["text"] if translations else "Error translating text"

    app_instance = QApplication.instance().activeWindow()
    if isinstance(app_instance, TranslatorApp):
        app_instance.log_extracted_text(text)

    return translated_text

def translate_texts(texts, source_lang, target_lang):
    if not load_api_key():
        translations = [google_translate_request(text, source_lang.lower(), target_lang.lower()) or "Error translating text" for text in texts]
    else:
        data = {
            "text": texts,
            "source_lang": source_lang.upper(),
            "target_lang": target_lang.upper()
        }
        translations = deepl_translate_request(data)
        translations = [item["text"] for item in translations] if translations else ["Error translating text"] * len(texts)

    app_instance = QApplication.instance().activeWindow()
    if isinstance(app_instance, TranslatorApp):
        for text in texts:
            app_instance.log_extracted_text(text)

    return translations

def update_overlay_position(overlay, region):
    try:
        print("Updating overlay position...")
        x, y, width, height = region
        overlay.setGeometry(x, y, width, height)
    except Exception as e:
        print(f"Error updating overlay position: {e}")

class OverlayWindow(QMainWindow):
    def __init__(self, screenshot_np, results, translations, on_key_callback, region, text_color, font_size):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(*region)
        self.setCursor(Qt.ArrowCursor)
        self.screenshot_np = screenshot_np
        self.results = results
        self.translations = translations
        self.on_key_callback = on_key_callback
        self.is_processing_click = False
        self.text_color = text_color
        self.font_size = font_size
        self.setMouseTracking(True)

    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            for (bbox, translation) in zip(self.results, self.translations):
                top_left, _, bottom_right, _ = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                width = bottom_right[0] - top_left[0]
                height = bottom_right[1] - top_left[1]

                is_vertical = height > width

                font_size = self.font_size
                painter.setFont(QFont("Arial", font_size, QFont.Bold))
                painter.setPen(self.text_color)

                text_width = painter.fontMetrics().width(translation)
                text_height = painter.fontMetrics().height()

                if is_vertical:
                    rect_width = font_size
                    rect_height = max(height, text_height * len(translation))
                else:
                    rect_width = max(width, text_width + 10)
                    rect_height = font_size + 10

                rect_x = top_left[0]
                rect_y = top_left[1]

                background_color = QColor(0, 0, 0, 150)
                painter.setBrush(background_color)
                painter.setPen(Qt.NoPen)
                painter.drawRect(rect_x, rect_y, rect_width, rect_height)

                painter.setPen(self.text_color)
                if is_vertical:
                    x = rect_x + rect_width // 2 - font_size // 2
                    y = rect_y
                    for char in translation:
                        painter.drawText(x, y, char)
                        y += font_size
                else:
                    text_x = rect_x + (rect_width - text_width) // 2
                    text_y = rect_y + rect_height - 5
                    painter.drawText(text_x, text_y, translation)
        except Exception as e:
            logging.error(f"Error during paint event: {e}")
            event.ignore()

def select_screen_region_with_mouse():
    print("Please select a part of the screen using the mouse.")
    region = {}

    def on_mouse_press(event):
        region['x1'] = event.x_root
        region['y1'] = event.y_root

    def on_mouse_release(event):
        region['x2'] = event.x_root
        region['y2'] = event.y_root
        root.quit()

    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-alpha", 0.3)
    root.configure(bg="black")

    canvas = Canvas(root, cursor="cross", bg="black")
    canvas.pack(fill="both", expand=True)

    canvas.bind("<ButtonPress-1>", on_mouse_press)
    canvas.bind("<ButtonRelease-1>", on_mouse_release)

    root.mainloop()
    root.destroy()

    if 'x1' in region and 'y1' in region and 'x2' in region and 'y2' in region:
        x = min(region['x1'], region['x2'])
        y = min(region['y1'], region['y2'])
        width = abs(region['x2'] - region['x1'])
        height = abs(region['y2'] - region['y1'])
        print(f"Selected region: (X: {x}, Y: {y}, Width: {width}, Height: {height})")
        return (x, y, width, height)
    else:
        print("No region selected. Exiting.")
        return None

def capture_screen_region(region):
    try:
        print(f"Capturing screen region: {region}...")
        screenshot = pyautogui.screenshot(region=region)
        return np.array(screenshot)
    except Exception as e:
        print(f"Error capturing screen region: {e}")
        return None

def save_detected_text_image(screenshot_np, results, filename="detected_text.png"):
    try:
        image = Image.fromarray(screenshot_np)
        draw = ImageDraw.Draw(image)

        for bbox, text, _ in results:
            top_left, _, bottom_right, _ = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            draw.rectangle([top_left, bottom_right], outline="red", width=2)
            draw.text(top_left, text, fill="red")

        image.save(filename)
        print(f"Detected text image saved as {filename}")
    except Exception as e:
        print(f"Error saving detected text image: {e}")

def capture_screen_with_text_detection_and_overlay(on_key_callback, region, source_lang, target_lang, text_color, font_size, ocr_params=None):
    try:
        print("Capturing selected screen region...")
        screenshot_np = capture_screen_region(region)
        if screenshot_np is None:
            print("Failed to capture screen region. Exiting.")
            return None

        print("Detecting text regions...")
        if ocr_params is None:
            ocr_params = {
                "detail": 1,
                "paragraph": False,
                "contrast_ths": 0.0,
                "adjust_contrast": 0.0
            }
        results = reader.readtext(screenshot_np, **ocr_params)
        save_detected_text_image(screenshot_np, results)

        print("Grouping text regions into phrases...")
        grouped_results = []
        results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
        current_phrase = []
        current_y = None
        last_x = None

        for result in results:
            if len(result) == 3:
                bbox, text, _ = result
            elif len(result) == 2:
                bbox, text = result
            else:
                logging.error(f"Unexpected result format: {result}")
                continue

            top_left, _, _, _ = bbox
            x, y = top_left

            if current_y is None or (last_x is None):
                current_phrase.append(text)
                current_y = y
                last_x = x
            else:
                grouped_results.append(" ".join(current_phrase))
                current_phrase = [text]
                current_y = y
                last_x = x

        if current_phrase:
            grouped_results.append(" ".join(current_phrase))

        print("Text detected. Proceeding with batch translation...")
        translations = translate_texts(grouped_results, source_lang, target_lang)

        print("Displaying overlay...")
        app = QApplication.instance()
        if (app is None):
            app = QApplication(sys.argv)

        overlay = OverlayWindow(screenshot_np, [r[0] for r in results if len(r) >= 2], translations, on_key_callback, region, text_color, font_size)
        overlay.show()

        app.processEvents()

        return overlay
    except Exception as e:
        print(f"Error capturing screen with text detection and overlay: {e}")
        return None

def load_config():
    try:
        with open("cfg.json", "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading configuration from cfg.json: {e}")
        return {}

def save_config(config):
    try:
        with open("cfg.json", "w") as file:
            json.dump(config, file, indent=4)
    except Exception as e:
        logging.error(f"Error saving configuration to cfg.json: {e}")

def ensure_config():
    config = load_config()
    if "deepl_api_key" not in config or not config["deepl_api_key"]:
        print("DeepL API key not found. You can use Google Translate as a fallback.")
        root = tk.Tk()
        root.withdraw()
        api_key = simpledialog.askstring("API Key", "Enter your DeepL API key (leave blank to use Google Translate):", parent=root)
        root.destroy()
        if api_key:
            config["deepl_api_key"] = api_key
        else:
            print("No API key entered. Defaulting to Google Translate.")
        save_config(config)
    if "keybinds" not in config:
        config["keybinds"] = {"Start Scanning": "H", "Stop Scanning": "Q", "Request Translation": "T"}
    config["keybinds"].setdefault("Start Monitoring", "M")
    if "font_size" not in config:
        config["font_size"] = 16
    if "text_color" not in config:
        config["text_color"] = "#FFFFFF"
    save_config(config)

def save_selected_region(region):
    config = load_config()
    config["last_selected_region"] = region
    save_config(config)

def load_selected_region():
    config = load_config()
    return config.get("last_selected_region", None)

class TranslatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Translator App")
        self.setGeometry(100, 100, 400, 450)

        self.config = load_config()

        self.source_lang_combo = QComboBox(self)
        self.target_lang_combo = QComboBox(self)

        languages = {
            "Japanese": "ja",
            "English": "en",
            "Russian": "ru"
        }

        self.source_lang_combo.addItems(languages.keys())
        self.target_lang_combo.addItems(languages.keys())

        last_source_lang = self.config.get("last_source_lang", "Japanese")
        last_target_lang = self.config.get("last_target_lang", "English")
        self.source_lang_combo.setCurrentText(last_source_lang)
        self.target_lang_combo.setCurrentText(last_target_lang)

        self.language_mapping = languages

        self.select_area_button = QPushButton("Select Area", self)

        self.keybinds = self.config.get("keybinds", {"Start Scanning": "H", "Stop Scanning": "Q", "Request Translation": "T", "Start Monitoring": "M"})
        self.keybind_labels = {}

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Source Language:"))
        layout.addWidget(self.source_lang_combo)
        layout.addWidget(QLabel("Target Language:"))
        layout.addWidget(self.target_lang_combo)
        layout.addWidget(self.select_area_button)

        layout.addWidget(QLabel("Keybinds:"))
        for action, key in self.keybinds.items():
            keybind_layout = QVBoxLayout()
            label = QLabel(f"{action}: {key}")
            self.keybind_labels[action] = label
            keybind_layout.addWidget(label)

            change_button = QPushButton(f"Change '{action}' Keybind", self)
            change_button.clicked.connect(lambda _, a=action: self.change_keybind(a))
            keybind_layout.addWidget(change_button)

            layout.addLayout(keybind_layout)

        self.monitoring_keybind_label = QLabel(f"Current Monitoring Keybind: {self.keybinds['Start Monitoring']}", self)
        layout.addWidget(self.monitoring_keybind_label)

        self.change_monitoring_keybind_button = QPushButton("Change Monitoring Keybind", self)
        self.change_monitoring_keybind_button.clicked.connect(lambda: self.change_keybind("Start Monitoring"))
        layout.addWidget(self.change_monitoring_keybind_button)

        self.text_color_button = QPushButton("Select Text Color", self)
        self.text_color_button.clicked.connect(self.select_text_color)
        self.text_color = QColor(self.config.get("text_color", "#FFFFFF"))

        layout.addWidget(self.text_color_button)

        self.font_size_spinbox = QSpinBox(self)
        self.font_size_spinbox.setRange(8, 72)
        self.font_size_spinbox.setValue(self.config.get("font_size", 16))
        self.font_size_spinbox.setPrefix("Font Size: ")

        layout.addWidget(self.font_size_spinbox)

        self.contrast_ths_spinbox = QSpinBox(self)
        self.contrast_ths_spinbox.setRange(0, 100)
        self.contrast_ths_spinbox.setValue(70)
        self.contrast_ths_spinbox.setPrefix("Contrast Threshold (%): ")

        self.paragraph_checkbox = QCheckBox("Enable Paragraph Detection", self)

        layout.addWidget(self.contrast_ths_spinbox)
        layout.addWidget(self.paragraph_checkbox)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.select_area_button.clicked.connect(self.select_area)

        self.selected_region = load_selected_region()
        if self.selected_region:
            print(f"Loaded last selected region: {self.selected_region}")

        self.overlay = None
        self.scanning = False
        self.monitoring = False
        self.monitoring_thread = None
        self.translation_in_progress = False
        self.current_ocr_languages = None

    def select_area(self):
        self.selected_region = select_screen_region_with_mouse()
        if self.selected_region:
            print(f"Selected region: {self.selected_region}")
            save_selected_region(self.selected_region)

    def change_keybind(self, action):
        root = tk.Tk()
        root.withdraw()

        new_key = simpledialog.askstring("Change Keybind", f"Enter new key for '{action}':", parent=root)
        if new_key:
            self.keybinds[action] = new_key.upper()
            self.keybind_labels[action].setText(f"{action}: {new_key.upper()}")
            print(f"Keybind for '{action}' changed to '{new_key.upper()}'")
            self.save_settings()

            # Update the monitoring keybind label if the action is "Start Monitoring"
            if action == "Start Monitoring":
                self.monitoring_keybind_label.setText(f"Current Monitoring Keybind: {new_key.upper()}")
        else:
            print("No key entered. Keybind not changed.")

        root.destroy()

    def select_text_color(self):
        color = QColorDialog.getColor(initial=self.text_color, parent=self, title="Select Text Color")
        if color.isValid():
            self.text_color = color
            print(f"Selected text color: {self.text_color.name()}")
            self.save_settings()

    def save_settings(self):
        self.config["keybinds"] = self.keybinds
        self.config["font_size"] = self.font_size_spinbox.value()
        self.config["text_color"] = self.text_color.name()
        self.config["last_source_lang"] = self.source_lang_combo.currentText()
        self.config["last_target_lang"] = self.target_lang_combo.currentText()
        save_config(self.config)

    def start_translation(self):
        if not self.selected_region:
            print("No region selected. Please select an area first.")
            return

        source_lang = self.language_mapping[self.source_lang_combo.currentText()]
        target_lang = self.language_mapping[self.target_lang_combo.currentText()]

        try:
            # Ensure the OCR reader is initialized
            global reader
            if reader is None:
                logging.info("Initializing EasyOCR reader...")
                reader = get_easyocr_reader([source_lang, target_lang])
        except ValueError:
            print(f"Error: The selected languages ({source_lang}, {target_lang}) are not compatible for OCR.")
            return

        font_size = self.font_size_spinbox.value()
        contrast_ths = self.contrast_ths_spinbox.value() / 100
        paragraph = self.paragraph_checkbox.isChecked()

        ocr_params = {"contrast_ths": contrast_ths, "paragraph": paragraph}

        try:
            logging.info("Starting translation process...")
            self.overlay = capture_screen_with_text_detection_and_overlay(
                self.request_new_translation, self.selected_region, source_lang, target_lang, self.text_color, font_size, ocr_params
            )
            if self.overlay:
                self.overlay.show()
        except Exception as e:
            logging.error(f"Error during translation process: {e}")

    def request_new_translation(self):
        logging.info("Requesting new translation...")
        if self.overlay:
            self.overlay.close()
        self.start_translation()

    def log_extracted_text(self, text):
        """Append the extracted text to a log file."""
        try:
            with open("extracted_text_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {text}\n")
        except Exception as e:
            logging.error(f"Error logging extracted text: {e}")

    def monitor_changes(self):
        if not self.selected_region:
            print("No region selected. Please select an area first.")
            return

        self.monitoring = True
        previous_capture = None
        previous_text = None

        while self.monitoring:
            try:
                current_capture = capture_screen_region(self.selected_region)
                if current_capture is None:
                    print("Failed to capture screen region. Exiting monitoring.")
                    break

                if self.overlay:
                    overlay_region = self.overlay.geometry()
                    x, y, width, height = overlay_region.x(), overlay_region.y(), overlay_region.width(), overlay_region.height()
                    current_capture[y:y + height, x:x + width] = 0

                source_lang = self.language_mapping[self.source_lang_combo.currentText()]
                target_lang = self.language_mapping[self.target_lang_combo.currentText()]

                global reader
                if self.current_ocr_languages != [source_lang, target_lang]:
                    logging.info(f"Reinitializing EasyOCR reader with languages: {source_lang}, {target_lang}")
                    reader = get_easyocr_reader([source_lang])
                    self.current_ocr_languages = [source_lang, target_lang]

                gray_curr = cv2.cvtColor(current_capture, cv2.COLOR_BGR2GRAY)
                ocr_params = {
                    "detail": 1,
                    "paragraph": True,
                    "contrast_ths": 0.5,
                    "adjust_contrast": 0.7
                }
                current_results = reader.readtext(gray_curr, **ocr_params)

                current_text = " ".join([result[1] for result in current_results]).strip()

                self.log_extracted_text(current_text)

                source_lang = self.language_mapping[self.source_lang_combo.currentText()]

                if previous_capture is not None:
                    gray_prev = cv2.cvtColor(previous_capture, cv2.COLOR_BGR2GRAY)
                    score, _ = compare_ssim(gray_prev, gray_curr, full=True)

                    text_similarity = SequenceMatcher(None, previous_text, current_text).ratio()

                    if text_similarity > 0.65:
                        print("Text similarity high. Ignoring background changes.")
                    elif score < 0.95 and not self.translation_in_progress:
                        print("Significant change detected in the selected region. Starting translation...")
                        self.translation_in_progress = True

                        if self.overlay:
                            logging.info("Closing old overlay before starting new translation.")
                            self.overlay.close()
                            self.overlay = None

                        self.start_translation()
                        self.translation_in_progress = False
                    else:
                        print("Minor changes detected. Ignoring...")

                previous_capture = current_capture
                previous_text = current_text

                if not self.monitoring:
                    print("Monitoring stopped.")
                    break

                time.sleep(0.5)
            except Exception as e:
                logging.error(f"Error during monitoring: {e}")
                self.translation_in_progress = False

    def start_monitoring(self):
        if not self.selected_region:
            print("No region selected. Please select an area first.")
            return

        print("Starting real-time monitoring...")
        self.monitoring = True
        self.monitoring_thread = threading.Thread(target=self.monitor_changes, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        if self.monitoring:
            print("Stopping monitoring...")
            self.monitoring = False
            if self.monitoring_thread.is_alive():
                self.monitoring_thread.join()

    def handle_keybinds(self):
        try:
            logging.info("Handling keybinds...")
            key_states = {
                "Start Scanning": False,
                "Stop Scanning": False,
                "Request Translation": False,
                "Start Monitoring": False
            }

            while True:
                if keyboard.is_pressed(self.keybinds["Start Scanning"]):
                    if not key_states["Start Scanning"]:
                        logging.info("Start Scanning key pressed.")
                        self.scanning = True
                        self.start_translation()
                        key_states["Start Scanning"] = True
                else:
                    key_states["Start Scanning"] = False

                if keyboard.is_pressed(self.keybinds["Stop Scanning"]):
                    if not key_states["Stop Scanning"]:
                        logging.info("Stop Scanning key pressed.")
                        self.scanning = False
                        self.stop_monitoring()
                        if self.overlay:
                            logging.info("Closing overlay.")
                            self.overlay.close()
                            self.overlay = None
                        key_states["Stop Scanning"] = True
                else:
                    key_states["Stop Scanning"] = False

                if keyboard.is_pressed(self.keybinds["Request Translation"]):
                    if not key_states["Request Translation"] and self.overlay:
                        logging.info("Requesting new translation...")
                        self.request_new_translation()
                        key_states["Request Translation"] = True
                else:
                    key_states["Request Translation"] = False

                if keyboard.is_pressed(self.keybinds.get("Start Monitoring", "M")):
                    if not key_states["Start Monitoring"]:
                        logging.info("Start Monitoring key pressed.")
                        self.start_monitoring()
                        key_states["Start Monitoring"] = True
                else:
                    key_states["Start Monitoring"] = False

                time.sleep(0.1)
        except Exception as e:
            print(f"Error in keybind handling: {e}")

    def closeEvent(self, event):
        print("Closing application...")
        self.scanning = False
        self.stop_monitoring()
        if self.overlay:
            self.overlay.close()
            self.overlay = None
        self.save_settings()
        QApplication.quit()
        event.accept()

    def reset_ocr(self):
        """Reset the OCR reader to reinitialize it."""
        global reader
        reader = None
        print("OCR reader has been reset. It will reinitialize on the next use.")

if __name__ == "__main__":
    ensure_config()
    app = QApplication(sys.argv)
    translator_app = TranslatorApp()
    translator_app.show()

    import threading
    keybind_thread = threading.Thread(target=translator_app.handle_keybinds, daemon=True)
    keybind_thread.start()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print("Application exited.")
