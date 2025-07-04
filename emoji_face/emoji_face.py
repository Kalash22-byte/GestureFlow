import cv2
import numpy as np
import random
import os
import time
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_overlay.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FaceOverlayApp:
    def __init__(self):
        self.OVERLAY_UPDATE_DELAY = 3.0
        self.MIN_FACE_SIZE = 100
        self.MAX_FACE_SIZE = 500
        self.HELP_TEXT = [
            "P: Toggle face visibility",
            "R: Random overlays",
            "A: Same overlay for all faces",
            "C: Cycle overlays (forward)",
            "V: Cycle overlays (backward)",
            "S: Save current frame",
            "H: Help (toggle)",
            "I: Show overlay name",
            "+/-: Adjust overlay size",
            "Q: Quit"
        ]
        self.show_real_face = False
        self.show_help = True
        self.same_overlay_mode = False
        self.cycle_index = 0
        self.current_overlays: Dict[int, Tuple[str, np.ndarray]] = {}
        self.last_update_time = time.time()
        self.selected_overlay: Optional[Tuple[str, np.ndarray]] = None
        self.overlay_scale = 1.0
        self.performance_stats = {
            'frame_count': 0,
            'start_time': time.time(),
            'face_detections': 0
        }
        self._load_face_detector()
        self._load_overlays()
        self._setup_camera()

    def _load_face_detector(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            raise FileNotFoundError("Haar cascade file not found")

    def _load_overlays(self) -> None:
        image_folder = 'C:/my-project/emoji_face'
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.png')]
        self.overlays: List[Tuple[str, np.ndarray]] = []
        for file in image_files:
            img = cv2.imread(os.path.join(image_folder, file), cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] == 4:
                self.overlays.append((file, img))
        if not self.overlays:
            raise ValueError("No valid overlay images with alpha channel")
        self.selected_overlay = random.choice(self.overlays)

    def _setup_camera(self) -> None:
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video capture device")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def _blend_overlay(self, frame: np.ndarray, overlay_img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        w, h = int(w * self.overlay_scale), int(h * self.overlay_scale)
        x, y = x - (w - int(w / self.overlay_scale)) // 2, y - (h - int(h / self.overlay_scale)) // 2
        h_frame, w_frame = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, w_frame - x), min(h, h_frame - y)
        if w <= 0 or h <= 0:
            return frame
        overlay_img = cv2.resize(overlay_img, (w, h))
        alpha = overlay_img[:, :, 3] / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])
        overlay_rgb = overlay_img[:, :, :3]
        roi = frame[y:y+h, x:x+w]
        blended = (1.0 - alpha) * roi + alpha * overlay_rgb
        frame[y:y+h, x:x+w] = blended.astype(np.uint8)
        return frame

    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.MIN_FACE_SIZE, self.MIN_FACE_SIZE),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        valid_faces = []
        for (x, y, w, h) in faces:
            if self.MIN_FACE_SIZE <= w <= self.MAX_FACE_SIZE and self.MIN_FACE_SIZE <= h <= self.MAX_FACE_SIZE:
                valid_faces.append((x, y, w, h))
                self.performance_stats['face_detections'] += 1
        return valid_faces

    def _update_overlay_selection(self) -> None:
        current_time = time.time()
        if len(self.current_overlays) != len(self.faces):
            self.current_overlays = {}
        for i, (x, y, w, h) in enumerate(self.faces):
            if self.same_overlay_mode:
                file_name, overlay_img = self.selected_overlay
            elif i not in self.current_overlays or current_time - self.last_update_time > self.OVERLAY_UPDATE_DELAY:
                self.current_overlays[i] = random.choice(self.overlays)
                file_name, overlay_img = self.current_overlays[i]
                self.last_update_time = current_time
            else:
                file_name, overlay_img = self.current_overlays[i]

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.faces = self._detect_faces(frame)
        self._update_overlay_selection()
        for i, (x, y, w, h) in enumerate(self.faces):
            if self.same_overlay_mode:
                file_name, overlay_img = self.selected_overlay
            else:
                file_name, overlay_img = self.current_overlays.get(i, self.selected_overlay)
            if not self.show_real_face:
                frame = self._blend_overlay(frame, overlay_img, x, y, w, h)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if file_name and not self.show_real_face:
                img_name = os.path.splitext(file_name)[0].capitalize()
                cv2.putText(frame, img_name, (x, y - 30 if y > 30 else y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame

    def _display_ui(self, frame: np.ndarray) -> None:
        if self.show_help:
            for idx, text in enumerate(self.HELP_TEXT):
                cv2.putText(frame, text, (10, 30 + idx*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(frame, text, (10, 30 + idx*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        fps = self.performance_stats['frame_count'] / (time.time() - self.performance_stats['start_time'])
        stats_text = f"FPS: {fps:.1f} | Faces: {len(self.faces)} | Overlay: {self.selected_overlay[0] if self.selected_overlay else 'None'}"
        cv2.putText(frame, stats_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, stats_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _save_current_frame(self, frame: np.ndarray) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"overlay_snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"Saved snapshot as {filename}")

    def _handle_keypress(self, key: int) -> bool:
        if key == ord('q'):
            return False
        elif key == ord('p'):
            self.show_real_face = not self.show_real_face
        elif key == ord('r'):
            self.same_overlay_mode = False
            self.current_overlays = {}
        elif key == ord('a'):
            self.same_overlay_mode = True
            self.selected_overlay = random.choice(self.overlays)
        elif key == ord('c'):
            self.same_overlay_mode = True
            self.cycle_index = (self.cycle_index + 1) % len(self.overlays)
            self.selected_overlay = self.overlays[self.cycle_index]
        elif key == ord('v'):
            self.same_overlay_mode = True
            self.cycle_index = (self.cycle_index - 1) % len(self.overlays)
            self.selected_overlay = self.overlays[self.cycle_index]
        elif key == ord('h'):
            self.show_help = not self.show_help
        elif key == ord('i'):
            if self.selected_overlay:
                logger.info(f"Current overlay: {self.selected_overlay[0]}")
        elif key == ord('s'):
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self._save_current_frame(frame)
        elif key == ord('+'):
            self.overlay_scale = min(1.5, self.overlay_scale + 0.1)
        elif key == ord('-'):
            self.overlay_scale = max(0.5, self.overlay_scale - 0.1)
        return True

    def run(self) -> None:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            frame = cv2.flip(frame, 1)
            frame = self._process_frame(frame)
            self._display_ui(frame)
            cv2.imshow("Face Overlay Pro 🚀", frame)
            self.performance_stats['frame_count'] += 1
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_keypress(key):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        app = FaceOverlayApp()
        app.run()
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        exit(1)
