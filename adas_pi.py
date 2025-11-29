import os
import time
import threading
import queue
import math
import numpy as np
import cv2
import glob
import csv

# Optional: YOLOv8
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Optional: TTS
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# Optional: Picamera2
try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None

# Optional: GPIO
try:
    import RPi.GPIO as GPIO
except Exception:
    GPIO = None


class Camera:
    def __init__(self, width=1280, height=720, use_display=True):
        self.use_display = use_display
        self.width = width
        self.height = height
        self.cap = None
        self.picam2 = None
        self._init_camera()

    def _init_camera(self):
        # Prefer standard OpenCV camera by default. Use Picamera2 only if explicitly requested.
        use_pipicam = os.getenv("ADAS_USE_PICAMERA", "0") == "1"
        if use_pipicam and Picamera2 is not None:
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(main={"size": (self.width, self.height), "format": "BGR888"})
                self.picam2.configure(config)
                self.picam2.start()
                return
            except Exception:
                self.picam2 = None
        # Choix du device caméra OpenCV : index (0,1,...) ou chemin (/dev/video20, etc.)
        cam_dev = os.getenv("ADAS_CAM_DEVICE", "0")
        try:
            # si c'est un nombre, utiliser un index entier
            cam_index = int(cam_dev)
            print(f"[Camera] Ouverture caméra OpenCV index {cam_index}")
            self.cap = cv2.VideoCapture(cam_index)
        except ValueError:
            # sinon, supposer un chemin de device (/dev/video20, etc.)
            print(f"[Camera] Ouverture caméra OpenCV device '{cam_dev}'")
            self.cap = cv2.VideoCapture(cam_dev)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        if self.picam2 is not None:
            frame = self.picam2.capture_array()
            return True, frame
        if self.cap is not None and self.cap.isOpened():
            return self.cap.read()
        return False, None

    def release(self):
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()
        if self.use_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                # OpenCV may be built without GUI support (headless). Ignore.
                print("[Camera] cv2.destroyAllWindows() not available (headless build).")


class LaneDetector:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.wheel_width_ratio = 0.18
        self.wheel_height_ratio = 0.12
        self.wheel_y_ratio = 0.85

    def _roi(self, img):
        h, w = img.shape[:2]
        mask = np.zeros_like(img)
        polygon = np.array([
            [int(0.05*w), h],
            [int(0.05*w), int(0.6*h)],
            [int(0.95*w), int(0.6*h)],
            [int(0.95*w), h]
        ])
        cv2.fillPoly(mask, [polygon], 255)
        return cv2.bitwise_and(img, mask)

    def detect(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # White mask
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        # Yellow mask
        yellow_lower = np.array([15, 70, 100])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        mask = self._roi(mask)
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=40, maxLineGap=100)
        line_segments = []
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                line_segments.append(((x1, y1), (x2, y2)))
        return line_segments, mask, edges

    def draw_lines(self, frame, line_segments, color=(0, 255, 255)):
        for (x1, y1), (x2, y2) in line_segments:
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    def _rects_for_wheels(self, w, h):
        rw = int(self.wheel_width_ratio * w)
        rh = int(self.wheel_height_ratio * h)
        y = int(self.wheel_y_ratio * h) - rh//2
        left_x = int(0.3 * w) - rw//2
        right_x = int(0.7 * w) - rw//2
        left = (left_x, y, rw, rh)
        right = (right_x, y, rw, rh)
        return left, right

    @staticmethod
    def _segment_intersects_rect(p1, p2, rect):
        x, y, w, h = rect
        rect_lines = [
            ((x, y), (x+w, y)),
            ((x+w, y), (x+w, y+h)),
            ((x+w, y+h), (x, y+h)),
            ((x, y+h), (x, y))
        ]
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        def intersect(a, b, c, d):
            return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)
        for r1, r2 in rect_lines:
            if intersect(p1, p2, r1, r2):
                return True
        return False

    def lane_departure(self, frame_shape, line_segments):
        h, w = frame_shape[:2]
        left_rect, right_rect = self._rects_for_wheels(w, h)
        left_cross = False
        right_cross = False
        for p1, p2 in line_segments:
            if self._segment_intersects_rect(p1, p2, left_rect):
                left_cross = True
            if self._segment_intersects_rect(p1, p2, right_rect):
                right_cross = True
        return left_cross or right_cross, (left_rect, right_rect)

    def draw_wheels(self, frame, rects, alert=False):
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255) if alert else (0, 255, 0), 2)


class SignDetector:
    def __init__(self, model_path, device=None, tts_enabled=True):
        self.model = YOLO(model_path) if (YOLO is not None and os.path.exists(model_path)) else None
        self.device = device
        self.tts_enabled = tts_enabled and (pyttsx3 is not None)
        self.tts = None
        if self.tts_enabled:
            try:
                self.tts = pyttsx3.init()
            except Exception as e:
                # Désactiver proprement la synthèse vocale si le moteur (eSpeak, etc.) n'est pas disponible.
                print(f"[SignDetector] TTS désactivé (pyttsx3.init a échoué) : {e}")
                self.tts_enabled = False
        self.last_announced = {}
        self.cooldown = 3.0
        # Prefer model-provided class names if available
        self.class_map = None
        self.class_source = "unknown"
        if self.model is not None:
            try:
                # ultralytics>=8: model.names is a dict {id: name}
                names = getattr(self.model, 'names', None)
                if names is None and hasattr(self.model, 'model'):
                    names = getattr(self.model.model, 'names', None)
                if isinstance(names, dict) and len(names) > 0:
                    self.class_map = {int(k): str(v) for k, v in names.items()}
                    self.class_source = "model.names"
            except Exception:
                self.class_map = None
        # Fallback: load names from a labels file (one class per line, ordered by index)
        force_labels = os.getenv('ADAS_FORCE_LABELS', '0') == '1'
        if self.class_map is None or force_labels:
            labels_path = os.getenv('ADAS_LABELS', os.path.join('static', 'models', 'coco.txt'))
            if os.path.exists(labels_path):
                try:
                    with open(labels_path, 'r', encoding='utf-8') as f:
                        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                    if lines:
                        self.class_map = {i: lines[i] for i in range(len(lines))}
                        self.class_source = f"labels_file:{labels_path}"
                except Exception:
                    pass
        # Fallback minimal mapping if names not found
        if self.class_map is None:
            self.class_map = {
                0: "Stop",
                1: "Yield",
                2: "Speed limit",
            }
            self.class_source = "fallback_default"

        # If class_map is empty or model failed to provide names, try a fallback model
        if (self.class_map is None) or (isinstance(self.class_map, dict) and len(self.class_map) == 0):
            fallback = os.getenv('ADAS_MODEL_FALLBACK', os.path.join('static', 'models', 'best.pt'))
            if fallback != model_path and os.path.exists(fallback):
                try:
                    print(f"[SignDetector] primary model provided no names — trying fallback model: {fallback}")
                    m2 = YOLO(fallback) if YOLO is not None else None
                    names2 = None
                    if m2 is not None:
                        names2 = getattr(m2, 'names', None)
                        if names2 is None and hasattr(m2, 'model'):
                            names2 = getattr(m2.model, 'names', None)
                    if isinstance(names2, dict) and len(names2) > 0:
                        self.model = m2
                        self.class_map = {int(k): str(v) for k, v in names2.items()}
                        self.class_source = f"model.names(fallback:{fallback})"
                except Exception:
                    pass

        # Print summary for debugging
        try:
            preview = ", ".join([self.class_map[i] for i in sorted(self.class_map.keys())[:5]])
            print(f"[SignDetector] Loaded {len(self.class_map)} classes from {self.class_source}. Preview: {preview} ...")
        except Exception:
            pass

    def announce(self, text):
        if not self.tts_enabled or not text or self.tts is None:
            return
        now = time.time()
        if text in self.last_announced and now - self.last_announced[text] < self.cooldown:
            return
        self.last_announced[text] = now
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception:
            pass

    @staticmethod
    def _prettify_label(label: str) -> str:
        if not label:
            return label
        s = label.replace('_', ' ').strip()
        # Simple French prettifier for common classes
        lower = s.lower()
        if lower.startswith('stop'):
            return 'Stop'
        if 'yield' in lower or 'give way' in lower:
            return 'Cédez le passage'
        if 'no entry' in lower or 'do not enter' in lower or 'no_enter' in lower or 'sens interdit' in lower:
            return 'Sens interdit'
        if 'priority road' in lower or 'priority' in lower:
            return 'Route prioritaire'
        if 'end of priority' in lower or 'end priority' in lower:
            return 'Fin de route prioritaire'
        if 'roundabout' in lower:
            return 'Cédez le passage – Rond-point'
        if 'turn left' in lower or 'left turn' in lower or 'keep left' in lower:
            return 'Tournez à gauche'
        if 'turn right' in lower or 'right turn' in lower or 'keep right' in lower:
            return 'Tournez à droite'
        if 'straight' in lower or 'go straight' in lower or 'ahead only' in lower:
            return 'Tout droit'
        if 'pedestrian' in lower or 'crosswalk' in lower or 'zebra' in lower:
            return 'Passage piéton'
        if 'children' in lower or 'school' in lower:
            return 'Attention école'
        if 'construction' in lower or 'road work' in lower or 'works' in lower:
            return 'Travaux'
        if 'bicycle' in lower or 'bike' in lower or 'cycle' in lower:
            return 'Piste cyclable'
        if 'no parking' in lower:
            return 'Stationnement interdit'
        if 'no stopping' in lower or 'no standing' in lower:
            return 'Arrêt interdit'
        if 'green light' in lower or ('traffic light' in lower and 'green' in lower):
            return 'Feu vert'
        if 'red light' in lower or ('traffic light' in lower and 'red' in lower):
            return 'Feu rouge'
        if 'yellow light' in lower or ('traffic light' in lower and 'yellow' in lower) or 'amber' in lower:
            return 'Feu orange'
        if 'speed' in lower and 'limit' in lower:
            # Try to extract number
            digits = ''.join(ch for ch in s if ch.isdigit())
            if digits:
                return f'Limitation de vitesse {digits}'
            return 'Limitation de vitesse'
        return s

    def detect(self, frame):
        results = []
        if self.model is None:
            return results
        if frame is None:
            return results
        try:
            # Use the same calling form used elsewhere in the project
            pred = self.model.predict(source=[frame], conf=0.4, verbose=False)[0]
            boxes = getattr(pred, 'boxes', None)
            if boxes is None or len(boxes) == 0:
                return results
            for b in boxes:
                xy = getattr(b, 'xyxy', None)
                if xy is None:
                    continue
                xyarr = xy.cpu().numpy()
                if xyarr.size == 0:
                    continue
                xyxy = xyarr[0].astype(int)
                cls = int(b.cls.cpu().numpy()[0]) if getattr(b, 'cls', None) is not None else -1
                conf = float(b.conf.cpu().numpy()[0]) if getattr(b, 'conf', None) is not None else 0.0
                raw_label = self.class_map.get(cls, f"classe_{cls}")
                label = self._prettify_label(raw_label)
                results.append((xyxy, label, conf))
        except Exception:
            # Keep detection failure silent and return empty list so main loop can continue
            return []
        return results


class UltrasonicSensor:
    def __init__(self, trig_pin=23, echo_pin=24, buzzer_pin=18, alert_threshold_cm=10.0):
        self.trig = trig_pin
        self.echo = echo_pin
        self.buzzer = buzzer_pin
        self.threshold = alert_threshold_cm
        self.distance = None
        self.running = False
        self.thread = None
        self.q = queue.Queue(maxsize=1)
        self.use_gpio = GPIO is not None
        if self.use_gpio:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.trig, GPIO.OUT)
            GPIO.setup(self.echo, GPIO.IN)
            GPIO.setup(self.buzzer, GPIO.OUT)
            GPIO.output(self.trig, GPIO.LOW)

    def _measure_once(self, timeout=0.03):
        if not self.use_gpio:
            # Simulate 30-200 cm
            return 100.0
        GPIO.output(self.trig, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trig, GPIO.LOW)
        start_time = time.time()
        while GPIO.input(self.echo) == 0:
            if time.time() - start_time > timeout:
                return None
        pulse_start = time.time()
        while GPIO.input(self.echo) == 1:
            if time.time() - pulse_start > timeout:
                return None
        pulse_end = time.time()
        duration = pulse_end - pulse_start
        distance = (duration * 34300) / 2
        return distance

    def _loop(self):
        while self.running:
            d = self._measure_once()
            if d is not None:
                self.distance = d
                try:
                    if not self.q.empty():
                        self.q.get_nowait()
                    self.q.put_nowait(d)
                except queue.Full:
                    pass
            if self.use_gpio:
                try:
                    GPIO.output(self.buzzer, GPIO.HIGH if (d is not None and d < self.threshold) else GPIO.LOW)
                except Exception:
                    pass
            time.sleep(0.05)

    def start(self):
        if self.thread is not None:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.thread = None
        if self.use_gpio:
            try:
                GPIO.output(self.buzzer, GPIO.LOW)
                GPIO.cleanup()
            except Exception:
                pass

    def latest_distance(self):
        return self.distance


def draw_overlay(frame, lane_detector, line_segments, wheel_rects, lane_alert, signs, distance):
    h, w = frame.shape[:2]
    lane_detector.draw_lines(frame, line_segments)
    lane_detector.draw_wheels(frame, wheel_rects, alert=lane_alert)
    if lane_alert:
        cv2.putText(frame, "Attention, franchissement de ligne", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    y = 70
    for xyxy, label, conf in signs[:3]:
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y += 20
    if distance is not None:
        cv2.putText(frame, f"Obstacle: {distance:.0f} cm", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def main():
    width = int(os.getenv("ADAS_CAM_W", 1280))
    height = int(os.getenv("ADAS_CAM_H", 720))
    show = os.getenv("ADAS_DISPLAY", "1") == "1"
    # Default to best1.pt (you can override with ADAS_MODEL env var)
    model_path = os.getenv("ADAS_MODEL", os.path.join("static", "models", "best.pt"))

    cam = Camera(width=width, height=height, use_display=show)
    lane = LaneDetector(width, height)
    sign = SignDetector(model_path=model_path, tts_enabled=True)
    ultra = UltrasonicSensor(alert_threshold_cm=10.0)
    ultra.start()

    # If display requested, quickly check whether OpenCV GUI functions are available.
    if show:
        try:
            # Create the window once; it will be reused by cv2.imshow in the main loop.
            cv2.namedWindow("ADAS", cv2.WINDOW_NORMAL)
        except Exception:
            print("[ADAS] OpenCV GUI functions not available. Running in headless mode.\nSet ADAS_DISPLAY=0 to silence this message or install a GUI-enabled OpenCV build.")
            show = False
            # ensure Camera won't try to destroy windows
            try:
                cam.use_display = False
            except Exception:
                pass

    # Optional analysis mode: analyze images from a folder (default: dataset test images)
    # Enable by setting ADAS_ANALYZE_INPUT=1. Override folder with ADAS_ANALYZE_FOLDER.
    analyze = os.getenv("ADAS_ANALYZE_INPUT", "0") == "1"
    if analyze:
        # Default analyze folder: match web UI and other scripts which use `static/input/gphoto`
        folder = os.getenv("ADAS_ANALYZE_FOLDER", os.path.join("static", "input", "gphoto"))
        out_csv = os.getenv("ADAS_ANALYZE_OUT", os.path.join(folder, "detections.csv"))
        imgs = []
        # recursive search so nested folders are supported
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            imgs.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
        unique_labels = set()
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["image", "x1", "y1", "x2", "y2", "label", "conf"])
            for path in imgs:
                img = cv2.imread(path)
                if img is None:
                    continue
                dets = sign.detect(img)
                for (x1, y1, x2, y2), label, conf in dets:
                    writer.writerow([os.path.basename(path), x1, y1, x2, y2, label, f"{conf:.3f}"])
                    unique_labels.add(label)
        print(f"Analyse terminée: {len(imgs)} images, CSV: {out_csv}")
        if unique_labels:
            print("Labels détectés:", ", ".join(sorted(unique_labels)))
        return

    last_sign_announce = 0.0
    sign_every_n_frames = 3
    fcount = 0
    # Persistence des panneaux pour que les noms restent visibles un peu plus longtemps
    sign_persist_sec = 1.5
    last_signs = []
    last_signs_time = 0.0

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.05)
                continue

            line_segments, mask, edges = lane.detect(frame)
            alert, wheel_rects = lane.lane_departure(frame.shape, line_segments)
            if alert:
                sign.announce("Attention, franchissement de ligne")

            signs = []
            now = time.time()
            if fcount % sign_every_n_frames == 0:
                signs = sign.detect(frame)
                if signs:
                    # Mémoriser les derniers panneaux détectés et l'instant de détection
                    last_signs = signs
                    last_signs_time = now
                for _, label, conf in signs:
                    if conf >= 0.6:
                        sign.announce(label)

            # Si aucun panneau détecté à ce frame, réutiliser les derniers panneaux
            # pendant sign_persist_sec secondes pour que le texte reste lisible.
            if not signs and (now - last_signs_time) < sign_persist_sec:
                signs = last_signs

            distance = ultra.latest_distance()

            draw_overlay(frame, lane, line_segments, wheel_rects, alert, signs, distance)

            if show:
                try:
                    cv2.imshow("ADAS", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                except Exception:
                    # OpenCV imshow may fail if package is headless; switch to headless mode
                    print("[ADAS] cv2.imshow failed — switching to headless mode. Set ADAS_DISPLAY=0 if you prefer.")
                    show = False
                    try:
                        cam.use_display = False
                    except Exception:
                        pass
            else:
                # Headless small sleep to avoid maxing CPU
                time.sleep(0.005)

            fcount += 1
    except KeyboardInterrupt:
        pass
    finally:
        ultra.stop()
        cam.release()


if __name__ == "__main__":
    main()
