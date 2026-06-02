from ultralytics import YOLO
import cv2
import time
from collections import defaultdict, deque
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "best (5).pt"

# Pickup Zone
PICKUP_X1 = 250
PICKUP_Y1 = 120
PICKUP_X2 = 420
PICKUP_Y2 = 330

OUTPUT_FILE = "robot_coordinates.txt"

CONFIDENCE_THRESHOLD = 0.30
IMG_SIZE = 416

MIN_BOX_WIDTH = 40
MIN_BOX_HEIGHT = 40

CONFIRM_FRAMES = 5
SMOOTHING_WINDOW = 5

# ==========================================
# LOAD MODEL
# ==========================================

try:
    model = YOLO(MODEL_PATH)
    print("[INFO] Model Loaded Successfully")
except Exception as e:
    print(f"[ERROR] {e}")
    exit()

# ==========================================
# DEEPSORT
# ==========================================

tracker = DeepSort(
    max_age=15,
    n_init=5,
    max_cosine_distance=0.4
)

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("[ERROR] Camera Not Found")
    exit()

print("[INFO] Press Q to Exit")

# ==========================================
# VARIABLES
# ==========================================

prev_time = 0

track_counter = defaultdict(int)

track_history = defaultdict(
    lambda: deque(maxlen=SMOOTHING_WINDOW)
)

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_h * frame_w

    # ======================================
    # YOLO DETECTION
    # ======================================

    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=0.45,
        imgsz=IMG_SIZE,
        verbose=False
    )

    detections = []

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        width = x2 - x1
        height = y2 - y1

        # Reject tiny boxes
        if width < MIN_BOX_WIDTH:
            continue

        if height < MIN_BOX_HEIGHT:
            continue

        # Reject giant boxes
        box_area = width * height

        if box_area > frame_area * 0.60:
            continue

        conf = float(box.conf[0])
        print(f"CONF: {conf:.2f}")
        cls = int(box.cls[0])

        detections.append(
            (
                [x1, y1, width, height],
                conf,
                cls
            )
        )

    # ======================================
    # TRACKING
    # ======================================

    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    coordinate_lines = []

    # ======================================
    # PROCESS TRACKS
    # ======================================

    for track in tracks:

        # Skip unconfirmed tracks
        if not track.is_confirmed():
            continue

        # Ignore stale / ghost tracks
        if track.time_since_update > 1:
            continue

        track_id = track.track_id

        track_counter[track_id] += 1

        # Require 5 stable frames
        if track_counter[track_id] < CONFIRM_FRAMES:
            continue

        ltrb = track.to_ltrb()

        x1 = int(ltrb[0])
        y1 = int(ltrb[1])
        x2 = int(ltrb[2])
        y2 = int(ltrb[3])

        width = x2 - x1
        height = y2 - y1

        # Bottom-center pickup point
        cx = int((x1 + x2) / 2)
        cy = int(y2 - (height * 0.1))

        # ==================================
        # COORDINATE SMOOTHING
        # ==================================

        track_history[track_id].append(
            (cx, cy)
        )

        avg_x = int(
            sum(
                p[0]
                for p in track_history[track_id]
            )
            /
            len(track_history[track_id])
        )

        avg_y = int(
            sum(
                p[1]
                for p in track_history[track_id]
            )
            /
            len(track_history[track_id])
        )

        # ==================================
        # PICKUP ZONE
        # ==================================

        inside_pickup = (
            PICKUP_X1 <= avg_x <= PICKUP_X2
            and
            PICKUP_Y1 <= avg_y <= PICKUP_Y2
        )

        color = (0, 255, 0)

        if inside_pickup:
            color = (0, 0, 255)

        # ==================================
        # DRAW BOX
        # ==================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.circle(
            frame,
            (avg_x, avg_y),
            5,
            (255, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"ID:{track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # ==================================
        # SAVE COORDINATES
        # ==================================

        if inside_pickup:

            coordinate_lines.append(
                f"ID:{track_id},X:{avg_x},Y:{avg_y}"
            )

    # ======================================
    # WRITE TO FILE
    # ======================================

    with open(OUTPUT_FILE, "w") as f:

        if coordinate_lines:

            for line in coordinate_lines:
                f.write(line + "\n")

        else:
            f.write(
                "NO_OBJECT_IN_PICKUP_ZONE\n"
            )

    # ======================================
    # DRAW PICKUP ZONE
    # ======================================

    cv2.rectangle(
        frame,
        (PICKUP_X1, PICKUP_Y1),
        (PICKUP_X2, PICKUP_Y2),
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "PICKUP ZONE",
        (PICKUP_X1, PICKUP_Y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # ======================================
    # FPS
    # ======================================

    current_time = time.time()

    fps = (
        1 / (current_time - prev_time)
        if prev_time != 0
        else 0
    )

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "Bottle Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()