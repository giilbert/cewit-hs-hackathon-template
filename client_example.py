import io
import sys
import time

import cv2
import requests
from PIL import Image

SERVER = "http://localhost:18001"
MODEL = "yolov8n.pt"
CONFIDENCE = 0.4
CAMERA_INDEX = 0

COLORS = [
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (44, 153, 168),
    (0, 194, 255),
    (52, 69, 147),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199),
]


def encode_frame(frame) -> bytes:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def draw_predictions(frame, predictions):
    for pred in predictions:
        x1 = int(pred["bbox"]["x1"])
        y1 = int(pred["bbox"]["y1"])
        x2 = int(pred["bbox"]["x2"])
        y2 = int(pred["bbox"]["y2"])
        cls_id = pred["class_id"]
        label = f"{pred['class']} {pred['confidence']:.2f}"
        color = COLORS[cls_id % len(COLORS)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
    return frame


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL
    infer_url = f"{SERVER}/infer/{model}"

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}")
        sys.exit(1)

    print(f"Streaming camera {CAMERA_INDEX} → {infer_url}  (q to quit)")

    fps_display = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        try:
            resp = requests.post(
                infer_url,
                files={"image": ("frame.jpg", encode_frame(frame), "image/jpeg")},
                data={"confidence": CONFIDENCE},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            frame = draw_predictions(frame, data["predictions"])
            server_ms = data["inference_time_ms"]
        except Exception as e:
            cv2.putText(
                frame,
                f"Server error: {e}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            server_ms = 0

        elapsed = time.perf_counter() - t0
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / elapsed)

        cv2.putText(
            frame,
            f"FPS: {fps_display:.1f}  server: {server_ms:.0f}ms",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLO Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
