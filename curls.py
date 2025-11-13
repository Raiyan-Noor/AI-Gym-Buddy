import argparse
import time
import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "This version of curls.py requires MediaPipe. "
        "Install it with `pip install mediapipe` and run again."
    ) from exc


POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
POSE_LANDMARK = mp.solutions.pose.PoseLandmark
DRAWING = mp.solutions.drawing_utils
DRAWING_STYLES = mp.solutions.drawing_styles

ARM_LANDMARKS = {
    "left": (
        POSE_LANDMARK.LEFT_SHOULDER,
        POSE_LANDMARK.LEFT_ELBOW,
        POSE_LANDMARK.LEFT_WRIST,
    ),
    "right": (
        POSE_LANDMARK.RIGHT_SHOULDER,
        POSE_LANDMARK.RIGHT_ELBOW,
        POSE_LANDMARK.RIGHT_WRIST,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Count dumbbell curls using MediaPipe Pose.")
    parser.add_argument("--video", default="dumbbell_curl.mp4", help="Path to input video file")
    parser.add_argument("--out", default="curls_output.mp4", help="Path to output video file")
    parser.add_argument("--arm", choices=["left", "right"], default="right", help="Arm to track")
    parser.add_argument(
        "--mode",
        choices=["full", "half_top", "half_bottom"],
        default="half_bottom",
        help="Rep counting mode",
    )
    parser.add_argument("--up-th", type=float, default=55.0, help="Angle threshold at curl peak")
    parser.add_argument("--down-th", type=float, default=160.0, help="Angle threshold at curl bottom")
    parser.add_argument("--mid-th", type=float, default=120.0, help="Angle threshold dividing top/bottom halves")
    parser.add_argument("--smooth", type=float, default=0.3, help="Exponential smoothing factor (0..1)")
    parser.add_argument("--preview", action="store_true", help="Show live preview window")
    return parser.parse_args()


def compute_angle(a, b, c) -> float:
    """Return angle ABC in degrees."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


class SmoothValue:
    def __init__(self, alpha: float):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.value: float | None = None

    def update(self, new_value: float) -> float:
        if self.value is None or self.alpha == 1.0:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value
        return self.value


class RepCounter:
    def __init__(self, mode: str, up_th: float, down_th: float, mid_th: float):
        self.mode = mode
        self.low_th, self.high_th = self._resolve_thresholds(up_th, down_th, mid_th)
        self.stage = "down"
        self.reps = 0
        self.last_rep_ts = 0.0

    def _resolve_thresholds(self, up: float, down: float, mid: float) -> tuple[float, float]:
        if self.mode == "full":
            return up, down
        if self.mode == "half_top":
            return up, mid
        return mid, down

    def update(self, angle: float) -> bool:
        counted = False
        if angle <= self.low_th:
            self.stage = "up"
        if angle >= self.high_th and self.stage == "up":
            self.stage = "down"
            self.reps += 1
            self.last_rep_ts = time.time()
            counted = True
        return counted

    def recently_counted(self, window: float = 0.5) -> bool:
        return (time.time() - self.last_rep_ts) < window

    def feedback(self, angle: float | None) -> str:
        if angle is None:
            return "Move the target arm into view"
        if angle > 175:
            return "Nice full extension"
        if angle < 45:
            return "Strong contraction"
        return "Squeeze at top" if self.stage == "up" else "Control the descent"


class MediaPipePoseTracker:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
        lm_style_fn = getattr(DRAWING_STYLES, "get_default_pose_landmarks_style", None)
        conn_style_fn = getattr(DRAWING_STYLES, "get_default_pose_connections_style", None)
        default_spec = DRAWING.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
        self.landmark_style = lm_style_fn() if lm_style_fn else default_spec
        self.connection_style = conn_style_fn() if conn_style_fn else default_spec

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if self.pose:
            self.pose.close()

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb)

    def draw_landmarks(self, frame, landmarks):
        DRAWING.draw_landmarks(
            frame,
            landmarks,
            POSE_CONNECTIONS,
            landmark_drawing_spec=self.landmark_style,
            connection_drawing_spec=self.connection_style,
        )

    def arm_points(self, landmarks, arm: str, width: int, height: int):
        idx_sh, idx_el, idx_wr = ARM_LANDMARKS[arm]
        pts = []
        for idx in (idx_sh, idx_el, idx_wr):
            lm = landmarks.landmark[idx]
            if lm.visibility < 0.4:
                return None
            pts.append((lm.x * width, lm.y * height))
        return pts


def draw_shadow_text(img, text, org, scale, color, thickness):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_arm_points(frame, points):
    for x, y in points:
        cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 255), -1)
    cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, (0, 200, 255), 2)


def process_video(args: argparse.Namespace):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    smoother = SmoothValue(args.smooth)
    counter = RepCounter(args.mode, args.up_th, args.down_th, args.mid_th)

    with MediaPipePoseTracker() as tracker:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                display = frame.copy()
                results = tracker.process(frame)
                angle = None
                rep_flash = False

                if results.pose_landmarks:
                    tracker.draw_landmarks(display, results.pose_landmarks)
                    arm_pts = tracker.arm_points(results.pose_landmarks, args.arm, width, height)
                    if arm_pts:
                        draw_arm_points(display, arm_pts)
                        angle = compute_angle(*arm_pts)
                        angle = smoother.update(angle)
                        rep_flash = counter.update(angle)
                        draw_shadow_text(
                            display,
                            f"Angle: {int(angle)} deg",
                            (30, height - 40),
                            1.0,
                            (255, 255, 255),
                            2,
                        )

                title = f"Curls ({args.arm.capitalize()} Arm, {args.mode.replace('_', ' ')})"
                draw_shadow_text(display, title, (30, 60), 1.2, (255, 255, 255), 2)

                rep_color = (0, 255, 0) if (rep_flash or counter.recently_counted()) else (255, 255, 255)
                draw_shadow_text(display, f"Reps: {counter.reps}", (30, 120), 1.8, rep_color, 5)

                feedback = counter.feedback(angle)
                draw_shadow_text(display, feedback, (30, 180), 0.9, (255, 255, 255), 2)

                writer.write(display)

                if args.preview:
                    cv2.imshow("Curl Counter", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

    print(f"Done. Output saved to {args.out}")


def main():
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
