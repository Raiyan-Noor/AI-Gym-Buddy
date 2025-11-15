# AI Gym Buddy

![Annotated curl demo](curl_processed.gif)

I always lose track of reps once a set really starts to burn. This POC was born to fix that, which is limited to bicep curls, but can be extended for other excercises as well. This was inspired by Farza Majeed’s AI super coach idea—read it here: https://www.linkedin.com/posts/farza-majeed-76685612a_i-used-gemini-25-pro-to-build-a-simple-shot-activity-7347242032068730899-Q0c5/?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAAC5mId8BdqdofF55P07W7etdOiH6XkfQ4WM

Although this repo is backend-only, it can be pluged into maybe a React Native client that handles capture and playback. The backend counts reps, annotates video, stores everything on S3, and returns presigned links.

## Features

- MediaPipe Pose + OpenCV pipeline draws landmarks, angles, and rep count overlays.
- Configurable rep modes (`full`, `half_top`, `half_bottom`) and angle thresholds for strictness tuning.
- JSON-driven coaching feedback (`feedback_rules.json`) so cues can change without code edits.
- FastAPI backend with a single upload endpoint that streams results back to a mobile app.

## Project Structure

```
.
├── curls.py            # MediaPipe/OpenCV processing & CLI
├── feedback_rules.json # Angle-driven coaching cues
├── main.py             # FastAPI service exposing /process
├── s3_utils.py         # boto3 helper for uploads + presigned URLs
├── curl_processed.gif  # Demo output (feel free to replace)
├── requirements.txt
└── .env                # Sample AWS credentials (never commit real ones)
```

## Getting Started

1. **Python**: Use 3.10+; MediaPipe wheels are smoothest there.
2. **Install deps**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install mediapipe opencv-python  # heavy wheels kept optional
   ```

3. **Configure AWS** (`.env` or environment variables)

   ```ini
   AWS_ACCESS_KEY_ID=your-key
   AWS_SECRET_ACCESS_KEY=your-secret
   AWS_REGION=ap-south-1
   AWS_S3_BUCKET=mybucketfor-images
   ```

4. **Run the API**

   ```bash
   uvicorn main:app --reload --port 8000
   ```

   Visit http://localhost:8000/docs for OpenAPI testing.

## API Reference

### `POST /process`

Processes a curl video, annotates it, uploads input/output to S3, and returns presigned URLs.

| Field   | Type   | Default        | Notes                                                |
|---------|--------|----------------|------------------------------------------------------|
| `file`  | binary | —              | Required MP4 upload                                  |
| `arm`   | str    | `right`        | `left` or `right`                                    |
| `mode`  | str    | `half_bottom`  | `full`, `half_top`, `half_bottom`                    |
| `up_th` | float  | `55.0`         | Angle at curl peak                                   |
| `down_th` | float| `160.0`        | Angle at lowest point                                |
| `mid_th`| float  | `120.0`        | Splits top/bottom halves                             |
| `smooth`| float  | `0.3`          | Exponential smoothing factor for angle signal        |

**Response**

```json
{
  "input_key": "uploads/123abc_input.mp4",
  "output_key": "results/123abc_annotated.mp4",
  "output_url": "https://s3.amazonaws.com/...",
  "arm": "right",
  "mode": "half_bottom"
}
```

`output_url` is a presigned link (defaults to 60 minutes). On errors, FastAPI returns `400` for validation issues and `500` for processing/S3 failures.

