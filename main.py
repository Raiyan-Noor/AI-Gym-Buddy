import os
import shutil
import tempfile
from types import SimpleNamespace
from typing import Annotated
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from s3_utils import S3ClientWrapper
from curls import process_video


app = FastAPI(title="Curl Counter API", version="1.0.0")
s3_client = None


def get_s3_client() -> S3ClientWrapper:
    global s3_client
    if s3_client is None:
        s3_client = S3ClientWrapper()
    return s3_client


def cleanup_files(*paths: str) -> None:
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


@app.post("/process", summary="Process a curl video and upload annotated result to S3")
async def process_video_endpoint(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="MP4 video of curls")],
    arm: Annotated[str, Form(description="Arm to track (left/right)")] = "right",
    mode: Annotated[str, Form(description="Rep counting mode")] = "half_bottom",
    up_th: Annotated[float, Form(description="Top-of-curl angle threshold")] = 55.0,
    down_th: Annotated[float, Form(description="Bottom-of-curl angle threshold")] = 160.0,
    mid_th: Annotated[float, Form(description="Midpoint angle threshold")] = 120.0,
    smooth: Annotated[float, Form(description="Exponential smoothing factor")] = 0.3,
):
    if file.content_type not in {"video/mp4", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only MP4 uploads are supported.")

    tmp_dir = tempfile.mkdtemp(prefix="curl_api_")
    input_path = os.path.join(tmp_dir, "input.mp4")
    output_path = os.path.join(tmp_dir, "output.mp4")

    try:
        with open(input_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

        args = SimpleNamespace(
            video=input_path,
            out=output_path,
            arm=arm,
            mode=mode,
            up_th=up_th,
            down_th=down_th,
            mid_th=mid_th,
            smooth=smooth,
            preview=False,
        )

        process_video(args)

        s3 = get_s3_client()
        upload_id = uuid4().hex
        input_key = f"uploads/{upload_id}_{os.path.basename(file.filename or 'input.mp4')}"
        output_key = f"results/{upload_id}_annotated.mp4"

        s3.upload_file(input_path, input_key)
        s3.upload_file(output_path, output_key)
        presigned_result = s3.presign_url(output_key)

        background_tasks.add_task(shutil.rmtree, tmp_dir, ignore_errors=True)

        return JSONResponse(
            {
                "input_key": input_key,
                "output_key": output_key,
                "output_url": presigned_result,
                "arm": arm,
                "mode": mode,
            }
        )
    except Exception as exc:  # noqa: BLE001
        cleanup_files(input_path, output_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc
