import io
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="YOLO Inference Server", version="1.0.0")

MODELS_DIR = Path("/app/models")
_model_cache: dict[str, YOLO] = {}


def get_model(model_id: str) -> YOLO:
    if model_id not in _model_cache:
        model_path = MODELS_DIR / model_id
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file '{model_id}' was not found in {MODELS_DIR}."
            )
        _model_cache[model_id] = YOLO(str(model_path))
    return _model_cache[model_id]


@app.get("/")
def root():
    return {"name": "YOLO Inference Server", "status": "running"}


@app.get("/info")
def info():
    return {
        "models_dir": str(MODELS_DIR),
        "loaded_models": list(_model_cache.keys()),
    }


@app.post("/infer/{model_id}")
async def infer(
    model_id: str,
    image: UploadFile = File(...),
    confidence: float = Form(0.4),
    iou: float = Form(0.45),
    image_size: int = Form(640),
):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    try:
        model = get_model(model_id)
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"Could not load model '{model_id}': {e}"
        )

    t0 = time.perf_counter()
    results = model.predict(
        source=pil_image,
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        verbose=False,
    )
    inference_time = (time.perf_counter() - t0) * 1000  # ms

    predictions = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            predictions.append(
                {
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "confidence": float(box.conf[0]),
                    "class": result.names[int(box.cls[0])],
                    "class_id": int(box.cls[0]),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

    return JSONResponse(
        {
            "model_id": model_id,
            "inference_time_ms": round(inference_time, 2),
            "image": {"width": pil_image.width, "height": pil_image.height},
            "predictions": predictions,
        }
    )


@app.get("/models")
def list_models():
    local = [p.name for p in MODELS_DIR.glob("*.pt")]
    return {"models": local, "loaded": list(_model_cache.keys())}


@app.post("/models/{model_id}/load")
def load_model(model_id: str):
    try:
        get_model(model_id)
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"Could not load model '{model_id}': {e}"
        )
    return {"loaded": model_id}


@app.delete("/models/{model_id}/unload")
def unload_model(model_id: str):
    if model_id in _model_cache:
        del _model_cache[model_id]
        return {"status": "success", "unloaded": model_id}
    # Return 200 even if model wasn't loaded (idempotent)
    return {"status": "success", "note": f"Model '{model_id}' was not loaded"}
