import os
import uuid
import shutil
import json
import numpy as np
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from stage_2_optimised import stage2_validate_optimized

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Stage 2 Photo Validation API - Optimized")

# Tune based on CPU cores
MAX_WORKERS = os.cpu_count() or 4


def convert_to_native_types(obj):
    """
    Recursively convert NumPy types to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    return obj


def process_single_image(image_path: str, profile_data: dict):
    """
    This function runs in a separate process with early exit optimization
    """
    try:
        result = stage2_validate_optimized(
            image_path=image_path,
            profile_data=profile_data,
            existing_photos=[]
        )
        # Convert NumPy types to native Python types
        result = convert_to_native_types(result)
        
        return {
            "image": os.path.basename(image_path),
            "success": True,
            "result": result
        }
    except Exception as e:
        return {
            "image": os.path.basename(image_path),
            "success": False,
            "error": str(e)
        }


@app.post("/stage2/validate-images")
def validate_multiple_images(
    files: List[UploadFile] = File(...),
    profile_data: str = Form(...)
):
    """
    Upload multiple images and validate them in parallel with early exit
    """

    try:
        profile_data = json.loads(profile_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid profile_data JSON")

    if not files:
        raise HTTPException(status_code=400, detail="No images uploaded")

    saved_paths = []

    # Save files first
    for file in files:
        filename = file.filename
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_paths.append(file_path)

    results = []

    # Parallel processing
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_image, path, profile_data): path
            for path in saved_paths
        }

        for future in as_completed(futures):
            results.append(future.result())

    # Cleanup uploaded files
    for path in saved_paths:
        try:
            os.remove(path)
        except:
            pass

    return JSONResponse({
        "total_images": len(files),
        "processed": len(results),
        "results": results
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fast_api_2_optimized:app", host="0.0.0.0", port=8000)
