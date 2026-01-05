from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
import shutil
import os
import uuid

from stage_1 import stage1_validate

app = FastAPI(
    title="Stage-1 Image Sanity Check",
    version="1.2"
)

UPLOAD_DIR = "/tmp/stage1_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/stage1/validate/batch")
async def stage1_validate_batch(
    files: List[UploadFile] = File(...),     # <-- MULTIPLE FILES
    person_id: str = Form(...),
    photo_type: str = Form("PRIMARY")
):
    """
    Stage-1 batch validation for multiple images
    """

    if not person_id.strip():
        raise HTTPException(status_code=400, detail="person_id is required")

    if photo_type not in {"PRIMARY", "SECONDARY"}:
        raise HTTPException(
            status_code=400,
            detail="photo_type must be PRIMARY or SECONDARY"
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one image must be uploaded"
        )

    results = []

    for file in files:
        ext = os.path.splitext(file.filename)[1]
        temp_filename = f"{person_id}_{uuid.uuid4()}{ext}"
        temp_path = os.path.join(UPLOAD_DIR, temp_filename)

        try:
            # ---------- Save ----------
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # ---------- Validate ----------
            result = stage1_validate(
                image_path=temp_path,
                photo_type=photo_type
            )

            # ---------- Attach metadata ----------
            result.update({
                "person_id": person_id,
                "file_name": file.filename
            })

            results.append(result)

        except Exception as e:
            results.append({
                "person_id": person_id,
                "file_name": file.filename,
                "stage": 1,
                "result": "REJECT",
                "reason": f"Processing error: {str(e)}",
                "checks": {}
            })

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return {
        "person_id": person_id,
        "photo_type": photo_type,
        "total_images": len(files),
        "results": results
    }
