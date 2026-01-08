import os
import cv2
import re
import json
import time
import shutil
import uuid
import numpy as np
from typing import List, Dict
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from deepface import DeepFace
import easyocr

# ==================== GPU CONFIG ====================

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError("❌ GPU NOT DETECTED")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("✅ TensorFlow GPU enabled:", gpus)

# ==================== CONFIGURATION ====================

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Gender detection confidence
GENDER_CONFIDENCE_THRESHOLD = 0.70

# Ethnicity thresholds
INDIAN_PROBABILITY_MIN = 0.30
DISALLOWED_ETHNICITIES = {
    "white": 0.60,
    "black": 0.60,
    "asian": 0.50,
    "middle eastern": 0.60,
    "latino hispanic": 0.60
}

# Age variance thresholds
AGE_VARIANCE_PASS = 8
AGE_VARIANCE_REVIEW = 15

# Face coverage
MIN_FACE_COVERAGE = 0.15  # Face should cover at least 15% of image

# Text detection patterns (PII)
PII_PATTERNS = {
    "phone": r'(\+?\d[\d\s\-\(\)]{7,}\d)',
    "email": r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
    "social_instagram": r'@[a-zA-Z0-9_.]+',
    "social_twitter": r'@[a-zA-Z0-9_]+',
    "url": r'(www\.|https?://)[^\s]+',
}

# ==================== INITIALIZE OCR ====================
print("Initializing EasyOCR on GPU...")
OCR_READER = easyocr.Reader(['en'], gpu=True)

# ==================== UTILITY FUNCTIONS ====================

def extract_text_from_image(img_path: str) -> List[str]:
    try:
        results = OCR_READER.readtext(img_path)
        return [text for (_, text, prob) in results if prob > 0.5]
    except Exception:
        return []

def check_for_pii(texts: List[str]) -> (bool, str):
    combined_text = ' '.join(texts)
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, combined_text, re.IGNORECASE)
        if matches:
            return True, f"{pii_type} detected: {matches[0]}"
    return False, None

# ==================== DEEPFACE HELPER ====================

def analyze_face_comprehensive(img_path: str) -> Dict:
    """
    Ensures we always pass a proper NumPy array to DeepFace to avoid KerasTensor errors.
    """
    try:
        # Read image in BGR -> convert to RGB
        img = cv2.imread(img_path)
        if img is None:
            return {"error": "Image could not be read", "data": None}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = DeepFace.analyze(
            img_rgb,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=True,
            detector_backend='retinaface',
            silent=True
        )

        if not results or len(results) == 0:
            return {"error": "No face detected", "data": None}

        face_data = results[0]
        return {"error": None, "data": face_data}

    except Exception as e:
        return {"error": str(e), "data": None}

# ==================== VALIDATION FUNCTIONS ====================

def validate_age(img_path: str, profile_age: int, face_data: Dict = None) -> Dict:
    try:
        if face_data is None:
            face_data = analyze_face_comprehensive(img_path)["data"]

        detected_age = face_data.get("age", None)
        if detected_age is None:
            return {"status": "REVIEW", "reason": "Could not detect age", "detected_age": None}

        variance = abs(detected_age - profile_age)

        if detected_age < 18:
            return {"status": "FAIL", "reason": f"Underage detected: {detected_age}", "action": "SUSPEND"}

        if variance < AGE_VARIANCE_PASS:
            return {"status": "PASS", "reason": f"Age verified: {detected_age}"}
        elif variance <= AGE_VARIANCE_REVIEW:
            return {"status": "REVIEW", "reason": f"Moderate age variance: {detected_age}"}
        else:
            return {"status": "FAIL", "reason": f"Large age variance: {detected_age}"}

    except Exception as e:
        return {"status": "REVIEW", "reason": f"Age detection failed: {str(e)}"}

def validate_gender(img_path: str, profile_gender: str, face_data: Dict = None) -> Dict:
    try:
        if face_data is None:
            face_data = analyze_face_comprehensive(img_path)["data"]

        gender_scores = face_data.get("gender", {})
        man_score = gender_scores.get("Man", 0)
        woman_score = gender_scores.get("Woman", 0)
        detected_gender = "Male" if man_score > woman_score else "Female"
        confidence = max(man_score, woman_score)/100.0

        if confidence < GENDER_CONFIDENCE_THRESHOLD:
            return {"status": "REVIEW", "reason": f"Low confidence ({confidence:.2f})", "detected": detected_gender}

        if detected_gender.lower() != profile_gender.lower():
            return {"status": "FAIL", "reason": f"Gender mismatch: {detected_gender}"}

        return {"status": "PASS", "reason": f"Gender verified as {detected_gender}"}

    except Exception as e:
        return {"status": "REVIEW", "reason": f"Gender detection failed: {str(e)}"}

def validate_ethnicity(img_path: str, face_data: Dict = None) -> Dict:
    try:
        if face_data is None:
            face_data = analyze_face_comprehensive(img_path)["data"]

        race_scores = face_data.get("race", {})
        indian_prob = race_scores.get("indian",0)/100.0

        for ethnicity, threshold in DISALLOWED_ETHNICITIES.items():
            prob = race_scores.get(ethnicity,0)/100.0
            if prob > threshold:
                return {"status": "FAIL", "reason": f"Non-Indian ethnicity detected: {ethnicity}"}

        if indian_prob < INDIAN_PROBABILITY_MIN:
            return {"status": "REVIEW", "reason": f"Low Indian probability ({indian_prob:.2f})"}

        return {"status": "PASS", "reason": f"Indian ethnicity verified ({indian_prob:.2f})"}

    except Exception as e:
        return {"status": "REVIEW", "reason": f"Ethnicity detection failed: {str(e)}"}

def check_text_and_pii(img_path: str) -> Dict:
    texts = extract_text_from_image(img_path)
    if not texts:
        return {"status": "PASS", "reason": "No text detected", "texts": []}

    has_pii, pii_details = check_for_pii(texts)
    if has_pii:
        return {"status": "FAIL", "reason": f"PII detected: {pii_details}", "action": "WARN_AND_SELFIE_VERIFY"}

    return {"status": "PASS", "reason": "No PII detected"}

# ==================== MAIN VALIDATION FUNCTION ====================

def stage2_validate(image_path: str, profile_data: Dict) -> Dict:
    results = {"stage":2, "matri_id": profile_data.get("matri_id"),
               "checks":{}, "checks_performed":[], "early_exit":False}

    face_data = analyze_face_comprehensive(image_path)["data"]
    if face_data is None:
        return {"stage":2, "matri_id": profile_data.get("matri_id"),
                "final_decision":"REVIEW", "action":"SEND_TO_HUMAN",
                "reason":"Face detection failed", "early_exit":True}

    # Age (critical)
    results["checks"]["age"] = validate_age(image_path, profile_data.get("age",25), face_data)
    results["checks_performed"].append("age")
    if results["checks"]["age"]["status"]=="FAIL":
        results["final_decision"]="SUSPEND"
        results["action"]="SUSPEND_PROFILE"
        results["reason"]="Underage or age mismatch"
        results["early_exit"]=True
        return results

    # Gender
    results["checks"]["gender"] = validate_gender(image_path, profile_data.get("gender","Unknown"), face_data)
    results["checks_performed"].append("gender")

    # Ethnicity
    results["checks"]["ethnicity"] = validate_ethnicity(image_path, face_data)
    results["checks_performed"].append("ethnicity")

    # PII
    results["checks"]["text_pii"] = check_text_and_pii(image_path)
    results["checks_performed"].append("text_pii")

    # Final decision
    fail_checks = [k for k,v in results["checks"].items() if v["status"]=="FAIL"]
    review_checks = [k for k,v in results["checks"].items() if v["status"]=="REVIEW"]

    if fail_checks:
        results["final_decision"]="REJECT"
        results["action"]="SELFIE_VERIFICATION"
        results["reason"]=f"Failed checks: {', '.join(fail_checks)}"
    elif review_checks:
        results["final_decision"]="MANUAL_REVIEW"
        results["action"]="SEND_TO_HUMAN"
        results["reason"]=f"Requires manual review: {', '.join(review_checks)}"
    else:
        results["final_decision"]="APPROVE"
        results["action"]="PUBLISH"
        results["reason"]="All checks passed"

    return results

# ==================== FASTAPI SETUP ====================

app = FastAPI(title="Stage 2 Photo Validation API - GPU")

def save_uploaded_file(file: UploadFile) -> str:
    filename =file.filename
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return file_path

@app.post("/stage2/validate-images")
async def validate_multiple_images(files: List[UploadFile] = File(...), profile_data: str = Form(...)):
    start_time = time.time()

    try:
        profile_data = json.loads(profile_data)
    except:
        raise HTTPException(status_code=400, detail="Invalid profile_data JSON")

    if not files:
        raise HTTPException(status_code=400, detail="No images uploaded")

    saved_paths = [save_uploaded_file(f) for f in files]
    results = [stage2_validate(p, profile_data) for p in saved_paths]

    # Cleanup
    for p in saved_paths:
        try: os.remove(p)
        except: pass

    response_time = round(time.time() - start_time, 3)
    return JSONResponse({
        "total_images": len(files),
        "processed": len(results),
        "response_time_seconds": response_time,
        "results": results
    })

# ==================== MAIN ENTRY ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
