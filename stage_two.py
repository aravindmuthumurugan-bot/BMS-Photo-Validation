import cv2
import os
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError("❌ GPU NOT DETECTED")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("✅ TensorFlow GPU enabled:", gpus)
from deepface import DeepFace
import easyocr


# ==================== CONFIGURATION ====================

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

# Pose angles (DeepFace doesn't provide these directly, so we'll use face size/position as proxy)
MIN_FACE_COVERAGE = 0.15  # Face should cover at least 15% of image

# Text detection
PII_PATTERNS = {
    "phone": r'(\+?\d[\d\s\-\(\)]{7,}\d)',
    "email": r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
    "social_instagram": r'@[a-zA-Z0-9_.]+',
    "social_twitter": r'@[a-zA-Z0-9_]+',
    "url": r'(www\.|https?://)[^\s]+',
}

# Enhancement/filter detection
FILTER_SATURATION_THRESHOLD = 1.5

# Paper-of-photo indicators
PAPER_WHITE_THRESHOLD = 240

# Face similarity thresholds for duplicate detection
DUPLICATE_THRESHOLD_STRICT = 0.40  # DeepFace distance (lower = more similar)
DUPLICATE_THRESHOLD_REVIEW = 0.50

# Initialize OCR (singleton)
_ocr_reader = None


print(" Initializing EasyOCR on GPU...")
OCR_READER = easyocr.Reader(['en'], gpu=True)

# ==================== UTILITY FUNCTIONS ====================

def extract_text_from_image(img_path: str) -> List[str]:
    try:
        results = OCR_READER.readtext(img_path)
        return [text for (_, text, prob) in results if prob > 0.5]
    except:
        return []



def check_for_pii(texts: List[str]) -> Tuple[bool, str]:
    """Check if extracted text contains PII"""
    combined_text = ' '.join(texts)
    
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, combined_text, re.IGNORECASE)
        if matches:
            return True, f"{pii_type} detected: {matches[0]}"
    
    return False, None


# ==================== DEEPFACE COMPREHENSIVE ANALYSIS ====================

def analyze_face_comprehensive(img_path: str) -> Dict:
    """
    Single DeepFace call for all attributes.
    This is more efficient than multiple calls.
    """
    try:
        # Single analysis call for all attributes
        results = DeepFace.analyze(
            img_path=img_path,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=True,
            detector_backend='retinaface',
            silent=True
        )
        
        if not results or len(results) == 0:
            return {
                "error": "No face detected",
                "data": None
            }
        
        face_data = results[0]
        
        return {
            "error": None,
            "data": face_data
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "data": None
        }


# ==================== PRIORITY 1: CRITICAL CHECKS (MUST PASS) ====================

def validate_age(img_path: str, profile_age: int, face_data: Dict = None) -> Dict:
    """Age verification - CRITICAL CHECK"""
    try:
        if face_data is None:
            analysis = analyze_face_comprehensive(img_path)
            if analysis["error"]:
                return {
                    "status": "REVIEW",
                    "reason": f"Age detection failed: {analysis['error']}",
                    "detected_age": None,
                    "profile_age": profile_age
                }
            face_data = analysis["data"]
        
        detected_age = face_data["age"]
        
        if detected_age is None:
            return {
                "status": "REVIEW",
                "reason": "Could not detect age from photo",
                "detected_age": None,
                "profile_age": profile_age
            }
        
        variance = abs(detected_age - profile_age)
        
        # CRITICAL: Check for underage
        if detected_age < 18:
            return {
                "status": "FAIL",
                "reason": f"Underage detected: {detected_age} years",
                "detected_age": detected_age,
                "profile_age": profile_age,
                "variance": variance,
                "action": "SUSPEND"
            }
        
        # Extra scrutiny for young ages
        if detected_age < 23:
            return {
                "status": "REVIEW",
                "reason": f"Young age detected: {detected_age} years. Manual verification recommended.",
                "detected_age": detected_age,
                "profile_age": profile_age,
                "variance": variance
            }
        
        if variance < AGE_VARIANCE_PASS:
            return {
                "status": "PASS",
                "reason": f"Age verified: {detected_age} (profile: {profile_age}, variance: {variance} years)",
                "detected_age": detected_age,
                "profile_age": profile_age,
                "variance": variance
            }
        elif variance <= AGE_VARIANCE_REVIEW:
            return {
                "status": "REVIEW",
                "reason": f"Moderate age variance: profile {profile_age}, detected {detected_age} (variance: {variance} years)",
                "detected_age": detected_age,
                "profile_age": profile_age,
                "variance": variance
            }
        else:
            return {
                "status": "FAIL",
                "reason": f"Large age variance: profile {profile_age}, detected {detected_age} (variance: {variance} years)",
                "detected_age": detected_age,
                "profile_age": profile_age,
                "variance": variance
            }
        
    except Exception as e:
        return {
            "status": "REVIEW",
            "reason": f"Age detection failed: {str(e)}",
            "detected_age": None,
            "profile_age": profile_age
        }


def check_fraud_database(img_path: str, fraud_db_photos: List[str] = None) -> Dict:
    """Check against fraud database - CRITICAL CHECK"""
    try:
        if fraud_db_photos is None or len(fraud_db_photos) == 0:
            return {
                "status": "PASS",
                "reason": "No fraud database configured",
                "checked": False
            }
        
        for idx, fraud_photo in enumerate(fraud_db_photos):
            try:
                result = DeepFace.verify(
                    img1_path=img_path,
                    img2_path=fraud_photo,
                    model_name='Facenet512',
                    detector_backend='retinaface',
                    enforce_detection=True,
                    silent=True
                )
                
                distance = result["distance"]
                
                if distance < 0.40:
                    return {
                        "status": "FAIL",
                        "reason": f"FRAUD ALERT: Matches fraud database entry #{idx}",
                        "distance": distance,
                        "fraud_id": idx,
                        "action": "SUSPEND"
                    }
            except:
                continue
        
        return {
            "status": "PASS",
            "reason": "No match in fraud database",
            "checked": True
        }
        
    except Exception as e:
        return {
            "status": "REVIEW",
            "reason": f"Fraud check error: {str(e)}",
            "checked": False
        }


def check_text_and_pii(img_path: str) -> Dict:
    """Detect PII - CRITICAL CHECK"""
    try:
        texts = extract_text_from_image(img_path)
        
        if not texts:
            return {
                "status": "PASS",
                "reason": "No text detected",
                "texts": []
            }
        
        has_pii, pii_details = check_for_pii(texts)
        
        if has_pii:
            return {
                "status": "FAIL",
                "reason": f"PII detected: {pii_details}",
                "texts": texts,
                "action": "WARN_AND_SELFIE_VERIFY"
            }
        
        return {
            "status": "REVIEW",
            "reason": f"Text detected but no PII: {texts[:3]}",
            "texts": texts
        }
        
    except Exception as e:
        return {
            "status": "REVIEW",
            "reason": f"OCR failed: {str(e)}",
            "texts": []
        }


# ==================== PRIORITY 2: HIGH IMPORTANCE CHECKS ====================

def validate_gender(img_path: str, profile_gender: str, face_data: Dict = None) -> Dict:
    """Gender validation"""
    try:
        if face_data is None:
            analysis = analyze_face_comprehensive(img_path)
            if analysis["error"]:
                return {
                    "status": "REVIEW",
                    "reason": f"Face analysis failed: {analysis['error']}",
                    "detected": None,
                    "expected": profile_gender
                }
            face_data = analysis["data"]
        
        gender_scores = face_data["gender"]
        man_score = gender_scores.get("Man", 0)
        woman_score = gender_scores.get("Woman", 0)
        
        detected_gender = "Male" if man_score > woman_score else "Female"
        confidence = max(man_score, woman_score) / 100.0
        
        if confidence < GENDER_CONFIDENCE_THRESHOLD:
            return {
                "status": "REVIEW",
                "reason": f"Gender detection confidence low ({confidence:.2f})",
                "detected": detected_gender,
                "expected": profile_gender,
                "confidence": confidence
            }
        
        if detected_gender.lower() != profile_gender.lower():
            return {
                "status": "FAIL",
                "reason": f"Gender mismatch: detected {detected_gender}, profile says {profile_gender}",
                "detected": detected_gender,
                "expected": profile_gender,
                "confidence": confidence
            }
        
        return {
            "status": "PASS",
            "reason": f"Gender verified as {detected_gender} ({confidence:.2f})",
            "detected": detected_gender,
            "expected": profile_gender,
            "confidence": confidence
        }
        
    except Exception as e:
        return {
            "status": "REVIEW",
            "reason": f"Gender detection failed: {str(e)}",
            "detected": None,
            "expected": profile_gender
        }


def validate_ethnicity(img_path: str, face_data: Dict = None) -> Dict:
    """Ethnicity validation"""
    try:
        if face_data is None:
            analysis = analyze_face_comprehensive(img_path)
            if analysis["error"]:
                return {
                    "status": "REVIEW",
                    "reason": f"Ethnicity detection failed: {analysis['error']}",
                    "indian_probability": None
                }
            face_data = analysis["data"]

        race_scores = face_data.get("race", {})
        indian_prob = race_scores.get("indian", 0) / 100.0

        for ethnicity, threshold in DISALLOWED_ETHNICITIES.items():
            prob = race_scores.get(ethnicity, 0) / 100.0
            if prob > threshold:
                return {
                    "status": "FAIL",
                    "reason": f"Non-Indian ethnicity detected: {ethnicity} ({prob:.2f})",
                    "indian_probability": indian_prob,
                    "detected_ethnicity": ethnicity,
                    "all_scores": race_scores
                }

        if indian_prob < INDIAN_PROBABILITY_MIN:
            return {
                "status": "REVIEW",
                "reason": f"Low Indian probability ({indian_prob:.2f})",
                "indian_probability": indian_prob,
                "all_scores": race_scores
            }

        return {
            "status": "PASS",
            "reason": f"Indian ethnicity verified ({indian_prob:.2f})",
            "indian_probability": indian_prob,
            "all_scores": race_scores
        }

    except Exception as e:
        return {
            "status": "REVIEW",
            "reason": f"Ethnicity detection failed: {str(e)}",
            "indian_probability": None
        }


def check_celebrity_database(img_path: str, celebrity_db_photos: List[str] = None) -> Dict:
    """Check against celebrity database"""
    try:
        if celebrity_db_photos is None or len(celebrity_db_photos) == 0:
            return {
                "status": "PASS",
                "reason": "No celebrity database configured",
                "checked": False
            }
        
        for idx, celeb_photo in enumerate(celebrity_db_photos):
            try:
                result = DeepFace.verify(
                    img1_path=img_path,
                    img2_path=celeb_photo,
                    model_name='Facenet512',
                    detector_backend='retinaface',
                    enforce_detection=True,
                    silent=True
                )
                
                distance = result["distance"]
                
                if distance < 0.35:
                    return {
                        "status": "FAIL",
                        "reason": f"Celebrity photo detected (distance: {distance:.3f})",
                        "distance": distance,
                        "celebrity_id": idx
                    }
            except:
                continue
        
        return {
            "status": "PASS",
            "reason": "No celebrity match detected",
            "checked": True
        }
        
    except Exception as e:
        return {
            "status": "REVIEW",
            "reason": f"Celebrity check error: {str(e)}",
            "checked": False
        }


# ==================== PRIORITY 3: STANDARD CHECKS ====================

def check_face_coverage(img_path: str, face_data: Dict = None) -> Dict:
    """Face coverage check"""
    try:
        if face_data is None:
            analysis = analyze_face_comprehensive(img_path)
            if analysis["error"]:
                return {
                    "status": "REVIEW",
                    "reason": f"Face coverage check failed: {analysis['error']}"
                }
            face_data = analysis["data"]
        
        region = face_data.get("region", {})
        face_x = region.get("x", 0)
        face_y = region.get("y", 0)
        face_w = region.get("w", 0)
        face_h = region.get("h", 0)
        
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        face_area = face_w * face_h
        img_area = img_w * img_h
        coverage = face_area / img_area if img_area > 0 else 0
        
        if coverage < MIN_FACE_COVERAGE:
            return {
                "status": "FAIL",
                "reason": f"Face too small in frame ({coverage:.2%} coverage)",
                "coverage": coverage
            }
        
        face_center_x = face_x + face_w / 2
        face_center_y = face_y + face_h / 2
        img_center_x = img_w / 2
        img_center_y = img_h / 2
        
        offset_x = abs(face_center_x - img_center_x) / img_w
        offset_y = abs(face_center_y - img_center_y) / img_h
        
        if offset_x > 0.3 or offset_y > 0.3:
            return {
                "status": "REVIEW",
                "reason": f"Face not centered. May indicate improper framing.",
                "coverage": coverage,
                "offset_x": offset_x,
                "offset_y": offset_y
            }
        
        return {
            "status": "PASS",
            "reason": f"Proper face framing ({coverage:.2%} coverage)",
            "coverage": coverage
        }
        
    except Exception as e:
        return {
            "status": "REVIEW",
            "reason": f"Face coverage check failed: {str(e)}"
        }


def check_duplicate(img_path: str, existing_photos: List[str] = None) -> Dict:
    """Duplicate detection"""
    try:
        if not existing_photos or len(existing_photos) == 0:
            return {
                "status": "PASS",
                "reason": "First photo, no duplicates to check"
            }
        
        for idx, existing_photo in enumerate(existing_photos):
            try:
                result = DeepFace.verify(
                    img1_path=img_path,
                    img2_path=existing_photo,
                    model_name='Facenet512',
                    detector_backend='retinaface',
                    enforce_detection=True,
                    silent=True
                )
                
                distance = result["distance"]
                
                if distance < DUPLICATE_THRESHOLD_STRICT:
                    return {
                        "status": "FAIL",
                        "reason": f"Duplicate or very similar to existing photo #{idx+1} (distance: {distance:.3f})",
                        "distance": distance,
                        "duplicate_index": idx
                    }
                elif distance < DUPLICATE_THRESHOLD_REVIEW:
                    return {
                        "status": "REVIEW",
                        "reason": f"Similar to existing photo #{idx+1} (distance: {distance:.3f})",
                        "distance": distance,
                        "duplicate_index": idx
                    }
            except:
                continue
        
        return {
            "status": "PASS",
            "reason": "Unique photo"
        }
        
    except Exception as e:
        return {
            "status": "REVIEW",
            "reason": f"Duplicate check failed: {str(e)}"
        }


def detect_digital_enhancement(img_path: str) -> Dict:
    """Enhancement detection"""
    img = cv2.imread(img_path)
    
    checks = {}
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()
    saturation_std = hsv[:, :, 1].std()
    
    if saturation > 150 and saturation_std < 40:
        checks["saturation"] = {
            "status": "FAIL",
            "reason": f"Unnaturally high saturation detected (filter applied)"
        }
    else:
        checks["saturation"] = {
            "status": "PASS",
            "reason": "Natural color saturation"
        }
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.count_nonzero(edges) / edges.size
    
    if edge_ratio > 0.20:
        checks["cartoon"] = {
            "status": "REVIEW",
            "reason": f"Possible cartoon/anime filter (high edge ratio: {edge_ratio:.3f})"
        }
    else:
        checks["cartoon"] = {
            "status": "PASS",
            "reason": "Natural photograph"
        }
    
    return checks


def detect_photo_of_photo(img_path: str) -> Dict:
    """Photo-of-photo detection"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    border_width = int(min(h, w) * 0.05)
    
    top_border = gray[:border_width, :].mean()
    bottom_border = gray[-border_width:, :].mean()
    left_border = gray[:, :border_width].mean()
    right_border = gray[:, -border_width:].mean()
    
    border_mean = np.mean([top_border, bottom_border, left_border, right_border])
    
    edges = cv2.Canny(gray, 50, 150)
    
    if border_mean > PAPER_WHITE_THRESHOLD:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (h * w * 0.3):
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    return {
                        "status": "FAIL",
                        "reason": "Photo of printed photo detected"
                    }
    
    return {
        "status": "PASS",
        "reason": "Original digital photo"
    }


def detect_ai_generated(img_path: str) -> Dict:
    """AI-generated detection"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    mean = cv2.filter2D(gray, -1, kernel)
    sqr_mean = cv2.filter2D(gray ** 2, -1, kernel)
    variance = sqr_mean - mean ** 2
    
    avg_variance = variance.mean()
    
    if avg_variance < 100:
        return {
            "status": "REVIEW",
            "reason": f"Possible AI-generated image (low texture variance: {avg_variance:.1f})",
            "confidence": "MEDIUM"
        }
    
    return {
        "status": "PASS",
        "reason": "Appears to be authentic photograph",
        "confidence": "MEDIUM"
    }


def detect_watermark(img_path: str) -> Dict:
    """Watermark detection"""
    texts = extract_text_from_image(img_path)
    
    watermark_keywords = [
        "shutterstock", "getty", "watermark", "©", "copyright",
        "dreamstime", "istockphoto", "alamy", "deposit"
    ]
    
    for text in texts:
        text_lower = text.lower()
        for keyword in watermark_keywords:
            if keyword in text_lower:
                return {
                    "status": "FAIL",
                    "reason": f"Third-party watermark detected: {text}"
                }
    
    return {
        "status": "PASS",
        "reason": "No third-party watermarks detected"
    }


# ==================== OPTIMIZED MAIN VALIDATOR WITH EARLY EXIT ====================

def stage2_validate_optimized(
    image_path: str,
    profile_data: Dict,
    existing_photos: List[str] = None,
    fraud_db_photos: List[str] = None,
    celebrity_db_photos: List[str] = None
) -> Dict:
    """
    Stage 2 validation with EARLY EXIT optimization.

    Priority levels:
    1. CRITICAL: Age (underage check), Fraud DB, PII
    2. HIGH: Gender, Ethnicity, Celebrity DB
    3. STANDARD: Face coverage, Duplicates, Enhancement, Photo-of-photo, AI, Watermark

    Exit behavior:
    - FAIL status: Exits immediately and stops all further checks
    - REVIEW status: Continues to all remaining checks to collect comprehensive review reasons
    - PASS status: Continues to next check
    """
    
    results = {
        "stage": 2,
        "matri_id": profile_data.get("matri_id"),
        "checks": {},
        "checks_performed": [],
        "checks_skipped": [],
        "final_decision": None,
        "action": None,
        "reason": None,
        "early_exit": False
    }
    
    if existing_photos is None:
        existing_photos = []
    
    # ============= SINGLE DEEPFACE ANALYSIS =============
    print("Running comprehensive face analysis...")
    analysis = analyze_face_comprehensive(image_path)
    face_data = analysis["data"] if not analysis["error"] else None
    
    if analysis["error"]:
        results["final_decision"] = "REVIEW"
        results["action"] = "SEND_TO_HUMAN"
        results["reason"] = f"Face detection failed: {analysis['error']}"
        results["early_exit"] = True
        return results
    
    # ============= PRIORITY 1: CRITICAL CHECKS (EARLY EXIT ON FAIL ONLY) =============

    # 1. AGE CHECK (CRITICAL - Underage detection)
    print("[P1] Checking age...")
    results["checks"]["age"] = validate_age(image_path, profile_data.get("age", 25), face_data)
    results["checks_performed"].append("age")

    # Exit immediately only on FAIL (SUSPEND action), continue on REVIEW
    if results["checks"]["age"]["status"] == "FAIL" and results["checks"]["age"].get("action") == "SUSPEND":
        results["final_decision"] = "SUSPEND"
        results["action"] = "SUSPEND_PROFILE"
        results["reason"] = "Underage detected - immediate suspension"
        results["early_exit"] = True
        results["checks_skipped"] = ["fraud_db", "text_pii", "gender", "ethnicity", "celebrity_db",
                                     "face_coverage", "duplicate", "enhancement", "photo_of_photo",
                                     "ai_generated", "watermark"]
        return results

    # 2. FRAUD DATABASE CHECK (CRITICAL)
    print("[P1] Checking fraud database...")
    results["checks"]["fraud_db"] = check_fraud_database(image_path, fraud_db_photos)
    results["checks_performed"].append("fraud_db")

    # Exit immediately only on FAIL
    if results["checks"]["fraud_db"]["status"] == "FAIL":
        results["final_decision"] = "SUSPEND"
        results["action"] = "SUSPEND_PROFILE"
        results["reason"] = "Fraud database match - immediate suspension"
        results["early_exit"] = True
        results["checks_skipped"] = ["text_pii", "gender", "ethnicity", "celebrity_db",
                                     "face_coverage", "duplicate", "enhancement", "photo_of_photo",
                                     "ai_generated", "watermark"]
        return results

    # 3. PII CHECK (CRITICAL)
    print("[P1] Checking for PII...")
    results["checks"]["text_pii"] = check_text_and_pii(image_path)
    results["checks_performed"].append("text_pii")

    # Exit immediately only on FAIL, continue on REVIEW
    if results["checks"]["text_pii"]["status"] == "FAIL":
        results["final_decision"] = "REJECT"
        results["action"] = "WARN_AND_SELFIE_VERIFY"
        results["reason"] = "PII detected in photo"
        results["early_exit"] = True
        results["checks_skipped"] = ["gender", "ethnicity", "celebrity_db", "face_coverage",
                                     "duplicate", "enhancement", "photo_of_photo", "ai_generated", "watermark"]
        return results

    # ============= PRIORITY 2: HIGH IMPORTANCE CHECKS (EARLY EXIT ON FAIL ONLY) =============

    # 4. GENDER CHECK
    print("[P2] Checking gender...")
    results["checks"]["gender"] = validate_gender(image_path, profile_data.get("gender", "Unknown"), face_data)
    results["checks_performed"].append("gender")

    # Exit immediately only on FAIL, continue on REVIEW
    if results["checks"]["gender"]["status"] == "FAIL":
        results["final_decision"] = "REJECT"
        results["action"] = "SELFIE_VERIFICATION"
        results["reason"] = "Gender mismatch detected"
        results["early_exit"] = True
        results["checks_skipped"] = ["ethnicity", "celebrity_db", "face_coverage", "duplicate",
                                     "enhancement", "photo_of_photo", "ai_generated", "watermark"]
        return results

    # 5. ETHNICITY CHECK
    print("[P2] Checking ethnicity...")
    results["checks"]["ethnicity"] = validate_ethnicity(image_path, face_data)
    results["checks_performed"].append("ethnicity")

    # Exit immediately only on FAIL, continue on REVIEW
    if results["checks"]["ethnicity"]["status"] == "FAIL":
        results["final_decision"] = "REJECT"
        results["action"] = "SELFIE_VERIFICATION"
        results["reason"] = "Ethnicity check failed"
        results["early_exit"] = True
        results["checks_skipped"] = ["celebrity_db", "face_coverage", "duplicate", "enhancement",
                                     "photo_of_photo", "ai_generated", "watermark"]
        return results

    # 6. CELEBRITY DATABASE CHECK
    print("[P2] Checking celebrity database...")
    results["checks"]["celebrity_db"] = check_celebrity_database(image_path, celebrity_db_photos)
    results["checks_performed"].append("celebrity_db")

    # Exit immediately only on FAIL, continue on REVIEW
    if results["checks"]["celebrity_db"]["status"] == "FAIL":
        results["final_decision"] = "REJECT"
        results["action"] = "SELFIE_VERIFICATION"
        results["reason"] = "Celebrity photo detected"
        results["early_exit"] = True
        results["checks_skipped"] = ["face_coverage", "duplicate", "enhancement", "photo_of_photo",
                                     "ai_generated", "watermark"]
        return results
    
    # ============= PRIORITY 3: STANDARD CHECKS (CONTINUE ALL) =============
    # These checks run to completion for comprehensive reporting
    
    print("[P3] Running standard checks...")
    
    # 7. Face Coverage
    results["checks"]["face_coverage"] = check_face_coverage(image_path, face_data)
    results["checks_performed"].append("face_coverage")
    
    # 8. Duplicate Detection
    results["checks"]["duplicate"] = check_duplicate(image_path, existing_photos)
    results["checks_performed"].append("duplicate")
    
    # 9. Digital Enhancement
    results["checks"]["enhancement"] = detect_digital_enhancement(image_path)
    results["checks_performed"].append("enhancement")
    
    # 10. Photo-of-photo
    results["checks"]["photo_of_photo"] = detect_photo_of_photo(image_path)
    results["checks_performed"].append("photo_of_photo")
    
    # 11. AI-generated
    results["checks"]["ai_generated"] = detect_ai_generated(image_path)
    results["checks_performed"].append("ai_generated")
    
    # 12. Watermark
    results["checks"]["watermark"] = detect_watermark(image_path)
    results["checks_performed"].append("watermark")
    
    # ============= FINAL DECISION LOGIC =============
    
    fail_checks = []
    review_checks = []
    
    for check_name, check_result in results["checks"].items():
        if isinstance(check_result, dict) and "status" in check_result:
            if check_result["status"] == "FAIL":
                fail_checks.append(check_name)
            elif check_result["status"] == "REVIEW":
                review_checks.append(check_name)
        else:
            for sub_check, sub_result in check_result.items():
                if sub_result["status"] == "FAIL":
                    fail_checks.append(f"{check_name}.{sub_check}")
                elif sub_result["status"] == "REVIEW":
                    review_checks.append(f"{check_name}.{sub_check}")
    
    if fail_checks:
        results["final_decision"] = "REJECT"
        results["action"] = determine_rejection_action(fail_checks, results["checks"])
        results["reason"] = f"Failed checks: {', '.join(fail_checks)}"
    elif review_checks:
        results["final_decision"] = "MANUAL_REVIEW"
        results["action"] = "SEND_TO_HUMAN"
        results["reason"] = f"Requires manual review: {', '.join(review_checks)}"
    else:
        results["final_decision"] = "APPROVE"
        results["action"] = "PUBLISH"
        results["reason"] = "All checks passed"
    
    return results


def determine_rejection_action(fail_checks: List[str], all_checks: Dict) -> str:
    """Determine action based on failed checks"""
    
    if any(check in fail_checks for check in ["age", "fraud_db"]):
        return "SUSPEND_PROFILE"
    
    if "text_pii" in fail_checks:
        return "WARN_AND_SELFIE_VERIFY"
    
    if any(check in fail_checks for check in ["celebrity_db", "gender", "ethnicity"]):
        return "SELFIE_VERIFICATION"
    
    if "duplicate" in fail_checks:
        return "NUDGE_DELETE_DUPLICATES"
    
    if "enhancement" in fail_checks:
        return "NUDGE_UPLOAD_ORIGINAL"
    
    if "photo_of_photo" in fail_checks:
        return "NUDGE_UPLOAD_DIGITAL"
    
    return "NUDGE_REUPLOAD_PROPER"


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    
    profile_data = {
        "matri_id": "BM123456",
        "gender": "Male",
        "age": 28
    }
    
    result = stage2_validate_optimized(
        image_path="Fullface.jpeg",
        profile_data=profile_data,
        existing_photos=[]
    )
    
    print("\n" + "="*60)
    print(f"STAGE 2 VALIDATION (OPTIMIZED) - Matri ID: {result['matri_id']}")
    print("="*60)
    print(f"\nEARLY EXIT: {result['early_exit']}")
    print(f"CHECKS PERFORMED: {result['checks_performed']}")
    if result['checks_skipped']:
        print(f"CHECKS SKIPPED: {result['checks_skipped']}")
    print(f"\nFINAL DECISION: {result['final_decision']}")
    print(f"ACTION: {result['action']}")
    print(f"REASON: {result['reason']}")
    
    print("\n" + "-"*60)
    print("CHECKS COMPLETED:")
    print("-"*60)
    
    for check_name in result["checks_performed"]:
        check_result = result["checks"][check_name]
        print(f"\n[{check_name.upper()}]")
        
        if isinstance(check_result, dict) and "status" in check_result:
            print(f"  Status: {check_result['status']}")
            print(f"  Reason: {check_result['reason']}")
        else:
            for sub_check, sub_result in check_result.items():
                print(f"  {sub_check}:")
                print(f"    Status: {sub_result['status']}")
                print(f"    Reason: {sub_result['reason']}")
    
    print("\n" + "="*60)
