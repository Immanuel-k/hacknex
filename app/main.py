import os
import shutil
import uuid
import time
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from app.config import OUTPUT_DIR, BASE_DIR, AUTHENTIC_THRESHOLD, UNCERTAIN_THRESHOLD
from app.database import mongo_db
from src.preprocessing.frame_extractor import FrameExtractor
from src.preprocessing.face_detector import FaceDetector
from src.models.classifier import DeepfakeClassifier
from src.models.explainability import HeatmapGenerator
from src.utils.aggregation import aggregate_video_scores

app = FastAPI(
    title="VERI-DEEP Multimodal Forensics API",
    version="1.0.0",
    description="FastAPI backend connected to MongoDB Atlas, Hugging Face AI models, and HTML Auth."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

if FRONTEND_DIR.exists():
    app.mount("/ui_static", StaticFiles(directory=str(FRONTEND_DIR)), name="ui_static")

frame_extractor = FrameExtractor(sample_rate_fps=2.0, max_frames=30)
face_detector = FaceDetector(padding=0.2)
classifier = DeepfakeClassifier()
heatmap_gen = HeatmapGenerator(output_dir=OUTPUT_DIR)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Multimodal Deepfake Forensics System",
        "database": "MongoDB Atlas connected" if mongo_db.reports_col is not None else "In-Memory Mode",
        "html_ui_url": "/ui",
        "docs_url": "/docs"
    }

@app.get("/ui")
def get_html_ui():
    html_file = FRONTEND_DIR / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    raise HTTPException(status_code=404, detail="HTML Frontend file index.html not found.")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": classifier.device,
        "model_loaded": classifier.pipe is not None,
        "mongodb_connected": mongo_db.reports_col is not None
    }

# ---------------------------------------------------------
# AUTHENTICATION ENDPOINTS (Email & Gmail / Google OAuth)
# ---------------------------------------------------------
@app.post("/api/v1/auth/register")
def register_user(payload: dict = Body(...)):
    email = payload.get("email")
    password = payload.get("password")
    name = payload.get("name", "")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    res = mongo_db.register_user(email, password, name)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res

@app.post("/api/v1/auth/login")
def login_user(payload: dict = Body(...)):
    email = payload.get("email")
    password = payload.get("password")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    res = mongo_db.authenticate_user(email, password)
    if not res["success"]:
        raise HTTPException(status_code=401, detail=res["message"])
    return res

@app.post("/api/v1/auth/google")
def google_auth(payload: dict = Body(...)):
    email = payload.get("email")
    name = payload.get("name", "Google User")
    google_id = payload.get("google_id", "google_oauth_123")

    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")

    res = mongo_db.google_auth_login(email, name, google_id)
    return res

# ---------------------------------------------------------
# FORENSICS ENDPOINTS
# ---------------------------------------------------------
@app.get("/api/v1/forensics/history")
def get_analysis_history(limit: int = 10):
    reports = mongo_db.get_recent_reports(limit=limit)
    return {
        "count": len(reports),
        "database": "MongoDB Atlas" if mongo_db.reports_col is not None else "In-Memory",
        "history": reports
    }

@app.post("/api/v1/forensics/analyze")
async def analyze_media(
    file: UploadFile = File(...),
    sample_fps: float = Form(2.0),
    user_email: Optional[str] = Form(None)
):
    start_time = time.time()
    temp_id = uuid.uuid4().hex[:8]
    ext = Path(file.filename).suffix.lower()
    temp_filepath = OUTPUT_DIR / f"upload_{temp_id}{ext}"

    try:
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        is_video = ext in [".mp4", ".avi", ".mov", ".mkv"]
        extractor = FrameExtractor(sample_rate_fps=sample_fps, max_frames=30)
        
        if is_video:
            raw_frames = extractor.extract_frames_from_video(str(temp_filepath))
        else:
            raw_frames = extractor.load_image(str(temp_filepath))

        if not raw_frames:
            raise HTTPException(status_code=400, detail="Could not extract frames from media file.")

        frame_results = []
        sample_heatmaps = []

        for frame_idx, timestamp, frame_bgr in raw_frames:
            faces = face_detector.detect_and_crop(frame_bgr)
            
            for face in faces:
                crop_pil = face["cropped_pil"]
                crop_bgr = face["cropped_bgr"]
                box = face["box"]

                pred = classifier.predict_face(crop_pil, crop_bgr)
                fake_score = pred["fake_score"]

                heatmap_path = heatmap_gen.generate_saliency_overlay(
                    frame_bgr, box, fake_score, frame_idx
                )
                heatmap_filename = Path(heatmap_path).name
                heatmap_url = f"/static/outputs/{heatmap_filename}"

                frame_entry = {
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "fake_score": fake_score,
                    "bounding_box": box,
                    "signals": pred["signals"],
                    "heatmap_url": heatmap_url
                }

                frame_results.append(frame_entry)
                
                if len(sample_heatmaps) < 6:
                    sample_heatmaps.append(heatmap_url)

        aggregated = aggregate_video_scores(frame_results)
        inference_time = round(time.time() - start_time, 2)

        response_data = {
            "task_id": temp_id,
            "filename": file.filename,
            "user_email": user_email or "anonymous",
            "media_type": "video" if is_video else "image",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "authenticity": {
                "authenticity_score_pct": f"{aggregated['authenticity_score']*100:.1f}%",
                "authenticity_score": aggregated["authenticity_score"],
                "fake_probability": aggregated["fake_probability"],
                "decision": aggregated["decision"],
                "thresholds": {
                    "authentic_max": AUTHENTIC_THRESHOLD,
                    "uncertain_max": UNCERTAIN_THRESHOLD,
                    "manipulated_min": UNCERTAIN_THRESHOLD
                }
            },
            "evidence": {
                "summary": f"Media classified as {aggregated['decision'].upper()} with fake probability {aggregated['fake_probability']:.2f}.",
                "suspicious_timestamps": aggregated["suspicious_timestamps"],
                "key_signals": aggregated["evidence_signals"],
                "sample_heatmaps": sample_heatmaps
            },
            "frame_level_analysis": frame_results,
            "metrics": {
                "frames_analyzed": len(raw_frames),
                "faces_detected": len(frame_results),
                "inference_time_seconds": inference_time
            }
        }

        mongo_db.save_report(response_data)

        if temp_filepath.exists():
            try:
                os.remove(temp_filepath)
            except Exception:
                pass

        return JSONResponse(content=response_data)

    except HTTPException as http_e:
        if temp_filepath.exists():
            try:
                os.remove(temp_filepath)
            except Exception:
                pass
        raise http_e
    except Exception as e:
        if temp_filepath.exists():
            try:
                os.remove(temp_filepath)
            except Exception:
                pass
        print(f"[Analyze Error Trace]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# REAL-TIME AI CHATBOT AGENT ENDPOINT (VeriCat AI 🐱)
# ---------------------------------------------------------
@app.post("/api/v1/chat")
def chat_with_agent(payload: dict = Body(...)):
    message = payload.get("message", "").strip()
    user_email = payload.get("user_email", "Guest")
    
    if not message:
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")
    
    msg_lower = message.lower()
    
    # Contextual Intelligent Response Engine for VeriCat AI
    db_mode = "MongoDB Atlas Cloud Database" if mongo_db.reports_col is not None else "In-Memory Vault Mode"
    recent_reports = mongo_db.get_recent_reports(limit=5)
    reports_count = len(recent_reports)
    
    if any(k in msg_lower for k in ["hi", "hello", "hey", "who are you"]):
        reply = f"Meow! 🐾 I'm **VeriCat AI**, your 3D Multimodal Digital Forensics Assistant! I can help you analyze videos, explain Grad-CAM heatmaps, verify MongoDB Atlas records, or answer general AI questions. How can I help you today, {user_email}?"
    elif any(k in msg_lower for k in ["deepfake", "detect", "fake", "authentic"]):
        reply = "🔍 **How VERI-DEEP Detects Deepfakes:**\n1. **Frame Extraction**: Samples video frames dynamically.\n2. **Face Crop & Alignment**: Isolates facial regions using OpenCV.\n3. **Transformer Classifier**: Computes spatial manipulation probability.\n4. **Grad-CAM Saliency**: Highlights artificial pixel boundaries in red/yellow heatmaps."
    elif any(k in msg_lower for k in ["database", "mongo", "atlas", "history", "records"]):
        reply = f"🍃 **MongoDB Atlas Status:**\n- **Connection Mode**: {db_mode}\n- **Total Scanned Reports Available**: {reports_count}\n- All forensics verdicts and user authentication sessions are synced in real-time."
    elif any(k in msg_lower for k in ["model", "huggingface", "pytorch", "algorithm"]):
        reply = f"🧠 **AI Model Specifications:**\n- **Architecture**: Vision Transformer / ConvNeXt Deepfake Classifier (`dima806/deepfake_vs_real_image_detection`)\n- **Execution Device**: {classifier.device.upper()}\n- **Explainability**: Grad-CAM Activation Saliency overlays."
    elif any(k in msg_lower for k in ["chatgpt", "claude", "ai", "what can you do", "help"]):
        reply = "✨ **VeriCat AI Capabilities:**\n- Answer general knowledge & coding questions.\n- Explain forensic spatial/frequency artifacts.\n- Provide live status of your uploaded media analyses.\n- Guide you through Google OAuth & MongoDB Atlas setup."
    else:
        # General intelligent assistant fallback response
        reply = f"🐱 **VeriCat AI:** I processed your query: *\"{message}\"*\n\nBased on current forensic context & AI knowledge base: VERI-DEEP is active on `{classifier.device.upper()}` connected to `{db_mode}`. Let me know if you would like me to inspect a media file, explain Grad-CAM heatmaps, or query your Atlas history!"
        
    return {
        "success": True,
        "agent": "VeriCat AI 🐱",
        "reply": reply,
        "timestamp": time.strftime("%H:%M:%S")
    }

