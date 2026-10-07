import subprocess
import time
import sys
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("LAUNCHING MULTIMODAL DEEPFAKE FORENSICS SYSTEM")
    print("=" * 60)

    base_dir = Path(__file__).resolve().parent

    # 1. Generate sample images
    try:
        from src.utils.generate_samples import create_sample_images
        create_sample_images()
    except Exception as e:
        print(f"Warning generating sample images: {e}")

    # 2. Launch FastAPI backend
    print("\n[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...")
    fastapi_cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]
    fastapi_proc = subprocess.Popen(fastapi_cmd, cwd=str(base_dir))

    time.sleep(3)

    # 3. Launch Streamlit Dashboard
    print("\n[2/2] Starting Streamlit Dashboard on http://localhost:8501 ...")
    streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"]
    streamlit_proc = subprocess.Popen(streamlit_cmd, cwd=str(base_dir))

    print("\n[SUCCESS] System running! Press Ctrl+C to terminate.")
    
    try:
        fastapi_proc.wait()
        streamlit_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        fastapi_proc.terminate()
        streamlit_proc.terminate()

if __name__ == "__main__":
    main()
