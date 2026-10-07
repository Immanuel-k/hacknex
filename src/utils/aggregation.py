import numpy as np
from typing import List, Dict, Any

def aggregate_video_scores(frame_results: List[Dict[str, Any]], top_k_ratio: float = 0.3) -> Dict[str, Any]:
    """
    Aggregates frame-level fake probabilities into a robust video-level score.
    Formula: P_video = 0.5 * mean(P_i) + 0.5 * mean(top_k(P_i))
    """
    if not frame_results:
        return {
            "fake_probability": 0.0,
            "authenticity_score": 1.0,
            "decision": "likely_authentic",
            "suspicious_timestamps": [],
            "evidence_signals": ["No facial regions detected in processed media."]
        }

    scores = [f["fake_score"] for f in frame_results]
    n_frames = len(scores)
    
    mean_all = float(np.mean(scores))

    # Top-K suspicious frames calculation
    k = max(1, int(n_frames * top_k_ratio))
    top_k_scores = sorted(scores, reverse=True)[:k]
    mean_top_k = float(np.mean(top_k_scores))

    # Video score formula
    video_fake_score = round(0.5 * mean_all + 0.5 * mean_top_k, 4)
    authenticity_real_score = round(1.0 - video_fake_score, 4)

    # Decision Threshold
    if video_fake_score >= 0.65:
        decision = "likely_manipulated"
    elif video_fake_score >= 0.35:
        decision = "uncertain"
    else:
        decision = "likely_authentic"

    # Identify suspicious timestamp range
    suspicious_frames = [f for f in frame_results if f["fake_score"] > 0.5]
    suspicious_timestamps = []
    if suspicious_frames:
        t_start = min(f["timestamp"] for f in suspicious_frames)
        t_end = max(f["timestamp"] for f in suspicious_frames)
        suspicious_timestamps.append({
            "start_time_sec": t_start,
            "end_time_sec": t_end,
            "peak_fake_score": max(f["fake_score"] for f in suspicious_frames)
        })

    # Collect unique evidence signals across top frames
    all_signals = []
    for f in frame_results:
        for sig in f.get("signals", []):
            if sig not in all_signals:
                all_signals.append(sig)

    return {
        "fake_probability": video_fake_score,
        "authenticity_score": authenticity_real_score,
        "decision": decision,
        "mean_frame_score": round(mean_all, 4),
        "top_k_frame_score": round(mean_top_k, 4),
        "suspicious_timestamps": suspicious_timestamps,
        "evidence_signals": all_signals[:5]  # Top 5 unique evidence points
    }
