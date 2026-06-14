"""
Main processing pipeline.
Orchestrates detection → tracking → VLM analysis → incident detection → reporting.
"""
import cv2
import os
import json
from typing import List, Dict, Optional, Callable
from datetime import datetime
import numpy as np
from tqdm import tqdm

from detector import ObjectDetector
from tracker import ObjectTracker
from vlm import VLMAnalyzer
from clip_generator import ClipGenerator
from report_generator import ReportGenerator


class VisionGuardPipeline:
    def __init__(self, 
                 yolo_model: str = "yolo11m.pt",
                 vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 output_dir: str = "output",
                 process_every_n_frames: int = 5,
                 vlm_analyze_every_n_seconds: float = 2.0):
        """
        Initialize the complete pipeline.
        
        Args:
            process_every_n_frames: Run detection every N frames (speed optimization)
            vlm_analyze_every_n_seconds: Run VLM analysis every N seconds
        """
        self.output_dir = output_dir
        self.process_every_n = process_every_n_frames
        self.vlm_every_sec = vlm_analyze_every_n_seconds
        
        os.makedirs(f"{output_dir}/frames", exist_ok=True)
        os.makedirs(f"{output_dir}/clips", exist_ok=True)
        os.makedirs(f"{output_dir}/reports", exist_ok=True)
        
        print("[Pipeline] Initializing components...")
        self.detector = ObjectDetector(yolo_model)
        self.tracker = ObjectTracker(yolo_model)
        self.vlm = VLMAnalyzer(vlm_model)
        self.clip_gen = ClipGenerator(f"{output_dir}/clips")
        self.report_gen = ReportGenerator(f"{output_dir}/reports")
        
        self.incidents = []
        self.frame_cache = []
        self.tracking_history = {}  # track_id -> list of positions
        
    def process_video(self, video_path: str, 
                      custom_prompt: str = None,
                      progress_callback: Callable = None,
                      target_classes: List[int] = None) -> Dict:
        """
        Process a complete video file.
        
        Args:
            video_path: Path to CCTV footage
            custom_prompt: Custom VLM prompt for analysis
            progress_callback: Function(progress_pct, status_msg)
            target_classes: Filter specific COCO classes (e.g., [0] for person only)
            
        Returns:
            Processing results dictionary
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"[Pipeline] Video: {video_path}")
        print(f"[Pipeline] Duration: {duration:.1f}s, FPS: {fps}, Frames: {total_frames}")
        
        frame_idx = 0
        last_vlm_time = -999
        
        # Incident detection state
        current_incident = None
        incident_start = None
        
        for frame_idx in tqdm(range(total_frames), desc="Processing"):
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_idx / fps
            
            # Progress update
            if progress_callback and frame_idx % 30 == 0:
                progress_callback((frame_idx / total_frames) * 100, 
                                f"Processing frame {frame_idx}/{total_frames}")
            
            # Detection & Tracking (every N frames for speed)
            if frame_idx % self.process_every_n == 0:
                tracks = self.tracker.track(frame, classes=target_classes)
                
                # Update tracking history
                for track in tracks:
                    tid = track["track_id"]
                    if tid not in self.tracking_history:
                        self.tracking_history[tid] = []
                    self.tracking_history[tid].append({
                        "frame": frame_idx,
                        "time": timestamp,
                        "bbox": track["bbox"],
                        "class": track["class_name"]
                    })
                
                # Save keyframe if significant activity
                if len(tracks) > 0 and frame_idx % (fps * 2) == 0:  # Every 2 seconds if activity
                    frame_path = f"{self.output_dir}/frames/frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(frame_path, frame)
            
            # VLM Analysis (every N seconds)
            if timestamp - last_vlm_time >= self.vlm_every_sec:
                last_vlm_time = timestamp
                
                # Prepare prompt
                prompt = custom_prompt or (
                    "Analyze this CCTV frame. Identify: 1) What is happening, "
                    "2) Any suspicious or unsafe activities, 3) Number of people/vehicles, "
                    "4) Any incidents requiring attention. Be concise."
                )
                
                try:
                    analysis = self.vlm.analyze_frame(frame, prompt)
                    
                    # Simple incident detection heuristic
                    incident_keywords = ["suspicious", "unauthorized", "theft", "fight", 
                                       "fall", "intruder", "loitering", "crowd", "panic",
                                       "weapon", "violence", "emergency", "unconscious"]
                    
                    is_incident = any(kw in analysis.lower() for kw in incident_keywords)
                    
                    if is_incident:
                        if current_incident is None:
                            current_incident = {
                                "start_time": timestamp,
                                "description": analysis,
                                "severity": "medium",
                                "frames": [],
                                "track_ids": set(),
                                "objects": set()
                            }
                            incident_start = timestamp
                        else:
                            current_incident["description"] += f" | {analysis}"
                            
                        # Capture frame
                        frame_path = f"{self.output_dir}/frames/incident_{len(self.incidents):03d}_{frame_idx:06d}.jpg"
                        cv2.imwrite(frame_path, frame)
                        current_incident["frames"].append(frame_path)
                        
                        # Add tracks info
                        for t in tracks:
                            current_incident["track_ids"].add(t["track_id"])
                            current_incident["objects"].add(t["class_name"])
                            
                    else:
                        # End current incident if gap > 5 seconds
                        if current_incident and (timestamp - incident_start > 5):
                            current_incident["end_time"] = timestamp
                            current_incident["duration"] = timestamp - current_incident["start_time"]
                            current_incident["track_ids"] = list(current_incident["track_ids"])
                            current_incident["objects"] = list(current_incident["objects"])
                            
                            # Severity classification
                            if any(kw in current_incident["description"].lower() 
                                  for kw in ["weapon", "violence", "emergency", "unconscious"]):
                                current_incident["severity"] = "high"
                            elif current_incident["duration"] > 30:
                                current_incident["severity"] = "high"
                                
                            self.incidents.append(current_incident)
                            current_incident = None
                            
                except Exception as e:
                    print(f"[Pipeline] VLM error at {timestamp:.1f}s: {e}")
                    continue
        
        # Close final incident
        if current_incident:
            current_incident["end_time"] = timestamp
            current_incident["duration"] = timestamp - current_incident["start_time"]
            current_incident["track_ids"] = list(current_incident["track_ids"])
            current_incident["objects"] = list(current_incident["objects"])
            self.incidents.append(current_incident)
            
        cap.release()
        
        # Generate outputs
        print(f"[Pipeline] Detected {len(self.incidents)} incidents")
        
        # Extract clips
        clips = []
        if self.incidents:
            clips = self.clip_gen.extract_incident_clips(video_path, self.incidents)
            for i, clip in enumerate(clips):
                self.incidents[i]["clip_path"] = clip["path"]
        
        # Generate reports
        html_report = self.report_gen.generate_html_report(
            self.incidents, video_path
        )
        
        # Save JSON data
        json_path = f"{self.output_dir}/incidents.json"
        with open(json_path, 'w') as f:
            json.dump({
                "video": video_path,
                "processed_at": datetime.now().isoformat(),
                "total_incidents": len(self.incidents),
                "incidents": self.incidents
            }, f, indent=2, default=str)
            
        return {
            "incidents": self.incidents,
            "incident_count": len(self.incidents),
            "html_report": html_report,
            "json_data": json_path,
            "clips": [c["path"] for c in clips],
            "tracking_history": self.tracking_history
        }
    
    def semantic_search(self, query: str) -> List[Dict]:
        """
        Search incidents by semantic description (placeholder for ChromaDB integration).
        """
        # Future: Integrate ChromaDB with sentence-transformers embeddings
        results = []
        for inc in self.incidents:
            if query.lower() in inc["description"].lower():
                results.append(inc)
        return results