"""
Clip extraction and generation module.
Extracts relevant video segments based on timestamps or incident triggers.
"""
import cv2
import os
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
import numpy as np


class ClipGenerator:
    def __init__(self, output_dir: str = "output/clips"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_clip(self, video_path: str, start_time: float, end_time: float,
                     output_name: str = None, padding: float = 2.0) -> str:
        """
        Extract a video clip from start_time to end_time with padding.
        
        Args:
            video_path: Source video file
            start_time: Start time in seconds
            end_time: End time in seconds
            padding: Extra seconds before/after
            output_name: Custom filename
            
        Returns:
            Path to extracted clip
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_sec = max(0, start_time - padding)
        end_sec = min(
            cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps,
            end_time + padding
        )
        
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"clip_{timestamp}_{start_sec:.1f}_{end_sec:.1f}.mp4"
            
        output_path = os.path.join(self.output_dir, output_name)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1
            
        cap.release()
        out.release()
        
        return output_path
    
    def extract_incident_clips(self, video_path: str, incidents: List[Dict],
                                padding: float = 3.0) -> List[str]:
        """
        Extract clips for multiple incidents.
        
        Args:
            incidents: List of {start_time, end_time, description, severity}
            
        Returns:
            List of extracted clip paths
        """
        clip_paths = []
        for i, incident in enumerate(incidents):
            output_name = f"incident_{i+1}_{incident.get('severity', 'medium')}.mp4"
            path = self.extract_clip(
                video_path,
                incident["start_time"],
                incident["end_time"],
                output_name=output_name,
                padding=padding
            )
            clip_paths.append({
                "path": path,
                "incident": incident,
                "duration": incident["end_time"] - incident["start_time"] + (padding * 2)
            })
        return clip_paths
    
    def create_highlight_reel(self, video_path: str, incidents: List[Dict],
                              output_name: str = "highlight_reel.mp4") -> str:
        """
        Create a compilation of all incidents with title cards.
        """
        # Implementation for concatenating clips with transitions
        # (Simplified version - uses ffmpeg if available, else sequential write)
        clips = self.extract_incident_clips(video_path, incidents, padding=2.0)
        
        # For now, return the list; full concatenation requires ffmpeg-python
        return clips