"""
Gradio Web UI for VisionGuard AI.
Provides real-time upload, processing, and interactive results.
"""
import gradio as gr
import os
import cv2
import tempfile
from pipeline import VisionGuardPipeline
from typing import Tuple, List


def create_ui():
    """Create and launch the Gradio interface."""
    
    # Global pipeline instance (lazy loaded)
    pipeline = None
    
    def get_pipeline():
        nonlocal pipeline
        if pipeline is None:
            pipeline = VisionGuardPipeline()
        return pipeline
    
    def process_video(video_file, custom_prompt, target_class, progress=gr.Progress()):
        """
        Main processing handler.
        
        Args:
            video_file: Uploaded video path
            custom_prompt: User's custom analysis prompt
            target_class: Class filter (Person, Vehicle, All)
        """
        if video_file is None:
            return "⚠️ Please upload a video file.", None, None, None, None
        
        progress(0, desc="Initializing pipeline...")
        
        # Map class selection to COCO IDs
        class_map = {
            "Person Only": [0],
            "Vehicle Only": [2, 3, 5, 7],  # car, motorcycle, bus, truck
            "All Classes": None
        }
        
        try:
            pipe = get_pipeline()
            
            # Progress callback
            def update_progress(pct, msg):
                progress(min(pct / 100, 0.99), desc=msg)
            
            results = pipe.process_video(
                video_file,
                custom_prompt=custom_prompt if custom_prompt else None,
                progress_callback=update_progress,
                target_classes=class_map.get(target_class, None)
            )
            
            progress(1.0, desc="Complete!")
            
            # Format outputs
            incident_text = f"## 🛡️ Processing Complete\n\n"
            incident_text += f"**Total Incidents Detected:** {results['incident_count']}\n\n"
            
            if results['incidents']:
                for i, inc in enumerate(results['incidents'], 1):
                    incident_text += (
                        f"### Incident #{i} ({inc['severity'].upper()})\n"
                        f"**Time:** {inc['start_time']:.1f}s - {inc['end_time']:.1f}s "
                        f"(Duration: {inc['duration']:.1f}s)\n\n"
                        f"**Description:** {inc['description'][:300]}...\n\n"
                        f"**Objects:** {', '.join(inc['objects'])}\n"
                        f"**Tracks:** {', '.join(map(str, inc['track_ids']))}\n\n"
                        f"---\n\n"
                    )
            else:
                incident_text += "No incidents detected in this footage.\n"
            
            # Collect output files
            clips = results.get('clips', [])
            report_path = results.get('html_report', '')
            
            return (
                incident_text,
                report_path if os.path.exists(report_path) else None,
                clips[0] if clips else None,
                "\n".join(clips) if clips else "No clips extracted",
                results.get('json_data', '')
            )
            
        except Exception as e:
            return f"❌ Error: {str(e)}", None, None, None, None
    
    def search_incidents(query, history):
        """Semantic search across processed incidents."""
        if not query:
            return "Enter a search query"
        
        pipe = get_pipeline()
        results = pipe.semantic_search(query)
        
        if not results:
            return "No matching incidents found."
            
        output = f"Found {len(results)} matching incidents:\n\n"
        for inc in results:
            output += (
                f"- **{inc['start_time']:.1f}s**: {inc['description'][:150]}... "
                f"[{inc['severity']}]\n"
            )
        return output
    
    # UI Layout
    with gr.Blocks(title="VisionGuard AI - CCTV Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🛡️ VisionGuard AI
            ### Advanced CCTV Incident Detection & Analysis
            *Powered by YOLO11m + ByteTrack + Qwen2.5-VL-3B*
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input Panel
                gr.Markdown("## 📁 Input")
                video_input = gr.Video(
                    label="Upload CCTV Footage",
                    format="mp4"
                )
                
                prompt_input = gr.Textbox(
                    label="Custom Analysis Prompt (Optional)",
                    placeholder="E.g., 'Look for unauthorized access near the restricted zone'",
                    lines=3
                )
                
                class_filter = gr.Radio(
                    choices=["All Classes", "Person Only", "Vehicle Only"],
                    value="All Classes",
                    label="Detection Focus"
                )
                
                process_btn = gr.Button("🚀 Analyze Footage", variant="primary", size="lg")
                
                # Search Panel
                gr.Markdown("## 🔍 Semantic Search")
                search_input = gr.Textbox(
                    label="Search Incidents",
                    placeholder="E.g., 'person loitering near entrance'"
                )
                search_btn = gr.Button("Search")
                search_output = gr.Textbox(label="Results", lines=10)
                
            with gr.Column(scale=2):
                # Results Panel
                gr.Markdown("## 📊 Analysis Results")
                
                with gr.Tab("Incident Report"):
                    incident_output = gr.Markdown()
                
                with gr.Tab("HTML Report"):
                    report_file = gr.File(label="Download Report")
                
                with gr.Tab("Video Clips"):
                    clip_player = gr.Video(label="Incident Clip Preview")
                    clip_list = gr.Textbox(label="All Extracted Clips", lines=5)
                
                with gr.Tab("Raw Data"):
                    json_file = gr.File(label="Download JSON")
        
        # Event Handlers
        process_btn.click(
            fn=process_video,
            inputs=[video_input, prompt_input, class_filter],
            outputs=[incident_output, report_file, clip_player, clip_list, json_file]
        )
        
        search_btn.click(
            fn=search_incidents,
            inputs=[search_input, incident_output],
            outputs=search_output
        )
        
        gr.Markdown(
            """
            ---
            **Note:** First run will download YOLO11m (~40MB) and Qwen2.5-VL-3B (~6GB) models.
            Processing time depends on video length and GPU availability.
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    # Launch with public URL for Colab, local for development
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public URL for Colab
        debug=True,
        show_error=True
    )