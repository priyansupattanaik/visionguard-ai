# 🛡️ VisionGuard AI

Advanced CCTV Incident Detection & Analysis System using YOLO11m, ByteTrack, and Qwen2.5-VL-3B.

## Features

- **Real-time Object Detection**: YOLO11m for high-accuracy detection
- **Multi-Object Tracking**: ByteTrack for consistent ID tracking
- **VLM Analysis**: Qwen2.5-VL-3B for incident understanding
- **Smart Clip Extraction**: Automatic relevant segment extraction
- **Structured Reporting**: HTML/PDF reports with timelines
- **Semantic Search**: Natural language incident search (ChromaDB-ready)
- **Interactive UI**: Gradio web interface with progress tracking

## Quick Start

### Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/visionguard-ai.git
cd visionguard-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```
