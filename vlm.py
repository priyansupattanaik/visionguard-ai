"""
Vision Language Model for incident understanding.
Uses Qwen2.5-VL-3B-Instruct for scene analysis and captioning.
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
from typing import Union, List, Dict
import gc


class VLMAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        """
        Initialize Qwen2.5-VL-3B-Instruct.
        Optimized for T4 GPU (16GB) with 4-bit quantization support.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[VLM] Loading {model_name} on {self.device}...")
        
        # Load with automatic device mapping for T4 compatibility
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        print("[VLM] Model loaded successfully")
        
    def analyze_frame(self, image: Union[np.ndarray, Image.Image], 
                      prompt: str = "Describe what is happening in this CCTV footage. Focus on any suspicious activities, people, vehicles, or safety concerns.") -> str:
        """
        Analyze a single frame/image with custom prompt.
        
        Args:
            image: numpy array (BGR) or PIL Image
            prompt: Custom query for the VLM
            
        Returns:
            Generated text description
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Clear cache to prevent OOM on T4
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            
        return output_text.strip()
    
    def analyze_batch(self, images: List[Image.Image], 
                      prompt: str = "Describe what is happening. Identify any incidents or suspicious behavior.") -> List[str]:
        """Analyze multiple frames (use sparingly to avoid OOM)."""
        results = []
        for img in images:
            results.append(self.analyze_frame(img, prompt))
        return results