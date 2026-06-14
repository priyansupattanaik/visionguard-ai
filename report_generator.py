"""
Report generation module.
Creates structured PDF/HTML reports from incident data.
"""
from jinja2 import Template
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
from typing import List, Dict
import os


class ReportGenerator:
    def __init__(self, output_dir: str = "output/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        
    def generate_html_report(self, incidents: List[Dict], video_path: str,
                             metadata: Dict = None) -> str:
        """
        Generate an interactive HTML report.
        
        Args:
            incidents: List of incident dictionaries
            video_path: Original video path
            metadata: Additional info (location, camera_id, etc.)
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>VisionGuard AI - Incident Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
                .incident { background: white; padding: 20px; margin-bottom: 20px; 
                           border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .severity-high { border-left: 5px solid #e74c3c; }
                .severity-medium { border-left: 5px solid #f39c12; }
                .severity-low { border-left: 5px solid #27ae60; }
                .timestamp { color: #666; font-size: 0.9em; }
                .badge { padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: bold; }
                .badge-high { background: #fee; color: #c33; }
                .badge-medium { background: #ffeaa7; color: #d68910; }
                .badge-low { background: #d5f5e3; color: #27ae60; }
                .clip-info { background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background: #667eea; color: white; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🛡️ VisionGuard AI Incident Report</h1>
                <p>Generated: {{ generated_at }} | Video: {{ video_name }}</p>
                {% if metadata %}
                <p>Location: {{ metadata.get('location', 'N/A') }} | Camera: {{ metadata.get('camera_id', 'N/A') }}</p>
                {% endif %}
            </div>
            
            <h2>Summary</h2>
            <p>Total Incidents Detected: <strong>{{ incidents|length }}</strong></p>
            <p>High Severity: <strong>{{ high_count }}</strong> | 
               Medium: <strong>{{ medium_count }}</strong> | 
               Low: <strong>{{ low_count }}</strong></p>
            
            <h2>Incident Details</h2>
            {% for incident in incidents %}
            <div class="incident severity-{{ incident.severity }}">
                <h3>Incident #{{ loop.index }} 
                    <span class="badge badge-{{ incident.severity }}">{{ incident.severity.upper() }}</span>
                </h3>
                <p class="timestamp">⏱️ {{ incident.start_time }} - {{ incident.end_time }} 
                   (Duration: {{ incident.duration }}s)</p>
                <p><strong>Description:</strong> {{ incident.description }}</p>
                <p><strong>Objects Involved:</strong> {{ incident.objects | join(', ') }}</p>
                <p><strong>Track IDs:</strong> {{ incident.track_ids | join(', ') }}</p>
                
                {% if incident.frame_path %}
                <div class="clip-info">
                    <p>📸 Key Frame: {{ incident.frame_path }}</p>
                    {% if incident.clip_path %}
                    <p>🎬 Extracted Clip: {{ incident.clip_path }}</p>
                    {% endif %}
                </div>
                {% endif %}
            </div>
            {% endfor %}
            
            <h2>Timeline</h2>
            <table>
                <tr>
                    <th>Time</th>
                    <th>Incident</th>
                    <th>Severity</th>
                    <th>Objects</th>
                </tr>
                {% for incident in incidents %}
                <tr>
                    <td>{{ incident.start_time }}</td>
                    <td>{{ incident.description[:80] }}...</td>
                    <td><span class="badge badge-{{ incident.severity }}">{{ incident.severity }}</span></td>
                    <td>{{ incident.objects | join(', ') }}</td>
                </tr>
                {% endfor %}
            </table>
        </body>
        </html>
        """
        
        # Count severities
        high = sum(1 for i in incidents if i.get('severity') == 'high')
        medium = sum(1 for i in incidents if i.get('severity') == 'medium')
        low = sum(1 for i in incidents if i.get('severity') == 'low')
        
        template = Template(html_template)
        html_content = template.render(
            incidents=incidents,
            video_name=os.path.basename(video_path),
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metadata=metadata or {},
            high_count=high,
            medium_count=medium,
            low_count=low
        )
        
        output_path = os.path.join(self.output_dir, 
                                   f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return output_path
    
    def generate_pdf_report(self, incidents: List[Dict], video_path: str) -> str:
        """Generate a PDF report (simplified version)."""
        output_path = os.path.join(self.output_dir,
                                   f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30
        )
        story.append(Paragraph("VisionGuard AI Incident Report", title_style))
        story.append(Spacer(1, 12))
        
        # Summary table
        data = [['#', 'Time', 'Severity', 'Description', 'Objects']]
        for i, inc in enumerate(incidents[:50], 1):  # Limit to 50 for PDF
            data.append([
                str(i),
                f"{inc['start_time']:.1f}s",
                inc.get('severity', 'medium'),
                inc['description'][:60] + "...",
                ", ".join(inc.get('objects', []))[:30]
            ])
            
        table = Table(data, colWidths=[0.5*inch, 1*inch, 1*inch, 3*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        doc.build(story)
        
        return output_path