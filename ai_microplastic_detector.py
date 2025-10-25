import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import json
from datetime import datetime
import base64
from skimage.feature import local_binary_pattern
import tempfile
import os
import time
import requests
import openai
from typing import Dict, List, Optional

# Import camera capture functionality
from camera_capture import camera_or_upload

# Page configuration
st.set_page_config(
    page_title="AI-Powered Microplastic Detection Software",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .ai-header {
        font-size: 2rem;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .ai-card {
        background-color: #fff5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
    .ai-button {
        background-color: #ff6b6b !important;
    }
    .ai-button:hover {
        background-color: #ff5252 !important;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .ai-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

class AIAnalysisProvider:
    def __init__(self):
        self.providers = {
            'Grok AI': self.analyze_with_grok,
            'Local': self.analyze_with_local
        }
    
    def analyze_with_grok(self, results: Dict, api_key: str) -> str:
        """Analyze results using Grok AI"""
        try:
            # Use the correct Grok API endpoint
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = self.create_analysis_prompt(results)
            
            payload = {
                "model": "grok-2",
                "messages": [
                    {"role": "system", "content": "You are a microplastic detection expert. Provide detailed, scientific analysis of microplastic detection results."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                # Fallback to Local AI if Grok fails
                return self.analyze_with_local(results, None)
            
        except Exception as e:
            # Fallback to Local AI if Grok fails
            return self.analyze_with_local(results, None)
    
    
    def analyze_with_local(self, results: Dict, api_key: str = None) -> str:
        """Local analysis using predefined templates"""
        return self.create_local_analysis(results)
    
    def create_analysis_prompt(self, results: Dict) -> str:
        """Create a detailed prompt for AI analysis"""
        prompt = f"""
        Analyze the following microplastic detection results and provide a comprehensive scientific report:

        DETECTION SUMMARY:
        - Microplastics Detected: {'YES' if results['microplastics_detected'] else 'NO'}
        - Confidence Score: {results['confidence']:.3f}
        - Particle Count: {results['particle_count']}
        - Image Dimensions: {results['image_shape'][1]}x{results['image_shape'][0]} pixels

        DETECTED PARTICLES:
        """
        
        if results['particles']:
            for i, particle in enumerate(results['particles']):
                prompt += f"""
        Particle {i+1}:
        - Area: {particle['area']:.1f} pixels
        - Confidence: {particle['confidence']:.3f}
        - Color: {particle['color_name']}
        - Aspect Ratio: {particle['aspect_ratio']:.2f}
        - Circularity: {particle['circularity']:.2f}
        """
        else:
            prompt += "No particles detected."
        
        if 'feature_analysis' in results:
            prompt += f"""
        
        FEATURE ANALYSIS:
        """
            for key, value in results['feature_analysis'].items():
                if isinstance(value, dict):
                    prompt += f"- {key}: Mean={value.get('mean', 'N/A'):.2f}, Std={value.get('std', 'N/A'):.2f}\n"
                else:
                    prompt += f"- {key}: {value:.2f}\n"
        
        prompt += """
        
        Please provide:
        1. A detailed interpretation of the results
        2. Scientific significance of the findings
        3. Potential sources of microplastics
        4. Environmental implications
        5. Recommendations for further analysis
        6. Quality assessment of the detection
        """
        
        return prompt
    
    def create_local_analysis(self, results: Dict) -> str:
        """Create local analysis without API"""
        analysis = f"""
# 🔬 Microplastic Detection Analysis Report

## 📊 Detection Summary
- **Status**: {'✅ Microplastics Detected' if results['microplastics_detected'] else '❌ No Microplastics Detected'}
- **Confidence Level**: {results['confidence']:.1%}
- **Particle Count**: {results['particle_count']}
- **Image Resolution**: {results['image_shape'][1]} × {results['image_shape'][0]} pixels

## 🔍 Detailed Analysis

### Detection Quality Assessment
"""
        
        if results['confidence'] > 0.7:
            analysis += "**High Confidence Detection**: The analysis shows high reliability in microplastic identification.\n\n"
        elif results['confidence'] > 0.4:
            analysis += "**Medium Confidence Detection**: The analysis shows moderate reliability. Consider additional verification.\n\n"
        else:
            analysis += "**Low Confidence Detection**: The analysis shows low reliability. Manual verification recommended.\n\n"
        
        if results['particles']:
            analysis += "### Detected Particles\n"
            for i, particle in enumerate(results['particles']):
                analysis += f"""
**Particle {i+1}**:
- Size: {particle['area']:.1f} pixels ({self.size_category(particle['area'])})
- Color: {particle['color_name']} (common in {self.color_analysis(particle['color_name'])})
- Shape: Aspect ratio {particle['aspect_ratio']:.2f}, Circularity {particle['circularity']:.2f}
- Confidence: {particle['confidence']:.1%}
"""
            
            analysis += f"""
### Environmental Implications
- **Total Microplastic Load**: {results['particle_count']} particles detected
- **Size Distribution**: {self.size_distribution_analysis(results['particles'])}
- **Color Analysis**: {self.color_distribution_analysis(results['particles'])}

### Recommendations
1. **Verification**: {'High confidence - proceed with analysis' if results['confidence'] > 0.7 else 'Consider manual verification'}
2. **Sampling**: Collect additional samples for statistical analysis
3. **Source Identification**: Investigate potential sources based on particle characteristics
4. **Monitoring**: Implement regular monitoring protocols
"""
        else:
            analysis += """
### No Microplastics Detected
- The analysis did not identify any microplastic particles
- This could indicate:
  - Clean sample with no microplastics
  - Detection parameters may need adjustment
  - Sample preparation may require optimization

### Recommendations
1. **Parameter Adjustment**: Try adjusting detection sensitivity
2. **Sample Preparation**: Ensure proper sample preparation
3. **Verification**: Manual inspection recommended
"""
        
        return analysis
    
    def size_category(self, area: float) -> str:
        """Categorize particle size"""
        if area < 100:
            return "very small"
        elif area < 500:
            return "small"
        elif area < 2000:
            return "medium"
        else:
            return "large"
    
    def color_analysis(self, color: str) -> str:
        """Analyze color significance"""
        color_sources = {
            'Blue': 'polyethylene bottles, packaging',
            'Red': 'polypropylene containers, textiles',
            'Green': 'polyethylene bags, bottles',
            'White': 'polystyrene foam, packaging',
            'Yellow': 'polyethylene containers',
            'Purple': 'polyethylene bottles, packaging'
        }
        return color_sources.get(color, 'various plastic sources')
    
    def size_distribution_analysis(self, particles: List[Dict]) -> str:
        """Analyze size distribution"""
        if not particles:
            return "No particles to analyze"
        
        areas = [p['area'] for p in particles]
        avg_size = np.mean(areas)
        
        if avg_size < 200:
            return f"Predominantly small particles (avg: {avg_size:.1f} pixels)"
        elif avg_size < 1000:
            return f"Mixed size distribution (avg: {avg_size:.1f} pixels)"
        else:
            return f"Predominantly large particles (avg: {avg_size:.1f} pixels)"
    
    def color_distribution_analysis(self, particles: List[Dict]) -> str:
        """Analyze color distribution"""
        if not particles:
            return "No particles to analyze"
        
        colors = [p['color_name'] for p in particles]
        color_counts = pd.Series(colors).value_counts()
        
        dominant_color = color_counts.index[0]
        return f"Dominant color: {dominant_color} ({color_counts.iloc[0]} particles)"

class MicroplasticDetector:
    def __init__(self):
        self.detection_params = {
            'min_particle_size': 50,
            'max_particle_size': 10000,
            'color_sensitivity': 0.3,
            'texture_threshold': 50,
            'edge_threshold_low': 50,
            'edge_threshold_high': 150
        }
    
    def detect_microplastics(self, image):
        """Detect microplastics in the image"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'image_shape': image.shape,
            'microplastics_detected': False,
            'confidence': 0.0,
            'particle_count': 0,
            'particles': [],
            'analysis_methods': [],
            'feature_analysis': {}
        }
        
        # Preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Enhanced Edge Detection
        edges = cv2.Canny(gray, 
                         self.detection_params['edge_threshold_low'], 
                         self.detection_params['edge_threshold_high'])
        
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Color-based Detection
        detected_particles = []
        
        color_ranges = [
            (np.array([100, 50, 50]), np.array([130, 255, 255])),  # Blue
            (np.array([0, 50, 50]), np.array([10, 255, 255])),    # Red
            (np.array([170, 50, 50]), np.array([180, 255, 255])),  # Red (wrap)
            (np.array([40, 50, 50]), np.array([80, 255, 255])),    # Green
            (np.array([0, 0, 200]), np.array([180, 30, 255])),    # White
            (np.array([20, 50, 50]), np.array([30, 255, 255])),    # Yellow
            (np.array([130, 50, 50]), np.array([160, 255, 255]))   # Purple
        ]
        
        color_names = ['Blue', 'Red', 'Red', 'Green', 'White', 'Yellow', 'Purple']
        
        for i, ((lower, upper), color_name) in enumerate(zip(color_ranges, color_names)):
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if (self.detection_params['min_particle_size'] < area < 
                    self.detection_params['max_particle_size']):
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Extract color features
                    roi = image[y:y+h, x:x+w]
                    mean_color = np.mean(roi, axis=(0, 1))
                    
                    confidence = self.calculate_particle_confidence(
                        area, aspect_ratio, circularity, mean_color, i
                    )
                    
                    if confidence > self.detection_params['color_sensitivity']:
                        particle = {
                            'area': area,
                            'bbox': (x, y, w, h),
                            'aspect_ratio': aspect_ratio,
                            'circularity': circularity,
                            'color_range': i,
                            'color_name': color_name,
                            'mean_color': mean_color.tolist(),
                            'confidence': confidence,
                            'perimeter': perimeter
                        }
                        detected_particles.append(particle)
        
        # Texture Analysis
        try:
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            texture_variance = np.var(lbp)
            texture_mean = np.mean(lbp)
            
            results['feature_analysis']['texture_variance'] = texture_variance
            results['feature_analysis']['texture_mean'] = texture_mean
            
            if texture_variance > self.detection_params['texture_threshold']:
                results['analysis_methods'].append('texture_analysis')
                
        except ImportError:
            pass
        
        # Size Distribution Analysis
        if detected_particles:
            areas = [p['area'] for p in detected_particles]
            results['feature_analysis']['size_distribution'] = {
                'mean': np.mean(areas),
                'std': np.std(areas),
                'min': np.min(areas),
                'max': np.max(areas)
            }
        
        # Combine results
        results['particles'] = detected_particles
        results['particle_count'] = len(detected_particles)
        
        if detected_particles:
            avg_confidence = np.mean([p['confidence'] for p in detected_particles])
            results['confidence'] = avg_confidence
            results['microplastics_detected'] = avg_confidence > 0.4
        
        results['analysis_methods'].extend(['edge_detection', 'color_filtering', 'feature_extraction'])
        
        return results
    
    def calculate_particle_confidence(self, area, aspect_ratio, circularity, mean_color, color_range):
        """Calculate confidence score for a detected particle"""
        confidence = 0.0
        
        # Size-based confidence
        if 100 < area < 5000:
            confidence += 0.3
        elif 50 < area < 10000:
            confidence += 0.2
        
        # Shape-based confidence
        if 0.3 < aspect_ratio < 3.0:
            confidence += 0.2
        
        if circularity > 0.3:
            confidence += 0.2
        
        # Color-based confidence
        color_weights = [0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]
        if color_range < len(color_weights):
            confidence += color_weights[color_range]
        
        # Brightness confidence
        brightness = np.mean(mean_color)
        if 50 < brightness < 200:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def create_visualization(self, image, results):
        """Create visualization of detected particles"""
        vis_image = image.copy()
        
        for i, particle in enumerate(results['particles']):
            x, y, w, h = particle['bbox']
            confidence = particle['confidence']
            
            # Color based on confidence
            if confidence > 0.7:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.4:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Add confidence label
            label = f"{i+1}: {confidence:.2f}"
            cv2.putText(vis_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image

def make_json_serializable(obj):
    """Convert objects to JSON-serializable format"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (int, float, str)):
        return obj
    else:
        return str(obj)

def create_analysis_charts(results):
    """Create analysis charts"""
    if not results['particles']:
        return None, None, None
    
    particles = results['particles']
    areas = [p['area'] for p in particles]
    confidences = [p['confidence'] for p in particles]
    colors = [p['color_name'] for p in particles]
    
    # Size distribution histogram
    size_fig = px.histogram(
        x=areas,
        nbins=20,
        title="Particle Size Distribution",
        labels={'x': 'Area (pixels)', 'y': 'Count'},
        color_discrete_sequence=['#1f77b4']
    )
    size_fig.update_layout(showlegend=False)
    
    # Confidence vs Area scatter plot
    scatter_fig = px.scatter(
        x=areas,
        y=confidences,
        title="Particle Confidence vs Area",
        labels={'x': 'Area (pixels)', 'y': 'Confidence Score'},
        color=confidences,
        color_continuous_scale='RdYlGn'
    )
    
    # Color distribution pie chart
    color_counts = pd.Series(colors).value_counts()
    color_fig = px.pie(
        values=color_counts.values,
        names=color_counts.index,
        title="Particle Color Distribution"
    )
    
    return size_fig, scatter_fig, color_fig

def main():
    # Initialize detector and AI provider
    if 'detector' not in st.session_state:
        st.session_state.detector = MicroplasticDetector()
    
    if 'ai_provider' not in st.session_state:
        st.session_state.ai_provider = AIAnalysisProvider()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI-Powered Microplastic Detection Software</h1>', unsafe_allow_html=True)
    
    # Sidebar for parameters and AI configuration
    with st.sidebar:
        st.header("⚙️ Detection Parameters")
        
        min_size = st.slider("Minimum Particle Size", 10, 200, 50)
        max_size = st.slider("Maximum Particle Size", 1000, 50000, 10000)
        sensitivity = st.slider("Color Sensitivity", 0.1, 1.0, 0.3)
        texture_threshold = st.slider("Texture Threshold", 10, 200, 50)
        
        # Update detector parameters
        st.session_state.detector.detection_params.update({
            'min_particle_size': min_size,
            'max_particle_size': max_size,
            'color_sensitivity': sensitivity,
            'texture_threshold': texture_threshold
        })
        
        st.header("🤖 AI Analysis Configuration")
        
        st.header("🤖 Grok AI Analysis")
        
        # Grok AI Configuration
        grok_api_key = "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"
        
        # Show API key status
        st.success("✅ Grok AI - API key configured")
        st.info("Using Grok AI for advanced microplastic analysis")
        
        # AI Analysis Button
        if st.button("🤖 Generate Grok AI Analysis", type="primary"):
            if 'analysis_results' in st.session_state:
                with st.spinner("🤖 Grok AI is analyzing the results..."):
                    analysis = st.session_state.ai_provider.providers["Grok AI"](
                        st.session_state.analysis_results, 
                        grok_api_key
                    )
                    st.session_state.ai_analysis = analysis
            else:
                st.warning("Please analyze an image first!")
        
        st.header("📊 Analysis Methods")
        st.info("""
        **Detection Methods Used:**
        - Color-based filtering
        - Edge detection
        - Shape analysis
        - Texture analysis
        - Size filtering
        - AI-powered analysis
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Image Input")
        
        # Use camera capture component
        image = camera_or_upload()
        
        # Analysis button
        if image is not None:
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image for microplastics..."):
                    # Perform analysis
                    results = st.session_state.detector.detect_microplastics(image)
                    st.session_state.analysis_results = results
                    
                    # Create visualization
                    vis_image = st.session_state.detector.create_visualization(image, results)
                    st.session_state.visualization = vis_image
                    
                    # Show success message
                    if results['microplastics_detected']:
                        st.markdown("""
                        <div class="success-box">
                            <h4>✅ Microplastics Detected!</h4>
                            <p>Found {particle_count} particles with {confidence:.1%} confidence</p>
                        </div>
                        """.format(
                            particle_count=results['particle_count'],
                            confidence=results['confidence']
                        ), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ No Microplastics Detected</h4>
                            <p>The analysis did not find any microplastics in the image</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        st.header("📊 Analysis Results")
        
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            
            # Summary metrics
            col1_metric, col2_metric, col3_metric = st.columns(3)
            
            with col1_metric:
                st.metric(
                    "Microplastics Detected",
                    "YES" if results['microplastics_detected'] else "NO",
                    delta=None
                )
            
            with col2_metric:
                st.metric(
                    "Particle Count",
                    results['particle_count'],
                    delta=None
                )
            
            with col3_metric:
                st.metric(
                    "Confidence Score",
                    f"{results['confidence']:.3f}",
                    delta=None
                )
            
            # Display visualization
            if 'visualization' in st.session_state:
                vis_image = st.session_state.visualization
                # Convert BGR to RGB for display
                vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                st.image(vis_image_rgb, caption="Detection Results", use_column_width=True)
            
            # Detailed results
            st.subheader("📋 Detailed Analysis")
            
            if results['particles']:
                # Create particles dataframe
                particles_data = []
                for i, particle in enumerate(results['particles']):
                    particles_data.append({
                        'Particle #': i + 1,
                        'Area (pixels)': f"{particle['area']:.1f}",
                        'Confidence': f"{particle['confidence']:.3f}",
                        'Color': particle['color_name'],
                        'Aspect Ratio': f"{particle['aspect_ratio']:.2f}",
                        'Circularity': f"{particle['circularity']:.2f}"
                    })
                
                df = pd.DataFrame(particles_data)
                st.dataframe(df, use_container_width=True)
                
                # Analysis charts
                st.subheader("📈 Analysis Charts")
                
                size_fig, scatter_fig, color_fig = create_analysis_charts(results)
                
                if size_fig:
                    st.plotly_chart(size_fig, use_container_width=True)
                    st.plotly_chart(scatter_fig, use_container_width=True)
                    st.plotly_chart(color_fig, use_container_width=True)
                
            else:
                st.info("No microplastics detected in the image.")
            
            # Feature analysis
            if 'feature_analysis' in results and results['feature_analysis']:
                st.subheader("🔬 Feature Analysis")
                
                feature_col1, feature_col2 = st.columns(2)
                
                with feature_col1:
                    if 'texture_variance' in results['feature_analysis']:
                        st.metric(
                            "Texture Variance",
                            f"{results['feature_analysis']['texture_variance']:.2f}"
                        )
                
                with feature_col2:
                    if 'size_distribution' in results['feature_analysis']:
                        st.metric(
                            "Average Particle Size",
                            f"{results['feature_analysis']['size_distribution']['mean']:.1f} pixels"
                        )
            
            # Download results
            st.subheader("💾 Download Results")
            
            # JSON download
            json_data = json.dumps(make_json_serializable(results), indent=2)
            st.download_button(
                label="📄 Download JSON Results",
                data=json_data,
                file_name=f"microplastic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # CSV download
            if results['particles']:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV Results",
                    data=csv_data,
                    file_name=f"microplastic_particles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("Please upload an image or capture from camera, then click 'Analyze Image' to see results.")
    
    # AI Analysis Section
    if 'ai_analysis' in st.session_state:
        st.markdown("---")
        st.markdown('<h2 class="ai-header">🤖 AI-Powered Analysis</h2>', unsafe_allow_html=True)
        
        # Display AI analysis
        st.markdown(st.session_state.ai_analysis)
        
        # Download AI analysis - Multiple formats
        st.subheader("💾 Download AI Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📄 Download as Markdown",
                data=st.session_state.ai_analysis,
                file_name=f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        with col2:
            # Convert to plain text
            plain_text = st.session_state.ai_analysis.replace('#', '').replace('**', '').replace('*', '')
            st.download_button(
                label="📝 Download as Text",
                data=plain_text,
                file_name=f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col3:
            # Convert to HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI Microplastic Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #1f77b4; }}
                    h2 {{ color: #ff6b6b; }}
                    .metric {{ background-color: #f0f2f6; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                </style>
            </head>
            <body>
                {st.session_state.ai_analysis.replace('\n', '<br>').replace('#', '<h1>').replace('##', '<h2>')}
            </body>
            </html>
            """
            st.download_button(
                label="🌐 Download as HTML",
                data=html_content,
                file_name=f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
        
        # Additional download options
        st.markdown("---")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            # JSON format
            analysis_json = {
                "timestamp": datetime.now().isoformat(),
                "analysis": st.session_state.ai_analysis,
                "type": "AI Microplastic Analysis"
            }
            st.download_button(
                label="📊 Download as JSON",
                data=json.dumps(analysis_json, indent=2),
                file_name=f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col5:
            # CSV format (summary)
            csv_data = f"Analysis Type,Timestamp,Content\nAI Analysis,{datetime.now().isoformat()},\"{st.session_state.ai_analysis.replace('\"', '\"\"')}\""
            st.download_button(
                label="📈 Download as CSV",
                data=csv_data,
                file_name=f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col6:
            # DOC format (simple)
            doc_content = f"""
AI Microplastic Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{st.session_state.ai_analysis}
            """
            st.download_button(
                label="📋 Download as DOC",
                data=doc_content,
                file_name=f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.doc",
                mime="application/msword"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 AI-Powered Microplastic Detection Software | Built with Streamlit</p>
        <p>✅ Advanced camera features and AI analysis are now available!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
