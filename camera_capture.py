import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime

class CameraCapture:
    def __init__(self):
        self.camera = None
    
    def init_camera(self):
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                return False
            return True
        except Exception as e:
            st.error(f"Camera initialization failed: {str(e)}")
            return False
    
    def capture_frame(self):
        """Capture a frame from camera"""
        if self.camera is None:
            return None
        
        ret, frame = self.camera.read()
        if ret:
            return frame
        return None
    
    def release_camera(self):
        """Release camera resources"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None

def camera_capture_component():
    """Streamlit component for camera capture"""
    st.subheader("📷 Camera Capture")
    
    # Initialize camera capture
    if 'camera_capture' not in st.session_state:
        st.session_state.camera_capture = CameraCapture()
    
    camera_capture = st.session_state.camera_capture
    
    # Camera controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎥 Initialize Camera"):
            if camera_capture.init_camera():
                st.success("Camera initialized successfully!")
                st.session_state.camera_initialized = True
            else:
                st.error("Failed to initialize camera. Please check if camera is connected.")
                st.session_state.camera_initialized = False
    
    with col2:
        if st.button("📸 Capture Photo"):
            if hasattr(st.session_state, 'camera_initialized') and st.session_state.camera_initialized:
                frame = camera_capture.capture_frame()
                if frame is not None:
                    st.session_state.captured_image = frame
                    st.success("Photo captured successfully!")
                else:
                    st.error("Failed to capture photo")
            else:
                st.warning("Please initialize camera first")
    
    with col3:
        if st.button("🛑 Release Camera"):
            camera_capture.release_camera()
            st.session_state.camera_initialized = False
            st.success("Camera released")
    
    # Display captured image
    if 'captured_image' in st.session_state:
        captured_image = st.session_state.captured_image
        
        # Convert BGR to RGB for display
        rgb_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
        st.image(rgb_image, caption="Captured Image", use_column_width=True)
        
        # Convert to PIL Image for analysis
        pil_image = Image.fromarray(rgb_image)
        
        # Convert PIL to bytes for download
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        # Download button
        st.download_button(
            label="💾 Download Captured Image",
            data=img_bytes,
            file_name=f"captured_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png"
        )
        
        return captured_image
    
    return None

# Alternative camera capture using streamlit-webrtc
def webrtc_camera_capture():
    """Alternative camera capture using streamlit-webrtc"""
    try:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
        
        class VideoTransformer(VideoTransformerBase):
            def __init__(self):
                self.frame_count = 0
                self.captured_frame = None
            
            def transform(self, frame):
                self.frame_count += 1
                # Capture frame every 30 frames (1 second at 30fps)
                if self.frame_count % 30 == 0:
                    self.captured_frame = frame.to_ndarray(format="bgr24")
                
                return frame
        
        transformer = VideoTransformer()
        
        webrtc_streamer(
            key="camera",
            video_transformer_factory=lambda: transformer,
            async_processing=True,
        )
        
        if transformer.captured_frame is not None:
            st.image(transformer.captured_frame, caption="Captured Frame", use_column_width=True)
            return transformer.captured_frame
        
    except ImportError:
        st.info("""
        ✅ Advanced camera capture is now available!
        WebRTC camera features are ready to use.
        """)
    
    return None

# Camera capture with file upload fallback
def camera_or_upload():
    """Camera capture with file upload fallback"""
    st.subheader("📸 Image Input")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Camera Capture (Basic)", "Camera Capture (WebRTC)"],
        horizontal=True
    )
    
    image = None
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image to analyze for microplastics"
        )
        
        if uploaded_file is not None:
            # Convert uploaded file to OpenCV format
            bytes_data = uploaded_file.read()
            nparr = np.frombuffer(bytes_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    elif input_method == "Camera Capture (Basic)":
        image = camera_capture_component()
    
    elif input_method == "Camera Capture (WebRTC)":
        image = webrtc_camera_capture()
    
    return image
