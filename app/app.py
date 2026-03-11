import streamlit as st
import os
import sys
import tempfile
import shutil
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.inference.predict import TreeSegmentationPredictor
from src.utils.utils import visualize_prediction, calculate_tree_statistics

# Page configuration
st.set_page_config(
    page_title="Urban Tree Segmentation",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 0.5rem 0;
    }
    .upload-area {
        border: 2px dashed #2E8B57;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f9fff9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.predictor = None
    st.session_state.current_results = None

def load_model():
    """Load the tree segmentation model"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_model.pth')
        if not os.path.exists(model_path):
            st.error("Model not found! Please train the model first.")
            return False
        
        st.session_state.predictor = TreeSegmentationPredictor(model_path)
        st.session_state.model_loaded = True
        st.success("Model loaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False

def process_uploaded_image(uploaded_file):
    """Process uploaded image and return results"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        shutil.copyfileobj(uploaded_file, tmp_file)
        tmp_file_path = tmp_file.name
    
    try:
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process image
            results = st.session_state.predictor.predict_single_image(
                tmp_file_path, 
                temp_dir, 
                save_results_flag=True
            )
            
            # Load visualization
            viz_path = os.path.join(temp_dir, 'visualization.png')
            if os.path.exists(viz_path):
                visualization = Image.open(viz_path)
            else:
                visualization = None
            
            # Load tree mask
            mask_path = os.path.join(temp_dir, 'tree_mask.png')
            if os.path.exists(mask_path):
                tree_mask = np.array(Image.open(mask_path)) / 255.0
            else:
                tree_mask = None
            
            return results, visualization, tree_mask
    
    finally:
        # Clean up temporary file
        os.unlink(tmp_file_path)

def create_metrics_dashboard(results):
    """Create metrics dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌳 Total Trees</h3>
            <h2 style="color: #2E8B57;">{results['total_trees']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Tree Coverage</h3>
            <h2 style="color: #2E8B57;">{results['coverage_percentage']:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📏 Tree Area</h3>
            <h2 style="color: #2E8B57;">{results['total_tree_area_meters']:.2f} m²</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_tree_size = results['total_tree_area_meters'] / max(results['total_trees'], 1)
        st.markdown(f"""
        <div class="metric-card">
            <h3>📐 Avg Tree Size</h3>
            <h2 style="color: #2E8B57;">{avg_tree_size:.2f} m²</h2>
        </div>
        """, unsafe_allow_html=True)

def create_analysis_plots(results):
    """Create analysis plots"""
    if not results or 'tree_details' not in results:
        return
    
    tree_details = results['tree_details']
    
    if not tree_details:
        st.info("No trees detected in the image.")
        return
    
    # Extract tree areas
    tree_areas = [tree['area_meters'] for tree in tree_details]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tree size distribution
        fig = px.histogram(
            x=tree_areas,
            nbins=20,
            title="Tree Size Distribution",
            labels={"x": "Tree Area (m²)", "y": "Frequency"},
            color_discrete_sequence=['#2E8B57']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tree locations (if centroids available)
        if tree_details and 'centroid' in tree_details[0]:
            centroids = [tree['centroid'] for tree in tree_details]
            y_coords, x_coords = zip(*centroids) if centroids else ([], [])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='markers',
                marker=dict(size=8, color='#2E8B57'),
                name='Tree Locations'
            ))
            fig.update_layout(
                title="Tree Locations",
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application"""
    # Header
    st.markdown('<h1 class="main-header">🌳 Urban Tree Segmentation</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; color: #666;">
    Upload satellite imagery to automatically detect and analyze trees in urban areas.
    Get detailed insights about tree count, coverage area, and spatial distribution.
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Settings")
    
    # Model loading
    if not st.session_state.model_loaded:
        if st.sidebar.button("Load Model", type="primary"):
            load_model()
    else:
        st.sidebar.success("✅ Model Ready")
        
        if st.sidebar.button("Reload Model"):
            st.session_state.model_loaded = False
            st.session_state.predictor = None
    
    # City information
    st.sidebar.subheader("📍 Location Information")
    city_name = st.sidebar.text_input("City Name", placeholder="e.g., New York, London, Tokyo")
    
    # Main content
    if st.session_state.model_loaded:
        # File upload
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a satellite image...",
            type=['jpg', 'jpeg', 'png', 'tiff', 'tif'],
            help="Upload high-resolution satellite imagery for tree detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                
                # Process button
                if st.button("🔍 Analyze Trees", type="primary", use_container_width=True):
                    with st.spinner("Processing image... This may take a few moments."):
                        try:
                            results, visualization, tree_mask = process_uploaded_image(uploaded_file)
                            st.session_state.current_results = results
                            st.success("Analysis completed!")
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
            
            # Display results
            if st.session_state.current_results:
                results = st.session_state.current_results
                
                # Metrics dashboard
                st.subheader("📊 Analysis Results")
                create_metrics_dashboard(results)
                
                # Visualization
                if visualization:
                    st.subheader("🖼️ Tree Detection Visualization")
                    st.image(visualization, caption="Tree Segmentation Results", use_container_width=True)
                
                # Analysis plots
                st.subheader("📈 Detailed Analysis")
                create_analysis_plots(results)
                
                # Download results
                st.subheader("💾 Export Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download JSON results
                    results_json = json.dumps(results, indent=2, default=str)
                    st.download_button(
                        label="📄 Download JSON",
                        data=results_json,
                        file_name=f"tree_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Download tree mask
                    if tree_mask is not None:
                        mask_image = Image.fromarray((tree_mask * 255).astype(np.uint8))
                        mask_bytes = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                        mask_image.save(mask_bytes.name)
                        
                        with open(mask_bytes.name, 'rb') as f:
                            st.download_button(
                                label="🎭 Download Mask",
                                data=f.read(),
                                file_name=f"tree_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png"
                            )
                        os.unlink(mask_bytes.name)
                
                with col3:
                    # City report
                    if city_name:
                        city_report = {
                            'city': city_name,
                            'analysis_date': datetime.now().isoformat(),
                            'results': results
                        }
                        city_json = json.dumps(city_report, indent=2, default=str)
                        st.download_button(
                            label="🏙️ Download City Report",
                            data=city_json,
                            file_name=f"city_report_{city_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
    
    else:
        st.info("👆 Please load the model from the sidebar to start analyzing images.")
        
        # Instructions
        st.subheader("📖 How to Use")
        st.markdown("""
        1. **Load Model**: Click 'Load Model' in the sidebar to initialize the tree detection system
        2. **Upload Image**: Choose a satellite image from your device
        3. **Analyze**: Click 'Analyze Trees' to process the image
        4. **View Results**: Explore the detailed analysis and visualizations
        5. **Export**: Download results in various formats
        
        ### Supported Image Formats:
        - JPEG (.jpg, .jpeg)
        - PNG (.png)
        - TIFF (.tiff, .tif)
        
        ### Recommended Image Specifications:
        - Resolution: 512x512 pixels or higher
        - Format: High-quality satellite imagery
        - Coverage: Urban areas with visible trees
        """)
        
        # Sample results (for demonstration)
        st.subheader("🎯 Sample Analysis")
        sample_data = {
            'total_trees': 42,
            'coverage_percentage': 23.5,
            'total_tree_area_meters': 1250.8,
            'tree_details': [
                {'area_meters': 30.2, 'centroid': [100, 150]},
                {'area_meters': 45.8, 'centroid': [200, 250]},
                {'area_meters': 28.1, 'centroid': [300, 100]}
            ]
        }
        
        create_metrics_dashboard(sample_data)
        create_analysis_plots(sample_data)

if __name__ == "__main__":
    main()
