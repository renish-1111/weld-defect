import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet50, ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Configure Streamlit page
st.set_page_config(
    page_title="🔥 Weld Defect Classifier",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B35;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-high {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    }
    .confidence-medium {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
    }
    .confidence-low {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
    }
</style>
""", unsafe_allow_html=True)

# Define the ensemble model class
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                outputs.append(torch.nn.functional.softmax(output, dim=1))
        # Average predictions
        ensemble_output = torch.mean(torch.stack(outputs), dim=0)
        return ensemble_output

@st.cache_resource
def load_ensemble_model():
    """Load the ensemble model from saved checkpoint"""
    try:
        # Check CUDA availability and set device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            st.success(f"🚀 CUDA detected! Using GPU: {torch.cuda.get_device_name(0)}")
            st.info(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device("cpu")
            st.info("💻 Using CPU for inference")
        
        # Load checkpoint
        checkpoint = torch.load('ensemble_weld_classifier.pth', map_location=device)
        
        # Validate checkpoint
        required_keys = ['class_names', 'efficientnet_state_dict', 'resnet50_state_dict', 'ensemble_accuracy']
        for key in required_keys:
            if key not in checkpoint:
                st.error(f"Missing key in checkpoint: {key}")
                return None, None, None, None

        # Load EfficientNet
        efficientnet_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        efficientnet_model.classifier[1] = nn.Linear(
            efficientnet_model.classifier[1].in_features, 
            len(checkpoint['class_names'])
        )
        efficientnet_model.load_state_dict(checkpoint['efficientnet_state_dict'])
        efficientnet_model = efficientnet_model.to(device)  # Move to GPU
        
        # Load ResNet50
        resnet50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        resnet50_model.fc = nn.Linear(
            resnet50_model.fc.in_features, 
            len(checkpoint['class_names'])
        )
        resnet50_model.load_state_dict(checkpoint['resnet50_state_dict'])
        resnet50_model = resnet50_model.to(device)  # Move to GPU
        
        # Create ensemble
        ensemble = EnsembleModel([efficientnet_model, resnet50_model])
        ensemble = ensemble.to(device)  # Move ensemble to GPU
        ensemble.eval()
        
        return ensemble, checkpoint['class_names'], checkpoint['ensemble_accuracy'], device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

# Define transforms for inference
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(model, image, class_names, device):
    """Predict the class of an image using GPU acceleration"""
    try:
        start_time = time.time()
        
        # Ensure image is valid
        if image is None:
            return None, None, None, None
        
        # Preprocess image
        input_tensor = inference_transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            if device.type == 'cuda':
                # Try with autocast, fallback if needed
                try:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_tensor)
                except:
                    outputs = model(input_tensor)
            else:
                outputs = model(input_tensor)
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        # Clean up GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        inference_time = time.time() - start_time
        predicted_class = class_names[predicted_idx]
        all_probs = probabilities[0].cpu().numpy()
        
        return predicted_class, confidence, all_probs, inference_time
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, None

def create_confidence_gauge(confidence):
    """Create a confidence gauge using Plotly"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_probability_chart(class_names, probabilities, predicted_class):
    """Create probability bar chart"""
    colors = ['#FF6B35' if cls == predicted_class else '#4ECDC4' for cls in class_names]
    
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=probabilities,
            marker_color=colors,
            text=[f'{prob:.1%}' for prob in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Weld Classes",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

# Main Streamlit App
def main():
    # Header
    st.markdown('<h1 class="main-header">🔥 Weld Defect Classifier</h1>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model, class_names, model_accuracy, device = load_ensemble_model()
    
    if model is None:
        st.error("❌ Failed to load the model. Please ensure 'ensemble_weld_classifier.pth' exists.")
        return
    
    # Sidebar
    st.sidebar.header("🔧 Model Info")
    device_info = "🚀 GPU" if device.type == 'cuda' else "💻 CPU"
    
    st.sidebar.info(f"""
    **Accuracy:** {model_accuracy:.1%}
    **Classes:** {len(class_names)}
    **Device:** {device_info}
    **Status:** Ready ✅
    """)
    
    st.sidebar.header("📋 Classes")
    class_descriptions = {
        'Bad Weld': '❌ Poor quality',
        'Good Weld': '✅ High quality',
        'Defect': '⚠️ Other defects'
    }
    
    for class_name in class_names:
        if class_name in class_descriptions:
            st.sidebar.write(f"**{class_name}:** {class_descriptions[class_name]}")
    
    # Main content
    tab1, tab2 = st.tabs(["🔥 Upload", "📊 About"])
    
    with tab1:
        st.header("🔥 Weld Classification")
        
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload weld images for classification"
        )
        
        # Single image processing
        if uploaded_files and len(uploaded_files) == 1:
            st.info("💡 Single image - detailed analysis")
            
            uploaded_file = uploaded_files[0]
            try:
                # Display uploaded image
                image = Image.open(uploaded_file).convert('RGB')
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📷 Image")
                    st.image(image, caption=f"{uploaded_file.name}", width=400)
                
                with col2:
                    st.subheader("🔮 Analysis")
                    
                    with st.spinner("🤖 Analyzing image..."):
                        result = predict_image(model, image, class_names, device)
                    
                    if result[0] is not None:
                        predicted_class, confidence, all_probs, inference_time = result
                        
                        # Prediction result card
                        if confidence > 0.8:
                            card_class = "confidence-high"
                            emoji = "🟢"
                        elif confidence > 0.6:
                            card_class = "confidence-medium" 
                            emoji = "🟡"
                        else:
                            card_class = "confidence-low"
                            emoji = "🔴"
                        
                        st.markdown(f"""
                        <div class="prediction-card {card_class}">
                            <h2>{emoji} {predicted_class}</h2>
                            <h3>{confidence:.1%} Confidence</h3>
                            <p>⚡ {inference_time*1000:.1f}ms</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Failed to process the image. Please try a different image.")
                        st.stop()
            
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.stop()
            
            # Show basic results only
            if 'predicted_class' in locals() and predicted_class is not None:
                st.header("� Results Summary")
                
                results_data = {
                    'Class': class_names,
                    'Probability': [f'{prob:.1%}' for prob in all_probs],
                    'Status': ['✅' if cls == predicted_class else '❌' for cls in class_names]
                }
                
                st.table(results_data)
        
        elif uploaded_files and len(uploaded_files) > 1:
            # Batch processing
            st.write(f"📁 Processing {len(uploaded_files)} images...")
            
            results = []
            progress_bar = st.progress(0)
            
            # Process each image
            total_time = 0
            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file).convert('RGB')
                result = predict_image(model, image, class_names, device)
                
                if result[0] is not None:
                    predicted_class, confidence, all_probs, inference_time = result
                    total_time += inference_time
                    
                    results.append({
                        'filename': uploaded_file.name,
                        'prediction': predicted_class,
                        'confidence': confidence,
                        'image': image,
                        'inference_time': inference_time
                    })
                else:
                    st.error(f"Failed to process {uploaded_file.name}")
                    continue
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Results grid
            st.header("🖼️ Results")
            
            cols = st.columns(min(3, len(results)))
            for i, result in enumerate(results):
                with cols[i % len(cols)]:
                    # Confidence indicator
                    if result['confidence'] > 0.8:
                        conf_emoji = "🟢"
                    elif result['confidence'] > 0.6:
                        conf_emoji = "🟡"
                    else:
                        conf_emoji = "🔴"
                    
                    st.image(result['image'], caption=f"{result['filename'][:15]}...", width=200)
                    st.write(f"{conf_emoji} **{result['prediction']}**")
                    st.write(f"{result['confidence']:.1%}")
            
            # Summary
            st.header("📊 Summary")
            
            # Create summary data
            summary_data = {}
            for class_name in class_names:
                count = sum(1 for r in results if r['prediction'] == class_name)
                summary_data[class_name] = count
            
            # Metrics
            confidences = [r['confidence'] for r in results]
            avg_confidence = np.mean(confidences)
            avg_inference_time = total_time / len(results)
            
            col7, col8 = st.columns([1, 1])
            with col7:
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            with col8:
                st.metric("Avg Time", f"{avg_inference_time*1000:.1f}ms")
            
            # Performance note
            if device.type == 'cuda':
                st.success(f"🚀 GPU: {len(results)} images in {total_time:.2f}s")
            else:
                st.info(f"💻 CPU: {len(results)} images in {total_time:.2f}s")
            
            summary_table = {
                'Class': list(summary_data.keys()),
                'Count': list(summary_data.values()),
                'Percentage': [f"{count/len(results)*100:.1f}%" for count in summary_data.values()]
            }
            st.table(summary_table)
        
        # Guidance when no files uploaded
        if not uploaded_files:
            st.info("📋 **Getting Started:**")
            col_guide1, col_guide2 = st.columns([1, 1])
            
            with col_guide1:
                st.markdown("""
                **🔥 Batch Processing:**
                - Upload multiple images
                - Get summary statistics
                - Efficient processing
                """)
            
            with col_guide2:
                st.markdown("""
                **📸 Single Image:**
                - Upload one image
                - Detailed analysis
                - Confidence gauges
                """)
            
            st.markdown("**💡 Tip:** Drop files above to start")
    
    with tab2:
        st.header("📊 About")
        
        st.markdown("""
        ### 🔧 Weld Defect Classification
        
        AI-powered weld quality assessment using deep learning.
        
        #### 🤖 **Model:**
        - Ensemble (EfficientNet-B0 + ResNet50)
        - Transfer learning from ImageNet
        - 77% accuracy on test data
        
        #### 🎯 **Features:**
        - Batch processing
        - Single image analysis
        - GPU acceleration
        - Real-time inference
        
        #### 🏷️ **Classes:**
        - **Bad Weld**: Poor quality welds
        - **Good Weld**: Acceptable quality
        - **Defect**: Other welding defects
        
        #### � **Tips:**
        - Upload clear, well-lit images
        - Multiple images for batch analysis
        - Higher confidence = more reliable
        """)

if __name__ == "__main__":
    main()
