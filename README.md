# 🔥 Weld Defect Classifier - Streamlit Web App

A powerful AI-powered web application for classifying weld defects using state-of-the-art deep learning models.

## 🚀 Features

- **🎯 Single Image Classification**: Upload and classify individual weld images
- **🔥 Batch Processing**: Process multiple images at once
- **📊 Interactive Visualizations**: Beautiful charts and confidence metrics
- **🤖 Advanced AI**: Ensemble model combining EfficientNet-B0 + ResNet50
- **🎨 Modern UI**: Clean, responsive web interface

## 🏷️ Classification Classes

- **❌ Bad Weld**: Poor quality welds requiring rework
- **✅ Good Weld**: Acceptable quality welds  
- **⚠️ Defect**: Other types of welding defects

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- The trained model file: `ensemble_weld_classifier.pth`

### Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   
   **Option A: Using the batch file (Windows)**
   ```bash
   run_app.bat
   ```
   
   **Option B: Command line**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open in Browser**:
   - The app will automatically open at: http://localhost:8501
   - If it doesn't open automatically, click the link in the terminal

## 📱 How to Use

### Single Image Classification
1. Click on the **"📸 Single Image"** tab
2. Upload a weld image (JPG, JPEG, PNG)
3. View the AI prediction with confidence scores
4. Analyze detailed probability breakdowns

### Batch Processing
1. Click on the **"🔥 Batch Upload"** tab  
2. Upload multiple weld images
3. View grid results with color-coded confidence
4. Check summary statistics and distributions

## 🎯 Model Performance

- **Architecture**: Ensemble (EfficientNet-B0 + ResNet50)
- **Training**: Transfer learning from ImageNet
- **Accuracy**: High-performance ensemble model
- **Features**: Data augmentation, class balancing, early stopping

## 📊 Output Interpretation

### Confidence Levels:
- 🟢 **High (>80%)**: Very reliable prediction
- 🟡 **Medium (60-80%)**: Good prediction confidence
- 🔴 **Low (<60%)**: Lower confidence, review recommended

### Visualizations:
- **Confidence Gauge**: Overall prediction certainty
- **Probability Chart**: Breakdown by each class
- **Batch Statistics**: Distribution analysis for multiple images

## 🔧 Technical Details

### Dependencies:
- `streamlit`: Web app framework
- `torch`: PyTorch deep learning
- `torchvision`: Computer vision models
- `plotly`: Interactive visualizations
- `PIL`: Image processing
- `opencv-python`: Image handling

### Model Files:
- `ensemble_weld_classifier.pth`: Main ensemble model
- `best_weld_classifier.pth`: Best single model
- `weld_defect_classifier.pth`: Original baseline model

## 📁 File Structure

```
weld defect/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── run_app.bat              # Windows launcher script
├── README.md                # This file
├── ensemble_weld_classifier.pth  # Trained AI model
└── Weld_Classification.ipynb    # Original training notebook
```

## 🎉 Success! You now have a professional web application for weld defect classification!

## 🚨 Troubleshooting

### Common Issues:

1. **Model file not found**:
   - Ensure `ensemble_weld_classifier.pth` is in the same directory
   - Run the Jupyter notebook to train and save the model

2. **Import errors**:
   - Run: `pip install -r requirements.txt`
   - Ensure you're using Python 3.8+

3. **Port already in use**:
   - Stop other Streamlit apps or use: `streamlit run streamlit_app.py --server.port 8502`

4. **Images not loading**:
   - Ensure images are in supported formats (JPG, JPEG, PNG)
   - Check image file sizes (very large images may take longer)

## 📧 Support

For technical support or questions about the model, refer to the original Jupyter notebook `Weld_Classification.ipynb` which contains the complete training pipeline and detailed explanations.

---

**Powered by PyTorch, Streamlit, and Computer Vision** 🔥
