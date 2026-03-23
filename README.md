# 🧠 MHS-Swin OCT Disease Classifier

This repository contains the **MHS-Swin model**, a state-of-the-art deep learning framework for **retinal OCT (Optical Coherence Tomography) image classification**. The model classifies OCT images into four categories:

- **CNV** – Choroidal Neovascularization  
- **DME** – Diabetic Macular Edema  
- **DRUSEN** – Retinal Drusen  
- **NORMAL** – Healthy retina  

The project integrates **Swin Transformer**, **CBAM (Convolutional Block Attention Module)**, and **DSPE (Depthwise Separable Patch Embedding)** modules for high accuracy and explainability. It also features **Grad-CAM visualizations**, multi-image uploads, and interactive Streamlit deployment.

---

## 🔗 Live Demo & Kaggle

- **Streamlit Demo:** [Click Here](https://zmwccrzstx4tktqxzydbgk.streamlit.app)  
- **Kaggle Notebook:** [Click Here](https://www.kaggle.com/code/reduanulislam0194/msh-swin)  

The live demo allows users to upload OCT images, view predictions, Grad-CAM heatmaps, and download visualizations.

---

## 🏗 Project Features

- **High-accuracy classification:** Uses a modified Swin Transformer with attention modules.  
- **Grad-CAM visualizations:** Highlights the regions of the retina influencing the model’s decision.  
- **Upload multiple images:** Batch classification support.  
- **Download Grad-CAM:** Save Grad-CAM images for research or reporting.  
- **True label vs Prediction:** Compare ground truth and predicted labels.  
- **Confusion Matrix:** Evaluate model performance with visual metrics.

---

## 📦 Repository Structure

```text
.
├── app.py                  # Streamlit application
├── model.py                # MHS-Swin model definition
├── utils.py                # Preprocessing & helper functions
├── gradcam.py              # Grad-CAM implementation
├── MHS_SWIN.pth            # Trained model weights (download separately)
├── README.md               # Project overview
└── requirements.txt        # Python dependencies
