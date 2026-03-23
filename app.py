import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import MHSSwin
from utils import preprocess
from gradcam import GradCAM

# ======================
# CONFIG
# ======================
classes = ["CNV", "DME", "DRUSEN", "NORMAL"]

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    model = MHSSwin()
    model.load_state_dict(torch.load("MHS_SWIN.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

target_layer = model.cbam4.sa.conv
gradcam = GradCAM(model, target_layer)

# ======================
# SIDEBAR
# ======================
st.sidebar.title("📄 Model Info")
st.sidebar.markdown("""
### MHS-Swin Transformer

**Modules:**
- DSPE (Efficient embedding)
- CBAM (Attention refinement)
- APFH (Dual pooling head)

**Dataset:** OCT 2017  
**Classes:** CNV, DME, DRUSEN, NORMAL
""")

page = st.sidebar.selectbox("Select Page", ["🔍 Inference", "📊 Confusion Matrix"])

# ======================
# PAGE 1: INFERENCE
# ======================
if page == "🔍 Inference":

    st.title("🧠 OCT Disease Classifier")
    st.markdown("### Research Demo (MHS-Swin)")

    uploaded_files = st.file_uploader(
        "Upload OCT Images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for file in uploaded_files:

            st.markdown("---")
            image = Image.open(file).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)

            input_tensor = preprocess(image)

            # Prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                pred = np.argmax(probs)

            with col2:
                st.success(f"Prediction: {classes[pred]}")
                st.write(f"Confidence: {probs[pred]*100:.2f}%")

                st.markdown("### Probabilities")
                for i, c in enumerate(classes):
                    st.progress(float(probs[i]), text=f"{c}: {probs[i]*100:.2f}%")

            # ======================
            # TRUE LABEL (OPTIONAL)
            # ======================
            true_label = st.selectbox(
                f"Select True Label for {file.name}",
                ["Unknown"] + classes,
                key=file.name
            )

            if true_label != "Unknown":
                if classes[pred] == true_label:
                    st.success("✅ Correct Prediction")
                else:
                    st.error("❌ Wrong Prediction")

            # ======================
            # GRAD-CAM
            # ======================
            cam = gradcam.generate(input_tensor)[0].detach().cpu().numpy()
            cam = cv2.resize(cam, (224, 224))

            img_np = np.array(image.resize((224, 224))) / 255.0

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = heatmap / 255.0

            overlay = 0.6 * img_np + 0.4 * heatmap
            overlay = np.clip(overlay, 0, 1)

            st.markdown("### 🔍 Grad-CAM Visualization")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.image(img_np, caption="Original")

            with c2:
                st.image(cam, caption="Attention")

            with c3:
                st.image(overlay, caption="Overlay")

            # ======================
            # DOWNLOAD BUTTON
            # ======================
            overlay_uint8 = np.uint8(255 * overlay)
            _, buffer = cv2.imencode(".png", overlay_uint8)

            st.download_button(
                label="⬇ Download Grad-CAM",
                data=buffer.tobytes(),
                file_name=f"{file.name}_gradcam.png",
                mime="image/png"
            )

# ======================
# PAGE 2: CONFUSION MATRIX
# ======================
elif page == "📊 Confusion Matrix":

    st.title("📊 Confusion Matrix (Manual Evaluation)")

    st.write("Upload images and select true labels to build confusion matrix.")

    uploaded_files = st.file_uploader(
        "Upload labeled images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        key="cm"
    )

    if uploaded_files:

        y_true = []
        y_pred = []

        for file in uploaded_files:
            image = Image.open(file).convert("RGB")
            input_tensor = preprocess(image)

            with torch.no_grad():
                outputs = model(input_tensor)
                pred = torch.argmax(outputs, dim=1).item()

            true_label = st.selectbox(
                f"True label for {file.name}",
                classes,
                key=f"cm_{file.name}"
            )

            y_true.append(classes.index(true_label))
            y_pred.append(pred)

        if st.button("Generate Confusion Matrix"):

            cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])

            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            disp.plot(ax=ax)

            st.pyplot(fig)