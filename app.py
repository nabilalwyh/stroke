# =========================
# 1. IMPORT
# =========================
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tensorflow.keras.backend as K

# =========================
# 2. CUSTOM LOSS (SAMA)
# =========================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def iou(y_true, y_pred):
    smooth = 1e-6
    intersection = K.sum(y_true * y_pred)
    return (intersection + smooth) / (K.sum(y_true + y_pred) - intersection + smooth)

# =========================
# 3. LOAD MODEL
# =========================
@st.cache_resource
def load_models():
    seg_model = tf.keras.models.load_model(
        "new2\seg_model (6).keras",
        custom_objects={
            'bce_dice_loss': bce_dice_loss,
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'iou': iou
        }
    )

    cls_model = tf.keras.models.load_model("new2\klas_model (6).keras")
    return seg_model, cls_model

seg_model, cls_model = load_models()

# =========================
# 4. LABEL
# =========================
class_labels = ["Normal", "Hemoragik", "Iskemik"]

# =========================
# 5. PREPROCESS (SAMA)
# =========================
def prepare_image(uploaded_file, size=(224, 224)):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(size)

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img, img_array

# =========================
# 6. UI
# =========================
st.title("🧠 Stroke Detection (Seg + Cls)")

uploaded_file = st.file_uploader("Upload CT Scan", type=["png", "jpg", "jpeg"])

# =========================
# 7. INFERENCE
# =========================
if uploaded_file is not None:

    img, img_array = prepare_image(uploaded_file)

    st.subheader("Original Image")
    st.image(img, use_container_width=True)

    # ======================
    # SEGMENTATION (FIX)
    # ======================
    raw_mask = seg_model.predict(img_array, verbose=0)[0]

    if raw_mask.shape[-1] == 1:
        raw_mask = raw_mask.squeeze()

    # ======================
    # FIX VISUAL (INI YANG BENER)
    # ======================
    mask_display = cv2.normalize(raw_mask, None, 0, 255, cv2.NORM_MINMAX)
    mask_display = mask_display.astype(np.uint8)

    # ======================
    # BINARY (SAMA KAYAK NOTEBOOK)
    # ======================
    pred_mask = (raw_mask > 0.3).astype(np.uint8)
    pred_mask = cv2.medianBlur(pred_mask, 5)

    # ======================
    # APPLY MASK
    # ======================
    masked_img = img_array[0] * np.expand_dims(pred_mask, axis=-1)

    if masked_img.shape[-1] == 1:
        masked_img = np.repeat(masked_img, 3, axis=-1)

    masked_img = cv2.resize(masked_img, (224, 224))
    masked_img = np.expand_dims(masked_img, axis=0)

    # ======================
    # CLASSIFICATION
    # ======================
    pred_cls = cls_model.predict(masked_img, verbose=0)

    idx = np.argmax(pred_cls)
    label = class_labels[idx]
    confidence = float(pred_cls[0][idx])

    # ======================
    # VISUALISASI (FIX)
    # ======================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img, caption="Original")

    with col2:
        # 🔥 PAKAI mask_display, BUKAN pred_mask
        st.image(mask_display, caption="Mask (Visible)", clamp=True)

    with col3:
        overlay = np.array(img)
        heatmap = cv2.applyColorMap(mask_display, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(overlay, 0.7, heatmap, 0.3, 0)
        st.image(overlay, caption="Overlay")

    # ======================
    # RESULT
    # ======================
    st.subheader("Prediction")

    if label == "Normal":
        st.success(label)
    else:
        st.error(label)

    st.write(f"Confidence: {confidence:.4f}")
    st.progress(float(confidence))
