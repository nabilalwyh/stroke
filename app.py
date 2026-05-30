import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tensorflow.keras.backend as K
import sqlite3
import pandas as pd
import os
import uuid
from datetime import datetime, date

# custom loss
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

# database
DB_NAME = "history_ct_scan.db"
SAVE_DIR = "history_images"

os.makedirs(SAVE_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            birth_date TEXT NOT NULL,
            upload_time TEXT NOT NULL,
            prediction_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            original_path TEXT NOT NULL,
            mask_path TEXT NOT NULL,
            overlay_path TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

def insert_history(patient_name, birth_date, prediction_label, confidence, original_path, mask_path, overlay_path):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO history (
            patient_name, birth_date, upload_time, prediction_label, confidence,
            original_path, mask_path, overlay_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_name,
        str(birth_date),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        prediction_label,
        confidence,
        original_path,
        mask_path,
        overlay_path
    ))

    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("""
        SELECT 
            id,
            patient_name,
            birth_date,
            upload_time,
            prediction_label,
            confidence,
            original_path,
            mask_path,
            overlay_path
        FROM history
        ORDER BY id DESC
    """, conn)
    conn.close()
    return df

def save_images(img, mask_display, overlay):
    file_id = uuid.uuid4().hex

    original_path = os.path.join(SAVE_DIR, f"{file_id}_original.png")
    mask_path = os.path.join(SAVE_DIR, f"{file_id}_mask.png")
    overlay_path = os.path.join(SAVE_DIR, f"{file_id}_overlay.png")

    img.save(original_path)

    mask_img = Image.fromarray(mask_display)
    mask_img.save(mask_path)

    overlay_img = Image.fromarray(overlay)
    overlay_img.save(overlay_path)

    return original_path, mask_path, overlay_path

init_db()

# load model
@st.cache_resource
def load_models():
    seg_model = tf.keras.models.load_model(
        "new2/seg_model (13).keras",
        custom_objects={
            'bce_dice_loss': bce_dice_loss,
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'iou': iou
        }
    )

    cls_model = tf.keras.models.load_model("new2/klas_model.keras")
    return seg_model, cls_model

seg_model, cls_model = load_models()

class_labels = ["Normal", "Hemoragik", "Iskemik"]

# data preprocessing
def prepare_image(uploaded_file, size=(224, 224)):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(size)

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img, img_array

# sidebar navigation
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Pilih Halaman",
    ["Upload CT Scan", "History CT Scan Pasien"]
)

# page 1: upload ct scan
if page == "Upload CT Scan":

    st.title("🧠 Stroke Detection")
    st.write("Sistem segmentasi dan klasifikasi penyakit stroke berdasarkan citra CT Scan otak.")

    st.subheader("Data Pasien")

    patient_name = st.text_input("Nama Pasien")
    birth_date = st.date_input(
        "Tanggal Lahir",
        value=date(2000, 1, 1),
        min_value=date(1900, 1, 1),
        max_value=date.today()
    )

    uploaded_file = st.file_uploader(
        "Upload CT Scan",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        if patient_name.strip() == "":
            st.warning("Nama pasien belum diisi.")
        else:
            img, img_array = prepare_image(uploaded_file)

            st.subheader("Original Image")
            st.image(img, use_container_width=True)

            # segmentasi
            raw_mask = seg_model.predict(img_array, verbose=0)[0]

            if raw_mask.shape[-1] == 1:
                raw_mask = raw_mask.squeeze()

            # mask untuk visualisasi
            mask_display = cv2.normalize(raw_mask, None, 0, 255, cv2.NORM_MINMAX)
            mask_display = mask_display.astype(np.uint8)

            # binary mask untuk input klasifikasi
            pred_mask = (raw_mask > 0.3).astype(np.uint8)
            pred_mask = cv2.medianBlur(pred_mask, 5)

            # spply mask ke original image
            masked_img = img_array[0] * np.expand_dims(pred_mask, axis=-1)

            if masked_img.shape[-1] == 1:
                masked_img = np.repeat(masked_img, 3, axis=-1)

            masked_img = cv2.resize(masked_img, (224, 224))
            masked_img = np.expand_dims(masked_img, axis=0)

            # klasifikasi
            pred_cls = cls_model.predict(masked_img, verbose=0)

            idx = np.argmax(pred_cls)
            label = class_labels[idx]
            confidence = float(pred_cls[0][idx])

            # overlay
            overlay = np.array(img)

            heatmap = cv2.applyColorMap(mask_display, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(overlay, 0.7, heatmap, 0.3, 0)

            # visualisasi hasil
            st.subheader("Hasil Segmentasi")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(img, caption="Original", use_container_width=True)

            with col2:
                st.image(mask_display, caption="Mask", clamp=True, use_container_width=True)

            with col3:
                st.image(overlay, caption="Overlay", use_container_width=True)

            # hasil
            st.subheader("Hasil Prediksi")

            st.write(f"Nama Pasien: **{patient_name}**")
            st.write(f"Tanggal Lahir: **{birth_date}**")

            if label == "Normal":
                st.success(f"Hasil Prediksi: {label}")
            else:
                st.error(f"Hasil Prediksi: {label}")

            st.write(f"Confidence: **{confidence:.4f}**")
            st.progress(float(confidence))

            # menyimpan history 
            if st.button("Simpan ke History"):
                original_path, mask_path, overlay_path = save_images(
                    img,
                    mask_display,
                    overlay
                )

                insert_history(
                    patient_name=patient_name,
                    birth_date=birth_date,
                    prediction_label=label,
                    confidence=confidence,
                    original_path=original_path,
                    mask_path=mask_path,
                    overlay_path=overlay_path
                )

                st.success("Data CT Scan pasien berhasil disimpan ke history.")

# page 2: history
elif page == "History CT Scan Pasien":

    st.title("📋 History CT Scan Pasien")
    st.write("Halaman ini menampilkan riwayat hasil pemeriksaan CT Scan pasien.")

    df_history = get_history()

    if df_history.empty:
        st.info("Belum ada data history yang tersimpan.")
    else:
        df_display = df_history[[
            "id",
            "patient_name",
            "birth_date",
            "upload_time",
            "prediction_label",
            "confidence"
        ]].copy()

        df_display["confidence"] = df_display["confidence"].apply(lambda x: f"{x:.4f}")

        st.subheader("Daftar History")
        st.dataframe(df_display, use_container_width=True)

        selected_id = st.selectbox(
            "Pilih ID untuk melihat detail",
            df_history["id"].tolist()
        )

        selected_data = df_history[df_history["id"] == selected_id].iloc[0]

        st.subheader("Detail Pemeriksaan")

        st.write(f"Nama Pasien: **{selected_data['patient_name']}**")
        st.write(f"Tanggal Lahir: **{selected_data['birth_date']}**")
        st.write(f"Waktu Upload: **{selected_data['upload_time']}**")
        st.write(f"Hasil Prediksi: **{selected_data['prediction_label']}**")
        st.write(f"Confidence: **{selected_data['confidence']:.4f}**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(selected_data["original_path"], caption="Original", use_container_width=True)

        with col2:
            st.image(selected_data["mask_path"], caption="Mask", use_container_width=True)

        with col3:
            st.image(selected_data["overlay_path"], caption="Overlay", use_container_width=True)