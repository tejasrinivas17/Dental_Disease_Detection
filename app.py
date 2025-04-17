import streamlit as st
from utils import load_model, predict_category
from PIL import Image

# --- CONFIGURE PAGE ---
st.set_page_config(
    page_title="Dental Disease Classifier",
    page_icon="🦷",
    layout="wide"
)

# --- LOAD MODEL ---
model = load_model('iitj_dental_cnn.pth')

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>🦷 Dental Disease Classifier</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- TEAM INFO ---
with st.container():
    st.markdown("""
    <div style='padding:10px; border: 2px solid #e1e4e8; border-radius: 10px; background-color: black;'>
        <h4>👨‍🏫 TEAM 17</h4>
        <ul>
            <li><b>M TEJA SRINIVAS</b> (21B21A4283)</li>
            <li><b>V SUSHANTH KUMAR</b> (21B21A4267)</li>
            <li><b>VVS VARA PRASAD</b> (21B21A4270)</li>
            <li><b>K CHENNA RAO</b> (21B21A4273)</li>
            <li><b>N SAI KRISHNA</b> (21B21A4268)</li>    
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- IMAGE UPLOAD ---
st.markdown("### 📤 Upload a dental image to get started:")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# --- DISPLAY RESULTS ---
if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🖼️ Uploaded Dental Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("#### 🔍 Predicted Condition")
        st.markdown("⏳ Running classification...")

        # Predict category
        prediction = predict_category(model, image)

        # Highlight result
        st.markdown(f"""
        <div style='padding:20px; border-radius:10px; background-color:#e6f4ea; border: 2px solid #34a853;'>
            <h3 style='color:#0b6623;'>✅ Prediction: <em>{prediction}</em></h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
else:
    st.info("Please upload a dental image to get a prediction.")

# --- FOOTER ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 0.9em;'>© 2025  | KIET CSM - TEAM-17</p>",
    unsafe_allow_html=True
)
