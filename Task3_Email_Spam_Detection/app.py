# =========================================
# Email Spam Detection App
# =========================================

import os
import logging
import streamlit as st
import pandas as pd
from src.predict import predict_message

# Paths to Model & Vectorizer
MODEL_PATH = r"C:\Users\ayesh\Projects\email_spam_detection\models\svm_spam_classifier.pkl"
VECTORIZER_PATH = r"C:\Users\ayesh\Projects\email_spam_detection\models\tfidf_vectorizer.pkl"

# Logging Setup
LOG_DIR = r"C:\Users\ayesh\Projects\email_spam_detection\logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent Streamlit from printing logs to terminal

# File handler
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("Email Spam Detection App Started")

# Streamlit Page Config
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Page Title
st.title("📧 Email Spam Detection Application")
st.markdown(
    "<p style='text-align:center;color:gray'>Predict whether an email or message is Spam or Ham</p>",
    unsafe_allow_html=True
)

# Sidebar - Instructions
with st.sidebar:
    st.header("📧 Email Spam Detection")
    st.markdown("---")
    st.subheader("Instructions")
    st.write(
        """
        1. Enter a single message in the text box or upload a CSV file for batch prediction.
        2. Click **Predict** to see the results.
        3. Check the logs in `logs/project.log` for detailed info.
        """
    )
    st.markdown("---")
    st.markdown("**Developed by Data Scientist**")

# Tabs for Single & Batch Prediction
tabs = st.tabs(["Single Message Prediction", "Batch Prediction"])

# Single Message Prediction
with tabs[0]:
    st.subheader("📝 Single Message Prediction")
    user_input = st.text_area(
        "Enter your message here:",
        placeholder="Type your message..."
    )

    if st.button("Predict Message"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a message to predict!")
        else:
            try:
                label = predict_message(user_input, MODEL_PATH, VECTORIZER_PATH, logger)
                if label == "Spam":
                    st.error(f"⚠️ Prediction: {label}")
                else:
                    st.success(f"✅ Prediction: {label}")
                logger.info(f"Single message predicted: {user_input} -> {label}")
            except FileNotFoundError as e:
                st.error(f"❌ File not found: {e}")
                logger.error(f"FileNotFoundError: {e}")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {e}")
                logger.exception("Prediction error")

# Batch Prediction
with tabs[1]:
    st.subheader("📊 Batch Prediction via CSV")
    uploaded_file = st.file_uploader("Upload CSV file with a 'text' column", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("❌ CSV must have a 'text' column")
            else:
                df["prediction"] = df["text"].apply(
                    lambda x: predict_message(x, MODEL_PATH, VECTORIZER_PATH, logger)
                )
                st.success("✅ Batch prediction completed!")
                st.dataframe(df.head(20))
                
                # Download CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
                
                logger.info(f"Batch prediction completed on {len(df)} messages")
        except Exception as e:
            st.error(f"❌ Error processing CSV: {e}")
            logger.exception("Batch prediction error")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center'>📌 Email Spam Detection | Data Science Project</p>",
    unsafe_allow_html=True
)
