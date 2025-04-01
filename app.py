import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import os
import io

# Paths
MODEL_PATH = "models/music_genre_model.pkl"
ENCODER_PATH = "models/music_genre_encoder.pkl"
DATA_PATH = "music_features.csv"
SONG_FOLDER = "songs/"

# Ensure model files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    st.error("❌ Model files not found! Please train the model first.")
    st.stop()

# Load trained model & label encoder
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# Load dataset
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    st.error("❌ Dataset not found!")
    st.stop()

# Extract correct feature names from dataset
expected_feature_names = df.columns[:-1].tolist()  # Skip the target column

# Function to extract MFCC features
def extract_features(audio_bytes):
    try:
        audio_data, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        return mfccs_mean.reshape(1, -1)
    except Exception as e:
        st.error(f"❌ Error extracting features: {e}")
        return None

# Streamlit UI
st.title("🎵 Music Genre Classification")

# **1️⃣ Display Dataset**
st.subheader("📊 Music Genre Dataset")
st.write("Here are some of the songs from the dataset:")
st.dataframe(df.head(10))

# **2️⃣ Filter Songs by Genre**
st.subheader("🎧 Filter Songs by Genre")
unique_genres = df["genre"].dropna().unique()

if len(unique_genres) > 0:
    selected_genre = st.selectbox("Select a Genre", unique_genres)
    filtered_df = df[df["genre"] == selected_genre]
    st.dataframe(filtered_df)
else:
    st.warning("⚠ No genres available in dataset.")

# **3️⃣ Select & Play Sample Songs**
st.subheader("🎶 Play Sample Songs & Predict Genre")

if os.path.exists(SONG_FOLDER):
    song_files = [f for f in os.listdir(SONG_FOLDER) if f.endswith((".wav", ".mp3"))]
else:
    song_files = []
    
# Option to upload a song
uploaded_file = st.file_uploader("Upload a song", type=["wav", "mp3"])

if uploaded_file is not None:
    song_name = uploaded_file.name
    st.audio(uploaded_file, format="audio/wav")
    audio_bytes = uploaded_file.read()
    features = extract_features(audio_bytes)
    
    if features is not None:
        features_df = pd.DataFrame(features, columns=expected_feature_names)
        try:
            prediction = model.predict(features_df)
            predicted_genre = encoder.inverse_transform(prediction)[0]
            st.success(f"🎵 Predicted Genre: **{predicted_genre}**")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# Select song from predefined folder if available
elif song_files:
    selected_song = st.selectbox("Select a song for prediction", ["Select a song"] + song_files)
    if selected_song != "Select a song":
        song_path = os.path.join(SONG_FOLDER, selected_song)
        st.audio(song_path, format="audio/wav")
        
        with open(song_path, "rb") as song_file:
            audio_bytes = song_file.read()
            features = extract_features(audio_bytes)
        
        if features is not None:
            features_df = pd.DataFrame(features, columns=expected_feature_names)
            try:
                prediction = model.predict(features_df)
                predicted_genre = encoder.inverse_transform(prediction)[0]
                st.success(f"🎵 Predicted Genre: **{predicted_genre}**")
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
else:
    st.warning("⚠ No songs found in the folder and no file uploaded.")
