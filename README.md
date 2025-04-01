# Music Genre Classification

## Overview
This project is a machine learning-based music genre classification system. It extracts audio features from music files, processes them, and predicts the genre using a trained machine learning model. The system is implemented as a Streamlit web app for easy interaction.

## Features
- Extracts **MFCC (Mel-frequency cepstral coefficients)** features using **Librosa**.
- Trains a **Random Forest Classifier** using **Scikit-learn**.
- Implements a **Streamlit-based web app** for real-time genre classification.
- Uses **Label Encoding** to manage categorical genre labels.
- Supports **audio file uploads** for classification.
- Allows **users to play uploaded audio** before classification.
- Integrates **Joblib** for model persistence and deployment.
- Provides an **interactive dataset exploration** feature for filtering and displaying songs by genre.
- Designed with a **modular and reusable code structure** for scalability.

## Tech Stack
- **Python**
- **Librosa** (for feature extraction)
- **Scikit-learn** (for model training and evaluation)
- **Streamlit** (for web interface)
- **Joblib** (for saving and loading the model)
- **Pandas, NumPy** (for data handling and processing)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/music-genre-classification.git
   cd music-genre-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Upload an audio file.
2. Play the uploaded audio.
3. Click the **Predict Genre** button.
4. View the predicted genre on the interface.
5. Explore dataset songs by filtering genres.

## Dataset
- The dataset consists of audio files with corresponding genre labels.
- Features are extracted using **Librosa**.
- Categorical labels are encoded using **Label Encoding**.

## Model Training
- Extracted **MFCC features** from audio files.
- Trained a **Random Forest Classifier** using Scikit-learn.
- Evaluated model performance using accuracy metrics.
- Saved the trained model using **Joblib** for deployment.

## File Structure
```
├── app.py                 # Streamlit app
├── models
     |-music_genre_encorder.pkl
     |- music_genre_model.pkl       # Trained model (saved using Joblib)
├── api.py                 # API functions
├── songs/                  # Dataset folder
├── requirements.txt       # Required dependencies
├── README.md              # Project documentation
```


## Future Enhancements
- Improve model accuracy with deep learning (e.g., CNNs, LSTMs).
- Add support for real-time audio streaming.
- Implement genre visualization with spectrograms.


