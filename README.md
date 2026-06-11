# Music Recommendation System 🎵

Spotify-style music recommender using
cosine similarity on audio features.

## Live Demo
[Click here](YOUR_STREAMLIT_URL)

## Features
- Song-based recommendations using cosine similarity
- Mood-based playlist generation (8 moods)
- Audio feature radar chart per song
- PCA visualization of song universe
- Feature-based song exploration
- Genre and artist analytics

## How It Works
Each song has audio features (energy, valence,
danceability, acousticness, tempo). Cosine
similarity finds songs with most similar
feature vectors.

## Songs Covered
100 songs across Bollywood, Indie, Pop,
Hip-Hop and Electronic genres

## Tools Used
- Python, Streamlit, Scikit-learn, Plotly, Pandas

## How to Run Locally
pip install streamlit pandas numpy plotly scikit-learn
streamlit run app.py
