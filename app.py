import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import (
    cosine_similarity)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Music Recommendation System")
st.markdown("Discover new songs based on audio "
            "features — like Spotify's algorithm.")
st.markdown("---")

# Generate music dataset
@st.cache_data
def generate_music_data():
    np.random.seed(42)

    songs = {
        'Bollywood': {
            'songs': [
                'Tum Hi Ho', 'Kesariya',
                'Raataan Lambiyan',
                'Apna Bana Le', 'Besharam Rang',
                'Jhoome Jo Pathaan',
                'Chaleya', 'Jai Ho',
                'Dil Se', 'Kal Ho Na Ho',
                'Tujh Mein Rab Dikhta Hai',
                'Gerua', 'Ae Dil Hai Mushkil',
                'Hawayein', 'Dilbaro',
                'Channa Mereya',
                'Bulleya', 'Agar Tum Saath Ho',
                'London Thumakda', 'Gallan Goodiyaan'
            ],
            'artists': [
                'Arijit Singh', 'Arijit Singh',
                'Jubin Nautiyal',
                'Arijit Singh', 'Shilpa Rao',
                'Sukhwinder Singh',
                'Arijit Singh', 'A.R. Rahman',
                'A.R. Rahman', 'Sonu Nigam',
                'Roop Kumar Rathod',
                'Arijit Singh', 'Arijit Singh',
                'Arijit Singh', 'Shreya Ghoshal',
                'Arijit Singh', 'Vishal Dadlani',
                'Alka Yagnik', 'Neha Kakkar',
                'Harshdeep Kaur'
            ]
        },
        'Indie': {
            'songs': [
                'Pasoori', 'Phir Bhi Tumko Chahunga',
                'Moh Moh ke Dhaage',
                'Ik Vaari Aa', 'Daayre',
                'Ilahi', 'Tu Kisi Rail Si',
                'Qaafirana', 'Enna Sona',
                'Naina Da Kya Kasoor',
                'Ve Maahi', 'Pachtaoge',
                'Bekhayali', 'Tera Ban Jaunga',
                'Judaai', 'Humnava Mere',
                'Main Rang Sharbaton Ka',
                'Tere Bin', 'Lamberghini',
                'Saware'
            ],
            'artists': [
                'Ali Sethi', 'Shashaa Tirupati',
                'Papon', 'Arijit Singh',
                'Arijit Singh', 'Mohit Chauhan',
                'Lucky Ali', 'Arijit Singh',
                'A.R. Rahman', 'Amit Trivedi',
                'Arijit Singh', 'Arijit Singh',
                'Sachet Tandon', 'Akhil Sachdeva',
                'Arijit Singh', 'Jubin Nautiyal',
                'Atif Aslam', 'Vishal Mishra',
                'The Doorbeen', 'Arijit Singh'
            ]
        },
        'Pop': {
            'songs': [
                'Shape of You', 'Blinding Lights',
                'Levitating', 'Peaches',
                'Stay', 'Good 4 U',
                'Montero', 'drivers license',
                'Happier', 'Watermelon Sugar',
                'Dynamite', 'Butter',
                'Permission to Dance', 'DNA',
                'Fake Love', 'Boy With Luv',
                'ON', 'MIC Drop',
                'Fire', 'Spring Day'
            ],
            'artists': [
                'Ed Sheeran', 'The Weeknd',
                'Dua Lipa', 'Justin Bieber',
                'The Kid LAROI', 'Olivia Rodrigo',
                'Lil Nas X', 'Olivia Rodrigo',
                'Marshmello', 'Harry Styles',
                'BTS', 'BTS', 'BTS', 'BTS',
                'BTS', 'BTS', 'BTS', 'BTS',
                'BTS', 'BTS'
            ]
        },
        'Hip-Hop': {
            'songs': [
                'God\'s Plan', 'HUMBLE.',
                'Rockstar', 'Lucid Dreams',
                'Sunflower', 'Old Town Road',
                'Sicko Mode', 'Congratulations',
                'Mask Off', 'Bad and Boujee',
                'Magnolia', 'XO Tour Life',
                'Relationship', 'Nice For What',
                'In My Feelings', 'Nonstop',
                'Mob Ties', 'Emotionless',
                'Sandra\'s Rose', 'Summer Games'
            ],
            'artists': [
                'Drake', 'Kendrick Lamar',
                'Post Malone', 'Juice WRLD',
                'Post Malone', 'Lil Nas X',
                'Travis Scott', 'Post Malone',
                'Future', 'Migos',
                'Playboi Carti', 'Lil Uzi Vert',
                'Post Malone', 'Drake',
                'Drake', 'Drake', 'Drake',
                'Drake', 'Drake', 'Drake'
            ]
        },
        'Electronic': {
            'songs': [
                'Strobe', 'One More Time',
                'Sandstorm', 'Levels',
                'Wake Me Up',
                'Animals', 'Titanium',
                'Lean On', 'Roses',
                'Don\'t You Worry Child',
                'Clarity', 'Beautiful Now',
                'Summer', 'Ocean',
                'Waiting For Love',
                'The nights', 'Wake Me Up',
                'Hey Brother', 'Addicted to You',
                'Silhouettes'
            ],
            'artists': [
                'deadmau5', 'Daft Punk',
                'Darude', 'Avicii', 'Avicii',
                'Martin Garrix', 'David Guetta',
                'Major Lazer', 'The Chainsmokers',
                'Swedish House Mafia',
                'Zedd', 'Zedd',
                'Calvin Harris', 'Martin Garrix',
                'Avicii', 'Avicii', 'Avicii',
                'Avicii', 'Avicii', 'Avicii'
            ]
        }
    }

    rows = []
    for genre, data in songs.items():
        for song, artist in zip(
            data['songs'], data['artists']
        ):
            # Audio features
            if genre == 'Electronic':
                energy  = np.random.uniform(0.7, 1.0)
                valence = np.random.uniform(0.5, 0.9)
                tempo   = np.random.uniform(120, 175)
                dance   = np.random.uniform(0.6, 0.95)
                acoust  = np.random.uniform(0.0, 0.15)
            elif genre == 'Hip-Hop':
                energy  = np.random.uniform(0.6, 0.9)
                valence = np.random.uniform(0.3, 0.8)
                tempo   = np.random.uniform(70, 115)
                dance   = np.random.uniform(0.7, 0.95)
                acoust  = np.random.uniform(0.0, 0.2)
            elif genre == 'Bollywood':
                energy  = np.random.uniform(0.4, 0.85)
                valence = np.random.uniform(0.4, 0.95)
                tempo   = np.random.uniform(75, 140)
                dance   = np.random.uniform(0.5, 0.9)
                acoust  = np.random.uniform(0.1, 0.5)
            elif genre == 'Indie':
                energy  = np.random.uniform(0.3, 0.7)
                valence = np.random.uniform(0.2, 0.7)
                tempo   = np.random.uniform(65, 120)
                dance   = np.random.uniform(0.3, 0.75)
                acoust  = np.random.uniform(0.3, 0.8)
            else:  # Pop
                energy  = np.random.uniform(0.5, 0.9)
                valence = np.random.uniform(0.5, 0.95)
                tempo   = np.random.uniform(95, 140)
                dance   = np.random.uniform(0.6, 0.95)
                acoust  = np.random.uniform(0.0, 0.3)

            rows.append({
                'song':          song,
                'artist':        artist,
                'genre':         genre,
                'energy':        round(energy, 3),
                'valence':       round(valence, 3),
                'danceability':  round(dance, 3),
                'acousticness':  round(acoust, 3),
                'tempo':         round(tempo, 1),
                'loudness':      round(
                    np.random.uniform(-12, -2), 1),
                'speechiness':   round(
                    np.random.uniform(0.02, 0.3), 3),
                'instrumentalness': round(
                    np.random.uniform(0.0, 0.3), 3),
                'popularity':    np.random.randint(
                    50, 100),
                'duration_min':  round(
                    np.random.uniform(2.5, 5.5), 2)
            })

    return pd.DataFrame(rows)

df = generate_music_data()

# Recommendation engine
@st.cache_data
def build_recommender(df):
    features = ['energy', 'valence',
                'danceability', 'acousticness',
                'tempo', 'speechiness',
                'instrumentalness']
    scaler   = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(
        df[features])
    sim_matrix = cosine_similarity(
        df_scaled[features])
    return sim_matrix, features, scaler

sim_matrix, features, scaler = \
    build_recommender(df)

def get_recommendations(song_name, n=5,
                        genre_filter=None):
    matches = df[df['song'] == song_name]
    if len(matches) == 0:
        return pd.DataFrame()

    idx      = matches.index[0]
    sim_scores = list(enumerate(
        sim_matrix[idx]))
    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )
    sim_scores = [
        s for s in sim_scores
        if s[0] != idx
    ]

    if genre_filter and \
            genre_filter != 'All Genres':
        genre_indices = df[
            df['genre'] == genre_filter
        ].index.tolist()
        sim_scores = [
            s for s in sim_scores
            if s[0] in genre_indices
        ]

    top_indices = [s[0] for s in
                   sim_scores[:n]]
    top_scores  = [s[1] for s in
                   sim_scores[:n]]

    recs = df.iloc[top_indices].copy()
    recs['similarity'] = [
        round(s * 100, 1)
        for s in top_scores
    ]
    return recs

# Mood to features mapping
MOOD_FEATURES = {
    "😄 Happy & Energetic": {
        'energy': 0.85, 'valence': 0.9,
        'danceability': 0.8, 'acousticness': 0.1
    },
    "😢 Sad & Emotional": {
        'energy': 0.3, 'valence': 0.2,
        'danceability': 0.3, 'acousticness': 0.7
    },
    "🏋️ Workout Mode": {
        'energy': 0.95, 'valence': 0.7,
        'danceability': 0.85, 'acousticness': 0.05
    },
    "😌 Relaxed & Chill": {
        'energy': 0.3, 'valence': 0.6,
        'danceability': 0.4, 'acousticness': 0.8
    },
    "🎉 Party Time": {
        'energy': 0.9, 'valence': 0.95,
        'danceability': 0.95, 'acousticness': 0.05
    },
    "🧘 Meditation": {
        'energy': 0.1, 'valence': 0.5,
        'danceability': 0.2, 'acousticness': 0.9
    },
    "💼 Focus & Study": {
        'energy': 0.4, 'valence': 0.5,
        'danceability': 0.3, 'acousticness': 0.6
    },
    "💔 Heartbreak": {
        'energy': 0.25, 'valence': 0.15,
        'danceability': 0.25, 'acousticness': 0.75
    }
}

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎵 Recommend",
    "😊 Mood Playlist",
    "📊 Music Analytics",
    "🗺️ Song Universe",
    "🔍 Explore"
])

# Tab 1 — Recommend by Song
with tab1:
    st.markdown("### 🎵 Find Similar Songs")

    col1, col2 = st.columns([2, 1])

    with col1:
        genre_options = ['All Genres'] + \
            sorted(df['genre'].unique())
        filter_genre  = st.selectbox(
            "Filter by genre:",
            genre_options
        )

        if filter_genre == 'All Genres':
            song_options = sorted(
                df['song'].unique())
        else:
            song_options = sorted(
                df[df['genre'] ==
                   filter_genre
                   ]['song'].unique())

        selected_song = st.selectbox(
            "Select a song you like:",
            song_options
        )

        n_recs = st.slider(
            "Number of recommendations:",
            3, 15, 8
        )
        rec_genre = st.selectbox(
            "Recommend from genre:",
            genre_options,
            key="rec_genre"
        )

    with col2:
        # Show selected song info
        song_info = df[
            df['song'] == selected_song
        ].iloc[0]

        st.markdown("### 🎧 Selected Song")
        st.markdown(
            f"**{song_info['song']}**")
        st.markdown(
            f"🎤 {song_info['artist']}")
        st.markdown(
            f"🎸 {song_info['genre']}")
        st.markdown(
            f"⭐ Popularity: "
            f"{song_info['popularity']}/100")
        st.markdown(
            f"⏱️ Duration: "
            f"{song_info['duration_min']} min")

        # Audio features radar
        feat_vals = [
            song_info['energy'],
            song_info['valence'],
            song_info['danceability'],
            song_info['acousticness'],
            song_info['speechiness']
        ]
        feat_names = [
            'Energy', 'Valence',
            'Dance', 'Acoustic', 'Speech'
        ]
        fig_radar = go.Figure(go.Scatterpolar(
            r=feat_vals + [feat_vals[0]],
            theta=feat_names + [feat_names[0]],
            fill='toself',
            fillcolor='rgba(52,152,219,0.3)',
            line=dict(color='#3498db', width=2)
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1])),
            height=300,
            title="Audio Features",
            showlegend=False
        )
        st.plotly_chart(
            fig_radar,
            use_container_width=True)

    # Get recommendations
    recs = get_recommendations(
        selected_song, n_recs, rec_genre
        if rec_genre != 'All Genres'
        else None)

    if len(recs) > 0:
        st.markdown("---")
        st.markdown(
            f"### 🎵 Songs Similar to "
            f"**{selected_song}**")

        for _, rec in recs.iterrows():
            col_a, col_b, col_c, col_d, \
                col_e = st.columns(
                [3, 2, 1, 1, 1])
            with col_a:
                st.markdown(
                    f"🎵 **{rec['song']}**")
            with col_b:
                st.markdown(
                    f"🎤 {rec['artist']}")
            with col_c:
                st.markdown(
                    f"🎸 {rec['genre']}")
            with col_d:
                st.markdown(
                    f"⭐ {rec['popularity']}")
            with col_e:
                st.markdown(
                    f"🔗 {rec['similarity']}%")

# Tab 2 — Mood Playlist
with tab2:
    st.markdown("### 😊 Mood-Based Playlist")

    col1, col2 = st.columns(2)

    with col1:
        mood = st.selectbox(
            "How are you feeling?",
            list(MOOD_FEATURES.keys())
        )
        mood_genre = st.selectbox(
            "Preferred genre:",
            ['All Genres'] +
            sorted(df['genre'].unique()),
            key="mood_genre"
        )
        n_mood = st.slider(
            "Playlist size:", 5, 20, 10,
            key="n_mood"
        )

        mood_feats = MOOD_FEATURES[mood]

        # Show mood description
        mood_desc = {
            "😄 Happy & Energetic":
                "High energy, positive vibes",
            "😢 Sad & Emotional":
                "Low energy, acoustic, emotional",
            "🏋️ Workout Mode":
                "Maximum energy, high tempo",
            "😌 Relaxed & Chill":
                "Low energy, acoustic, calm",
            "🎉 Party Time":
                "Maximum danceability, festive",
            "🧘 Meditation":
                "Minimal energy, very acoustic",
            "💼 Focus & Study":
                "Low distraction, instrumental",
            "💔 Heartbreak":
                "Very low energy, sad tone"
        }
        st.info(
            f"**{mood}**\n\n"
            f"{mood_desc.get(mood, '')}")

    with col2:
        # Mood feature display
        fig_mood = go.Figure(go.Bar(
            x=list(mood_feats.keys()),
            y=list(mood_feats.values()),
            marker_color=[
                '#e74c3c', '#f39c12',
                '#2ecc71', '#3498db'
            ]
        ))
        fig_mood.update_layout(
            title=f'Audio Profile: {mood}',
            yaxis_range=[0, 1],
            height=300,
            template='plotly_white'
        )
        st.plotly_chart(
            fig_mood,
            use_container_width=True)

    if st.button("🎵 Generate Playlist",
                 type="primary"):
        # Score songs by mood match
        df_score = df.copy()
        score    = np.zeros(len(df))

        for feat, target in mood_feats.items():
            if feat in df.columns:
                diff    = abs(
                    df[feat] - target)
                score  += (1 - diff)

        df_score['mood_score'] = score

        if mood_genre != 'All Genres':
            df_score = df_score[
                df_score['genre'] ==
                mood_genre]

        playlist = df_score.nlargest(
            n_mood, 'mood_score')

        st.success(
            f"🎵 {mood} Playlist — "
            f"{len(playlist)} songs")

        for i, (_, song) in enumerate(
            playlist.iterrows(), 1
        ):
            col_a, col_b, col_c, col_d = \
                st.columns([1, 3, 2, 1])
            with col_a:
                st.markdown(f"**{i}.**")
            with col_b:
                st.markdown(
                    f"🎵 **{song['song']}**")
            with col_c:
                st.markdown(
                    f"🎤 {song['artist']}")
            with col_d:
                st.markdown(
                    f"⭐ {song['popularity']}")

# Tab 3 — Analytics
with tab3:
    st.markdown("### 📊 Music Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Genre distribution
        genre_counts = df['genre'].value_counts()
        fig2 = px.pie(
            values=genre_counts.values,
            names=genre_counts.index,
            title='Songs by Genre',
            color_discrete_sequence=
                px.colors.qualitative.Set2
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2,
                        use_container_width=True)

    with col2:
        # Audio features by genre
        feat_genre = df.groupby('genre')[
            ['energy', 'valence',
             'danceability', 'acousticness']
        ].mean().reset_index()

        fig3 = go.Figure()
        features_plot = ['energy', 'valence',
                         'danceability',
                         'acousticness']
        colors_plot   = ['#e74c3c', '#f39c12',
                         '#2ecc71', '#3498db']
        for feat, color in zip(
            features_plot, colors_plot
        ):
            fig3.add_trace(go.Bar(
                name=feat.title(),
                x=feat_genre['genre'],
                y=feat_genre[feat],
                marker_color=color
            ))
        fig3.update_layout(
            title='Avg Audio Features by Genre',
            barmode='group',
            height=350,
            template='plotly_white',
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig3,
                        use_container_width=True)

    # Top artists
    st.markdown("#### 🎤 Top Artists")
    top_artists = df.groupby(
        'artist').agg(
        songs=('song', 'count'),
        avg_popularity=('popularity', 'mean')
    ).sort_values(
        'songs', ascending=False
    ).head(10).reset_index()

    fig4 = px.bar(
        top_artists,
        x='songs', y='artist',
        orientation='h',
        title='Artists by Number of Songs',
        color='avg_popularity',
        color_continuous_scale='Viridis'
    )
    fig4.update_layout(
        height=400,
        template='plotly_white'
    )
    st.plotly_chart(fig4,
                    use_container_width=True)

    # Feature correlations
    st.markdown("#### 🔗 Feature Correlations")
    corr_features = [
        'energy', 'valence',
        'danceability', 'acousticness',
        'tempo', 'popularity'
    ]
    corr = df[corr_features].corr()
    fig5 = px.imshow(
        corr,
        title='Audio Feature Correlations',
        color_continuous_scale='RdYlGn',
        zmin=-1, zmax=1
    )
    fig5.update_layout(
        height=400,
        template='plotly_white'
    )
    st.plotly_chart(fig5,
                    use_container_width=True)

# Tab 4 — Song Universe
with tab4:
    st.markdown("### 🗺️ Song Universe")
    st.markdown("See all songs plotted by "
                "their audio features.")

    # PCA visualization
    feat_cols = ['energy', 'valence',
                 'danceability', 'acousticness',
                 'tempo', 'speechiness']
    pca_data  = df[feat_cols].values
    scaler_v  = MinMaxScaler()
    pca_data  = scaler_v.fit_transform(pca_data)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pca_data)

    df_viz = df.copy()
    df_viz['PC1'] = pca_result[:, 0]
    df_viz['PC2'] = pca_result[:, 1]

    fig6 = px.scatter(
        df_viz,
        x='PC1', y='PC2',
        color='genre',
        size='popularity',
        hover_data=['song', 'artist',
                    'energy', 'valence'],
        title='Song Universe — '
              'PCA of Audio Features',
        labels={
            'PC1': f'Component 1 '
                   f'({pca.explained_variance_ratio_[0]:.1%})',
            'PC2': f'Component 2 '
                   f'({pca.explained_variance_ratio_[1]:.1%})'
        }
    )
    fig6.update_layout(
        height=550,
        template='plotly_white'
    )
    st.plotly_chart(fig6,
                    use_container_width=True)

    st.caption(
        f"PCA explains "
        f"{sum(pca.explained_variance_ratio_):.1%} "
        f"of total variance. Hover over "
        f"points to see song details.")

# Tab 5 — Explore
with tab5:
    st.markdown("### 🔍 Explore by Features")

    col1, col2 = st.columns(2)
    with col1:
        energy_range = st.slider(
            "Energy Range:",
            0.0, 1.0, (0.4, 1.0), 0.05
        )
        valence_range = st.slider(
            "Valence (Positivity):",
            0.0, 1.0, (0.4, 1.0), 0.05
        )

    with col2:
        dance_range = st.slider(
            "Danceability:",
            0.0, 1.0, (0.4, 1.0), 0.05
        )
        explore_genre = st.multiselect(
            "Genres:",
            df['genre'].unique(),
            default=df['genre'].unique()
        )

    explored = df[
        (df['energy'] >= energy_range[0]) &
        (df['energy'] <= energy_range[1]) &
        (df['valence'] >= valence_range[0]) &
        (df['valence'] <= valence_range[1]) &
        (df['danceability'] >= dance_range[0]) &
        (df['danceability'] <= dance_range[1]) &
        (df['genre'].isin(explore_genre))
    ].sort_values(
        'popularity', ascending=False)

    st.markdown(
        f"**{len(explored)} songs** "
        f"match your criteria")

    if len(explored) > 0:
        display = explored[[
            'song', 'artist', 'genre',
            'energy', 'valence',
            'danceability', 'popularity'
        ]].head(20).copy()
        display.columns = [
            'Song', 'Artist', 'Genre',
            'Energy', 'Valence',
            'Dance', 'Popularity'
        ]
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Music Recommendation System | "
    "Powered by Cosine Similarity + PCA"
)