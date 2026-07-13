import streamlit as st
import pickle 
import pandas as pd
import numpy as np
import os 
from dotenv import load_dotenv
import requests

st.title('Movie Recommender System')
def download_file(url, filename):
    if not os.path.exists(filename):
        response = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            f.write(response.content)

@st.cache_resource
def load_assets():
    # Define your external URLs here
    files = {
        "model.pkl": "https://drive.google.com/uc?export=download&id=14eK83XcDmEZnXL0p9mzHSvzIGbGsTBvz",
        "movies_data.pkl": "https://drive.google.com/uc?export=download&id=13VgIS3QTWW-xhdF_BfqbDUKitJLWcrwD",
        "tfidf_matrix.pkl": "https://drive.google.com/uc?export=download&id=1OUl5MKuTm0l4FDBsrAPsJP-wu9KpD3s8"
    }
    
    for filename, url in files.items():
        download_file(url, filename)
    
    # Load the files
    model = pickle.load(open("model.pkl", "rb"))
    movies_data = pickle.load(open("movies_data.pkl", "rb"))
    tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
    
    return model, movies_data, tfidf_matrix

# Load assets
model, movies_data, tfidf_matrix = load_assets()


movies = pd.DataFrame(movies_data)



load_dotenv()

api_key = os.getenv('TMDB_API_KEY')



selected_movie = st.selectbox(
    "Choose a Movies",
    movies['title'].values
)
def fetch_poster(id):
    url = f"https://api.themoviedb.org/3/movie/{id}?api_key={api_key}"
    response = requests.get(url)
    data = response.json()
    return 'https://image.tmdb.org/t/p/original/' + data['poster_path']



def recommend(title ):
    idx = movies[movies.title == title].index[0]
    distances ,indices = model.kneighbors(tfidf_matrix[idx].reshape(1,-1))
    indices = indices.reshape(6,)
    distances = distances.reshape(6,)
    movies_list = np.column_stack((indices,distances))
    names =[]
    posters =[]
    for i in movies_list[1:6]:
        name = movies.iloc[int(i[0])].title
        names.append(name) 
        movie_id =movies[movies.title==name].id.iloc[0]
        poster = fetch_poster(movie_id)
        posters.append(poster)
    return names,posters 

if st.button("Recommend",type='primary'):
    with st.spinner('Searching'):
        names,posters = recommend(selected_movie)
    
    col1, col2, col3,col4,col5 = st.columns(5)

    with col1:
        st.write(names[0])
        st.image(posters[0])
    with col2:
        st.write(names[1])
        st.image(posters[1])
    with col3:
        st.write(names[2])
        st.image(posters[2])
    with col4:
        st.write(names[3])
        st.image(posters[3])
    with col5:
        st.write(names[4])
        st.image(posters[4])
    st.success("Here are your Recommendations")

def add_footer():
    footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: grey;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        <p>Built  by Hardik Sachdeva | 
        <a href="https://github.com/hxrdikk21" target="_blank">GitHub</a> | 
        <a href="https://linkedin.com/in/hxrdik-sachdeva" target="_blank">LinkedIn</a>
        </p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

st.sidebar.title("About this Project")
st.sidebar.info("""
This Movie Recommender System uses a content-based filtering approach:
- **Model:** KNearestNeighbors with Cosine Similarity.
- **Data:** 45,000+ movies from the TMDB dataset.
""")

with st.sidebar.expander("How to use this app"):
    st.write("1. Select a movie from the dropdown menu.")
    st.write("2. Click the 'Recommend' button.")
    st.write("3. View 5 similar movies based on your selection!")


add_footer()



