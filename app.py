import streamlit as st
import pickle 
import pandas as pd
import numpy as np
import os 
from dotenv import load_dotenv
import requests

st.title('Movie Recommender System')

movies_data = pickle.load(open('movies_data.pkl','rb'))
movies = pd.DataFrame(movies_data)
model = pickle.load(open('model.pkl','rb'))
tfidf_matrix = pickle.load(open('tfidf.pkl','rb'))

load_dotenv()

api_key = os.getenv('TMDB_API_KEY')



selected_movie = st.selectbox(
    "Choose a Movies",
    movies['title'].values
)
def fetch_poster(id):
    url = f"https://api.tmdb.org/3/movie/{id}?api_key={api_key}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return 'https://image.tmdb.org/t/p/original/' + poster_path
    except Exception as e:
        print(f"Error fetching poster: {e}")
    return 'https://placehold.co/500x750?text=No+Poster+Available'



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



