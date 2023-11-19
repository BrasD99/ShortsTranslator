import streamlit as st
from pytube import YouTube
from engine import Engine

def update_progress(percent, text):
    progress_bar.progress(percent)
    status_text.text(text)

def process(video_uri, language):
    update_progress(0, 'Downloading video from YouTube')
    yt = YouTube(video_uri)
    yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').first().download(filename='temp.mp4')
    engine = Engine(language, update_progress)
    engine.process('temp.mp4', 'out.mp4')
    update_progress(100, 'Done!')

languages = ['English', 'Kazakh']

st.title("HeyGenClone - Shorts demo!")
st.image("https://i.ibb.co/N2w50HD/corgi.jpg", use_column_width=True)
video_uri = st.text_input("Enter a link:")
language = st.selectbox("Select desired language:", languages)
if st.button("Process"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    process(video_uri, language)
    st.video('out.mp4')