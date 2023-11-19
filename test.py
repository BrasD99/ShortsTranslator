from pytube import YouTube
from engine import Engine

youtube_link = 'https://www.youtube.com/shorts/zpWWs_v2jf4'
language = 'English'

def update_progress(percent, text):
    print(f'{percent}: {text}')

#yt = YouTube(youtube_link)
#yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').first().download(filename='temp.mp4')
engine = Engine(language, update_progress)
engine.process('temp.mp4', 'out.mp4')