import yt_dlp

vid_url = 'https://www.youtube.com/watch?v=tkMg8g8vVUo&t=24s'

ydl_opts = {
    'format': 'best',
    'outtmpl': 'downloaded_video.mp4'
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([vid_url])


