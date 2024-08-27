import yt_dlp

vid_url = 'https://www.youtube.com/watch?v=eeAq4gkOEUY'

ydl_opts = {
    'format': 'best',
    'outtmpl': 'downloaded_video.mp4'
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([vid_url])


