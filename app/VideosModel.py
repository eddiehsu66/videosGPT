# 将视频转录为文本
from moviepy.video.io.VideoFileClip import VideoFileClip
from app import openaifunc


class VideosModel:
    def __init__(self):
        self.videos = 'videos1'

    def extract_audio(self):
        video = VideoFileClip('../datastorage/' + self.videos + '.mp4')  # 视频所在路径
        audio = video.audio
        audio.write_audiofile('../datastorage/' + self.videos + '.mp3')  # 音频所在路径

    def audio_to_txt(self):
        transcript = openaifunc.openapi().whisper_api()
        s = bytes(transcript, 'utf-8').decode('unicode_escape')
        print(transcript)
