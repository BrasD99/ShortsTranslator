import subprocess
import json
from common.files_manager import FilesManager

def to_segments(updates, audio_duration):
    segments = []
    prev_end = 0

    for i, update in enumerate(updates):
        start = update['start']
        end = update['end']
        voice = update['voice']

        if start > prev_end:
            segments.append({'start': prev_end, 'end': start, 'empty': True})

        segments.append({'start': start, 'end': end,
                        'empty': False, 'voice': voice})

        if i + 1 == len(updates) and end < audio_duration:
            segments.append(
                {'start': end, 'end': audio_duration, 'empty': True})

        prev_end = end

    return segments

def format_duration(duration):
    hours, remainder = divmod(int(duration), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((duration - int(duration)) * 1000)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}'

def get_duration(filename):
    command = f'ffprobe -i {filename} -show_entries format=duration -v quiet -print_format json'
    output = subprocess.check_output(command, shell=True)
    data = json.loads(output)
    return float(data['format']['duration'])

def merge(audio_filename, avi_filename, out_filename):
    audio_duration = get_duration(audio_filename)
    video_duration = get_duration(avi_filename)
    duration = format_duration(video_duration)

    if audio_duration > video_duration:
        temp_manager = FilesManager()
        temp_wav = temp_manager.create_temp_file(suffix='.wav').name
        command = 'ffmpeg -i {} -ss 00:00:00 -to {} -c copy {}'.format(
            audio_filename, duration, temp_wav
        )
        subprocess.call(command, shell=True)
        audio_filename = temp_wav
    else:
        duration = format_duration(audio_duration)

    command = 'ffmpeg -y -i {} -i {} -ss 00:00:00.000 -to {} -strict -2 -q:v 1 {} -loglevel {}'.format(
        audio_filename, avi_filename, duration, out_filename, 'verbose'
    )
    subprocess.call(command, shell=True)