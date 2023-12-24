import subprocess
import json
import cv2
from common.files_manager import FilesManager
from PIL import ImageFont, ImageDraw, Image
import numpy as np

FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)
LINE_THICKNESS = 2
BOTTOM_PERCENT = 20

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

def to_avi(frames, fps):
    temp_manager = FilesManager()
    temp_result_avi = temp_manager.create_temp_file(suffix='.avi').name

    frame_h, frame_w = frames[0].shape[:-1]

    out = cv2.VideoWriter(
        temp_result_avi, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h)
    )

    for frame in frames:
        out.write(frame)
    out.release()

    return temp_result_avi

def get_text(timestamp, subtitles):
    for subtitle in subtitles:
        if subtitle['start'] <= timestamp and subtitle['end'] >= timestamp:
            return subtitle['text']
    return None

def get_longest_text(subtitles):
    output = ''
    for subtitle in subtitles:
        if len(output) <= len(subtitle['text']):
            output = subtitle['text']
    return output

def get_batches(array, batch_size):
    batches = []
    for i in range(0, len(array), batch_size):
        batch = array[i:i+batch_size]
        batches.append(batch)
    return batches

def get_subtitles_v2(updates, batch_size):
    outputs = []
    for update in updates:
        batches = get_batches(update['word_segments'], batch_size)
        for batch in batches:
            outputs.append({
                'start': batch[0]['start'],
                'end': batch[-1]['end'],
                'text': ' '.join([d['word'] for d in batch])
            })
    return outputs

def get_subtitles(updates, batch_size):
    outputs = []
    for update in updates:
        start = update['start']
        end = update['end']
        duration = end - start
        words = update['text'].split()
        batches = get_batches(words, batch_size)
        interval = duration / len(batches)
        current_from = start
        for batch in batches:
            outputs.append({
                'start': current_from,
                'end': current_from + interval,
                'text': " ".join(batch)
            })
            current_from = current_from + interval

    return outputs

def add_subtitles(frames, subtitles, fps):
    outputs = []

    longest_text = get_longest_text(subtitles)
    orig_height, orig_width, _ = frames[0].shape
    initial_font_size = 50
    font_size_decrement = 1

    pil_im = Image.fromarray(frames[0])
    draw = ImageDraw.Draw(pil_im) 
    font = ImageFont.truetype("Montserrat-Medium.ttf", initial_font_size)
    text_width = draw.textbbox((0, 0), longest_text, font=font)[2]

    while text_width > orig_width:
        initial_font_size -= font_size_decrement
        font = ImageFont.truetype("Montserrat-Medium.ttf", initial_font_size)
        text_width = draw.textbbox((0, 0), longest_text, font=font)[2]

    for frame_num, frame in enumerate(frames):
        timestamp = frame_num / fps * 1000
        text = get_text(timestamp, subtitles)
        if text:
            pil_im = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_im) 
            _, _, text_width, _ = draw.textbbox((0, 0), text, font=font)
            draw.text(((orig_width - text_width)/2, orig_height * (1 - BOTTOM_PERCENT / 100)), text, font=font, fill='white')
            frame = np.array(pil_im)
        outputs.append(frame)
    return outputs

def get_text_v2(timestamp, subtitles):
    for subtitle in subtitles:
        if subtitle['start'] <= timestamp and subtitle['end'] >= timestamp:
            return subtitle['text']

    return None

def add_subtitles_v2(frames, subtitles, fps):
    outputs = []

    longest_text = get_longest_text(subtitles)
    orig_height, orig_width, _ = frames[0].shape
    initial_font_size = 50
    font_size_decrement = 5

    pil_im = Image.fromarray(frames[0])
    draw = ImageDraw.Draw(pil_im) 
    font = ImageFont.truetype("Montserrat-Medium.ttf", initial_font_size)
    text_width = draw.textbbox((0, 0), longest_text, font=font)[2]

    while text_width > orig_width:
        initial_font_size -= font_size_decrement
        font = ImageFont.truetype("Montserrat-Medium.ttf", initial_font_size)
        text_width = draw.textbbox((0, 0), longest_text, font=font)[2]

    for frame_num, frame in enumerate(frames):
        timestamp = frame_num / fps * 1000
        text = get_text_v2(timestamp, subtitles)
        if text:
            orig_height, orig_width, _ = frame.shape
            pil_im = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_im) 
            _, _, text_width, _ = draw.textbbox((0, 0), text, font=font)
            draw.text(((orig_width - text_width) / 2, orig_height * (1 - BOTTOM_PERCENT / 100)), text, font=font, fill='white', stroke_width=3, stroke_fill='black')
            frame = np.array(pil_im)
        outputs.append(frame)
    return outputs

def ms_to_log(ms):
    seconds, milliseconds = divmod(int(ms), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}.{:03d}".format(hours, minutes, seconds, milliseconds)