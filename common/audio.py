import itertools

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.silence import split_on_silence

from audiostretchy.stretch import stretch_audio
from common.files_manager import FilesManager

def split_on_silence2(audio_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=100,
                     seek_step=1):
    def pairwise(iterable):
        's -> (s0,s1), (s1,s2), (s2, s3), ...'
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    if isinstance(keep_silence, bool):
        keep_silence = len(audio_segment) if keep_silence else 0

    output_ranges = [
        [start - keep_silence, end + keep_silence]
        for (start, end)
        in detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step)
    ]

    for range_i, range_ii in pairwise(output_ranges):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end+next_start)//2
            range_ii[0] = range_i[1]

    return [
        {
            'audio': audio_segment[max(start, 0): min(end, len(audio_segment))],
            'start': max(start, 0),
            'end': min(end, len(audio_segment))
        }
        for start, end in output_ranges
    ]

def speedup_audio(cloner, dst_text, dst_audio_filename):
    dst = AudioSegment.from_file(dst_audio_filename)
    dst_duration = dst.duration_seconds

    # Пытаемся озвучить без ускорения
    cloned_wav = cloner.process(
        speaker_wav_filename=dst_audio_filename,
        text=dst_text
    )
    # Теперь получаем результат через pydub
    non_speed_voice = AudioSegment.from_file(cloned_wav)
    non_speed_voice_duration = non_speed_voice.duration_seconds

    # Считаем отношение между длительностями
    speed = non_speed_voice_duration / dst_duration

    # Клонируем снова, но с нужной скоростью!
    cloned_wav = cloner.process(
        speaker_wav_filename=dst_audio_filename,
        text=dst_text,
        speed=speed
    )
    speed_voice = AudioSegment.from_file(cloned_wav)
    speed_voice_duration = speed_voice.duration_seconds

    # Считаем ratio для финального ускорения (возможно шаг не нужен вовсе)
    ratio = dst_duration / speed_voice_duration

    # Временный файл
    temp_manager = FilesManager()
    temp_file = temp_manager.create_temp_file(suffix='.wav').name

    # Ускоряем/замедляем клонированную аудио (возможно шаг не нужен вовсе)
    stretch_audio(cloned_wav, temp_file, ratio=ratio)

    stretched_audio = AudioSegment.from_file(temp_file)
    cropped_audio = stretched_audio[:dst_duration * 1000]

    cropped_audio.export(temp_file, format='wav')

    return temp_file


def combine_audio(audio_file_1, audio_file_2):
    a1 = AudioSegment.from_wav(audio_file_1)
    a2 = AudioSegment.from_wav(audio_file_2)

    tmpsound = a1.overlay(a2)
    temp_manager = FilesManager()
    temp_file = temp_manager.create_temp_file(suffix='.wav').name

    tmpsound.export(temp_file, format='wav')
    return temp_file