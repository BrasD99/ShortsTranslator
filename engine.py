from moviepy.video.io.VideoFileClip import VideoFileClip, AudioFileClip
from common.files_manager import FilesManager
from common.dereverb import MDXNetDereverb
from common.audio import split_on_silence2
from common.whisperx.asr import load_model, load_audio
from common.whisperx.alignment import load_align_model, align
from common.translator import TextHelper
from common.voice_cloner import VoiceCloner
from common.helpers import to_segments, merge
from common.audio import speedup_audio, combine_audio
from pydub import AudioSegment
import torch

class Engine:
    def __init__(self, language, update_progress):
        self.dst_lang = language
        self.update_progress = update_progress
        update_progress(5, 'Preparing voice cloner')
        self.cloner = VoiceCloner(language)
        self.files_manager = FilesManager()
        update_progress(10, 'Preparing dereverb')
        self.dereverb = MDXNetDereverb(15)
        self.text_helper = TextHelper()
        self.whisper_batch_size = 16
        update_progress(20, 'Preparing whisper')
        self.whisper = load_model('large-v2', device='cpu', compute_type='int8')
    
    def process(self, video_file, out_video_filename):
        self.update_progress(30, 'Getting audio from video')
        audioclip = AudioFileClip(video_file)
        #orig_clip = VideoFileClip(video_file, verbose=False)
        original_audio_file = self.files_manager.create_temp_file(suffix='.wav').name
        #orig_clip.audio.write_audiofile(original_audio_file, codec='pcm_s16le', verbose=False, logger=None)
        audioclip.write_audiofile(original_audio_file, codec='pcm_s16le', verbose=False, logger=None)
        orig_audio = AudioSegment.from_file(original_audio_file, format='wav')
        self.update_progress(40, 'Getting voice and noise from audio')
        dereverb_out = self.dereverb.split(original_audio_file)
        voice_audio = AudioSegment.from_file(dereverb_out['voice_file'], format='wav')
        noise_audio = AudioSegment.from_file(dereverb_out['noise_file'], format='wav')
        audio_segments = split_on_silence2(voice_audio, silence_thresh=-50)

        updates = []
        self.update_progress(40, 'Cloning voice...')
        for audio_segment in audio_segments:
            audio_segment_file = self.files_manager.create_temp_file(suffix='.wav').name
            sub_audio = voice_audio[audio_segment['start'] * 1000: audio_segment['end'] * 1000]
            sub_audio.export(audio_segment_file, format='wav')
            result, language = self.transcribe_audio_extended(audio_segment_file)
            for transcribtion in result['segments']:
                dst_text = self.text_helper.translate(transcribtion['text'], src_lang=language, dst_lang=self.dst_lang)
                speaker_wav_filename = self.files_manager.create_temp_file(suffix='.wav').name
                voice_audio[(audio_segment['start'] + transcribtion['start']) * 1000:
                            (audio_segment['start'] + transcribtion['end']) * 1000].export(speaker_wav_filename, format='wav')
                output_wav = speedup_audio(self.cloner, dst_text, speaker_wav_filename)
                updates.append({
                    'start': (audio_segment['start'] + transcribtion['start']) * 1000,
                    'end': (audio_segment['start'] + transcribtion['end']) * 1000,
                    'voice': output_wav
                })
        
        self.update_progress(70, 'Merging audio...')
        original_audio_duration = orig_audio.duration_seconds * 1000
        audio_segments = to_segments(updates, original_audio_duration)
        speech_final_audio = AudioSegment.silent(duration=0)
        for segment in audio_segments:
            if segment['empty']:
                duration = segment['end'] - segment['start']
                speech_final_audio += AudioSegment.silent(duration=duration)
            else:
                speech_final_audio += AudioSegment.from_file(segment['voice'])
        speech_audio_wav = self.files_manager.create_temp_file(suffix='.wav').name
        speech_final_audio.export(speech_audio_wav, format='wav')

        noise_audio_wav = self.files_manager.create_temp_file(suffix='.wav').name
        noise_audio.export(noise_audio_wav, format='wav')

        combined_audio = combine_audio(speech_audio_wav, noise_audio_wav)

        self.update_progress(90, 'Merging video and audio...')
        merge(combined_audio, video_file, out_video_filename)

    def transcribe_audio_extended(self, audio_file):
        audio = load_audio(audio_file)
        result = self.whisper.transcribe(audio, batch_size=self.whisper_batch_size)
        language = result['language']
        model_a, metadata = load_align_model(language_code=language, device=torch.device('cpu'))
        result = align(result['segments'], model_a, metadata, audio, torch.device('cpu'), return_char_alignments=False)
        return result, language