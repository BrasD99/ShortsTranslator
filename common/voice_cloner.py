from TTS.api import TTS
from common.files_manager import FilesManager
from common.mapper import map
import torch

class VoiceCloner:
    def __init__(self, lang):
        self.lang_code = map(lang)
        device = torch.device('cpu')
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def process(self, speaker_wav_filename, text, speed=1.0, out_filename=None):
        temp_manager = FilesManager()
        if not out_filename:
            out_filename = temp_manager.create_temp_file(suffix='.wav').name
        self.tts.tts_to_file(
            text=text, 
            speaker_wav=speaker_wav_filename, 
            language=self.lang_code, 
            file_path=out_filename,
            speed=speed
        )
        return out_filename