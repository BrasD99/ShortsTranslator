import random
import numpy as np
import soundfile as sf
import torch
from audiosr import build_model, super_resolution
from common.files_manager import FilesManager

class SpeechEnhancer:
    def __init__(self) -> None:
        self.sr = 48000
        self.files_manager = FilesManager()
        self.audiosr = build_model(model_name='speech', device=torch.device('cpu'))
    
    def process(self, input_audio):
        seed = random.randint(0, 2**32 - 1)
        waveform = super_resolution(
            self.audiosr,
            input_audio,
            seed=seed,
            guidance_scale=3.5,
            ddim_steps=50,
            latent_t_per_second=12.8
        )
        out_filename = self.files_manager.create_temp_file(suffix='.wav').name
        out_wav = (waveform[0] * 32767).astype(np.int16).T
        sf.write(out_filename, data=out_wav, samplerate=48000)

        return out_filename