from moviepy.video.io.VideoFileClip import VideoFileClip
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from common.files_manager import FilesManager
from common.dereverb import MDXNetDereverb
from common.whisperx.asr import load_model, load_audio
from common.whisperx.alignment import load_align_model, align
from common.voice_cloner import VoiceCloner
from common.audio import speedup_audio, combine_audio
from pydub import AudioSegment
from common.helpers import to_segments, to_avi, merge
from common.detector import FaceDetector
import os
from tqdm import tqdm
import cv2
import shutil
import json
from common.translator import TextHelper
import datetime
import torch
import glob
from common.lipsyncv2.wrapper import LipSync

def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def recreate_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)

def initial_process(project_name, video_filename):
    project_folder = os.path.join('projects', project_name)
    frames_folder = os.path.join(project_folder, 'frames')
    recreate_folder(project_folder)
    os.mkdir(frames_folder)

    orig_clip = VideoFileClip(video_filename, verbose=False)
    scenes = detect_scenes(video_filename)

    scenes_info = []
    for scene in scenes:
        start, end = scene
        scenes_info.append((start.frame_num, end.frame_num))
    
    def get_scene_by_frame_num(frame_num):
        for i, scene in enumerate(scenes):
            start, end = scene
            if start.frame_num <= frame_num and end.frame_num -1 >= frame_num:
                return i

    for frame_num, frame in tqdm(enumerate(orig_clip.iter_frames()), desc='Getting frames'):
        scene_id = get_scene_by_frame_num(frame_num)
        scene_folder = os.path.join(frames_folder, f'scene_{scene_id}')
        if not os.path.exists(scene_folder):
            os.mkdir(scene_folder)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_filename = os.path.join(scene_folder, f'{frame_num}.jpg')
        cv2.imwrite(frame_filename, frame)
    
    original_audio_file = os.path.join(project_folder, 'orig.wav')
    orig_clip.audio.write_audiofile(original_audio_file, codec='pcm_s16le', verbose=False, logger=None)

    orig_video_file = os.path.join(project_folder, 'orig.mp4')
    shutil.copyfile(video_filename, orig_video_file)

    config = {
        'project_name': project_name,
        'fps': orig_clip.fps,
        'audio_filename': original_audio_file,
        'frames_folder': frames_folder,
        'video_filename': orig_video_file
    }

    config = json.dumps(config, ensure_ascii=False)

    config_file = os.path.join(project_folder, 'config.json')

    with open(config_file, 'w', encoding='utf-8') as file:
        file.write(config)

def dereverb(project_name):
    project_folder = os.path.join('projects', project_name)
    config_file = os.path.join(project_folder, 'config.json')
    config = read_json(config_file)

    dereverb = MDXNetDereverb(15)

    dereverb_out = dereverb.split(config['audio_filename'])
    del dereverb

    voice_filename = os.path.join(project_folder, 'voice.wav')
    noise_filename = os.path.join(project_folder, 'noise.wav')
    shutil.move(dereverb_out['voice_file'], voice_filename)
    shutil.move(dereverb_out['noise_file'], noise_filename)

    #whisper = load_model('large-v2', device='cpu', compute_type='int8')
    #result, language = transcribe_audio_extended(whisper, voice_filename)
    #del whisper
    results = transcribe_audio_extended_v2(voice_filename)

    transcribtions_file = os.path.join(project_folder, 'o_transcribtions.json')
    
    results = json.dumps(results, ensure_ascii=False)

    with open(transcribtions_file, 'w', encoding='utf-8') as file:
        file.write(results)
    
    config['voice_audio_file'] = voice_filename
    config['noise_audio_file'] = noise_filename
    config['transcribtions_file'] = transcribtions_file
    config = json.dumps(config, ensure_ascii=False)

    with open(config_file, 'w', encoding='utf-8') as file:
        file.write(config)

def translate(project_name, dst_lang):
    project_folder = os.path.join('projects', project_name)
    config_file = os.path.join(project_folder, 'config.json')
    config = read_json(config_file)
    transcribtions = read_json(config['transcribtions_file'])

    text_helper = TextHelper()

    translated = []
    for i, transcribtion in enumerate(transcribtions['segments']):
        dst_text = text_helper.translate(transcribtion['text'], src_lang=transcribtions['language'], dst_lang=dst_lang)
        translated.append({
            'id': i,
            'start': transcribtion['start'],
            'end': transcribtion['end'],
            'text': dst_text
        })
    
    translated = json.dumps(translated, ensure_ascii=False)
    translated_file = os.path.join(project_folder, 'translated_transcribtions.json')

    with open(translated_file, 'w', encoding='utf-8') as file:
        file.write(translated)

    config['translated_file'] = translated_file

    config = json.dumps(config, ensure_ascii=False)

    with open(config_file, 'w', encoding='utf-8') as file:
        file.write(config)

def clone(project_name, dst_lang):
    project_folder = os.path.join('projects', project_name)
    config_file = os.path.join(project_folder, 'config.json')
    config = read_json(config_file)
    translations = read_json(config['translated_file'])

    files_manager = FilesManager()
    cloner = VoiceCloner(dst_lang)
    #whisper = load_model('large-v2', device='cpu', compute_type='int8')

    voice_folder = os.path.join(project_folder, 'voices')
    recreate_folder(voice_folder)

    orig_audio = AudioSegment.from_file(config['voice_audio_file'], format='wav')
    
    cloned_transcribtions = []

    for segment in translations:
        speaker_wav_filename = files_manager.create_temp_file(suffix='.wav').name
        orig_audio[segment['start'] * 1000: segment['end'] * 1000].export(speaker_wav_filename, format='wav')
        output_wav = speedup_audio(cloner, segment['text'], speaker_wav_filename)
        results = transcribe_audio_extended_v2(output_wav)
        #result, _ = transcribe_audio_extended(whisper, output_wav)
        voice_filename = os.path.join(voice_folder, f'{segment["id"]}.wav')
        shutil.move(output_wav, voice_filename)
        cloned_transcribtions.append({
            'transcribtion_id': segment['id'],
            'voice_file': voice_filename,
            'words': [{'start': (segment['start'] + w['start']) * 1000, \
                        'end': (segment['start'] + w['end']) * 1000, \
                        'word': w['text']} for w in results['segments'][0]['words']]
        })
        '''
        cloned_transcribtions.append({
            'transcribtion_id': translation['id'],
            'voice_file': voice_filename,
            'words': [{'start': (translation['start'] + d['start']) * 1000, \
                                  'end': (translation['start'] + d['end']) * 1000, \
                                  'word': d['word']} for d in result['word_segments']]
        })
        '''

    cloned_transcribtions = json.dumps(cloned_transcribtions, ensure_ascii=False)
    cloned_transcribtions_file = os.path.join(project_folder, 'cloned.json')

    with open(cloned_transcribtions_file, 'w', encoding='utf-8') as file:
        file.write(cloned_transcribtions)

    config['cloned_file'] = cloned_transcribtions_file

    config = json.dumps(config, ensure_ascii=False)

    with open(config_file, 'w', encoding='utf-8') as file:
        file.write(config)

def re_clone(project_name, transcribtion_id, dst_lang):
    project_folder = os.path.join('projects', project_name)
    config_file = os.path.join(project_folder, 'config.json')
    config = read_json(config_file)
    translations = read_json(config['translated_file'])

    files_manager = FilesManager()
    cloner = VoiceCloner(dst_lang)
    #whisper = load_model('large-v2', device='cpu', compute_type='int8')

    recloned_folder = os.path.join(project_folder, 'recloned')

    if not os.path.exists(recloned_folder):
        os.mkdir(recloned_folder)

    orig_audio = AudioSegment.from_file(config['voice_audio_file'], format='wav')

    for segment in translations['segments']:
        if segment['id'] == transcribtion_id:
            speaker_wav_filename = files_manager.create_temp_file(suffix='.wav').name
            orig_audio[segment['start'] * 1000: segment['end'] * 1000].export(speaker_wav_filename, format='wav')
            output_wav = speedup_audio(cloner, segment['text'], speaker_wav_filename, 1)
            results = transcribe_audio_extended_v2(output_wav)
            #result, _ = transcribe_audio_extended(whisper, output_wav)
            current_datetime = datetime.datetime.now()
            current_date = current_datetime.strftime("%Y-%m-%d")
            current_time = current_datetime.strftime("%H-%M-%S")
            voice_filename = os.path.join(recloned_folder, f'{segment["id"]}_{current_date}_{current_time}.wav')
            shutil.move(output_wav, voice_filename)

            cloned_transcribtion = {
                'transcribtion_id': segment['id'],
                'voice_file': voice_filename,
                'words': [{'start': (segment['start'] + w['start']) * 1000, \
                        'end': (segment['start'] + w['end']) * 1000, \
                        'word': w['text']} for w in results['segments'][0]['words']]
            }

            cloned_transcribtion = json.dumps(cloned_transcribtion, ensure_ascii=False)
            cloned_transcribtion_filename = os.path.join(recloned_folder, f'{segment["id"]}_{current_date}_{current_time}.json')
            with open(cloned_transcribtion_filename, 'w', encoding='utf-8') as file:
                file.write(cloned_transcribtion)
            print(voice_filename)
            break

def detect_faces(project_name):
    project_folder = os.path.join('projects', project_name)
    config_file = os.path.join(project_folder, 'config.json')
    config = read_json(config_file)
    frames_folder = config['frames_folder']

    scenes_folders = [f for f in os.listdir(frames_folder) \
                     if os.path.isdir(os.path.join(frames_folder, f))]
    
    detector = FaceDetector()
    
    for scene_folder in scenes_folders:
        scene_folder = os.path.join(frames_folder, scene_folder)
        persons_folder = os.path.join(scene_folder, 'persons')
        recreate_folder(persons_folder)
        dump = dict()
        frame_files = glob.glob(scene_folder + '/*.jpg')
        frame_files = sorted(frame_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        for frame_file in frame_files:
            frame_num = int(os.path.splitext(os.path.basename(frame_file))[0])
            dump[frame_num] = dict()
            frame = cv2.imread(frame_file)
            faces = detector.detect(frame, face_det_tresh=0.2)
            for face_id, face in enumerate(faces):
                bbox = face[1]
                dump[frame_num][face_id] = bbox
                face_filename = os.path.join(persons_folder, f'face_{frame_num}_{face_id}.jpg')
                cv2.imwrite(face_filename, face[0])
        dump_filename = os.path.join(scene_folder, 'dump.json')
        dump = json.dumps(dump, ensure_ascii=False)
        with open(dump_filename, 'w', encoding='utf-8') as file:
            file.write(dump)

def use_lipsync(project_name, ignored_scenes):
    project_folder = os.path.join('projects', project_name)
    config_file = os.path.join(project_folder, 'config.json')
    config = read_json(config_file)

    wrapper = LipSync(config['fps'])

    frames_folder = config['frames_folder']
    #images = glob.glob(frames_folder + '/**/*.jpg', recursive=True)
    images = [
        image
        for image in glob.glob(frames_folder + '/**/*.jpg', recursive=True)
        if not 'persons' in image
    ]
    images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # To remove
    max_index = 50
    images = images[:max_index + 1]
    
    full_frames = []
    for image in images:
        frame = cv2.imread(image)
        full_frames.append(frame)

    scene_ids = [
        int(folder_name.replace("scene_", ""))
        for folder_name in os.listdir(frames_folder)
        if os.path.isdir(os.path.join(frames_folder, folder_name))
        and folder_name.startswith("scene_")
    ]
    scene_ids = sorted(scene_ids)

    faces_dict = dict()

    for scene_id in scene_ids:
        if not scene_id in ignored_scenes:
            persons_folder = os.path.join(frames_folder, f'scene_{scene_id}', 'persons')
            dump = read_json(os.path.join(frames_folder, f'scene_{scene_id}', 'dump.json'))
            for frame_num, values in dump.items():
                for id, bbox in values.items():
                    if int(frame_num) <= max_index and os.path.exists(os.path.join(persons_folder, f'face_{frame_num}_{id}.jpg')):
                        faces_dict[int(frame_num)] = {
                            'image': cv2.imread(os.path.join(persons_folder, f'face_{frame_num}_{id}.jpg')),
                            'bbox': bbox
                        }
                        break

    wrapper.process(audio_filename=config['merged_audio_filename'], full_frames=full_frames, faces_dict=faces_dict)

def merge_audio(project_name):
    project_folder = os.path.join('projects', project_name)
    config_file = os.path.join(project_folder, 'config.json')
    config = read_json(config_file)
    cloned = read_json(config['cloned_file'])
    orig_audio = AudioSegment.from_file(config['voice_audio_file'], format='wav')

    original_audio_duration = orig_audio.duration_seconds * 1000

    files_manager = FilesManager()

    updates = []

    for clone in cloned:
        updates.append({
            'start': clone['words'][0]['start'],
            'end': clone['words'][-1]['end'],
            'voice': clone['voice_file']
        })
    
    audio_segments = to_segments(updates, original_audio_duration)
    speech_final_audio = AudioSegment.silent(duration=0)
    for segment in audio_segments:
        if segment['empty']:
            duration = segment['end'] - segment['start']
            speech_final_audio += AudioSegment.silent(duration=duration)
        else:
            speech_final_audio += AudioSegment.from_file(segment['voice'])
    speech_audio_wav = files_manager.create_temp_file(suffix='.wav').name
    speech_final_audio.export(speech_audio_wav, format='wav')

    combined_audio = combine_audio(speech_audio_wav, config['noise_audio_file'])
    merged_audio_filename = os.path.join(project_folder, 'merged.wav')
    shutil.move(combined_audio, merged_audio_filename)

    config['merged_audio_filename'] = merged_audio_filename

    config = json.dumps(config, ensure_ascii=False)

    with open(config_file, 'w', encoding='utf-8') as file:
        file.write(config)

def render(project_name, out_video_filename):
    project_folder = os.path.join('projects', project_name)
    config_file = os.path.join(project_folder, 'config.json')
    config = read_json(config_file)
    frames_folder = config['frames_folder']
    images = glob.glob(frames_folder + '/**/*.jpg', recursive=True)
    images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    frames = []
    for image in images:
        frame = cv2.imread(image)
        frames.append(frame)

    temp_result_avi = to_avi(frames, config['fps'])
    merge(config['merged_audio_filename'], temp_result_avi, out_video_filename)

def detect_scenes(video_file):
    videoManager = VideoManager([video_file])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager)
    return sceneManager.get_scene_list(baseTimecode, start_in_scene=True)

def transcribe_audio_extended(whisper, audio_file, whisper_batch_size=16):
    audio = load_audio(audio_file)
    result = whisper.transcribe(audio, batch_size=whisper_batch_size)
    language = result['language']
    model_a, metadata = load_align_model(language_code=language, device=torch.device('cpu'))
    result = align(result['segments'], model_a, metadata, audio, torch.device('cpu'), return_char_alignments=False)
    return result, language

def transcribe_audio_extended_v2(audio_file):
    from whisper_timestamped import transcribe_timestamped
    return transcribe_timestamped('small', audio_file)
