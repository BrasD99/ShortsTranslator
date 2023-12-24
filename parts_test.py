from parts import *
from common.helpers import merge

project_name = 'Tobai'
video_filename = '/Users/brasd99/Downloads/IMG_0093.mp4'

#initial_process(project_name, video_filename)
#dereverb(project_name)
#translate(project_name, 'english')
#clone(project_name, 'english')
#re_clone(project_name, 0, 'english')
#merge_audio(project_name)
#render(project_name, 'out_2.mp4')
#detect_faces(project_name)
#use_lipsync(project_name, [])

merge('merged.wav', 'result.avi', 'final.mp4')