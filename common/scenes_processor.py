from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from moviepy.video.io.VideoFileClip import VideoFileClip
from common.detector import FaceDetector
import cv2

class ScenesProcessor:
    def process(self, video_file):
        detector = FaceDetector()
        scenes = self.detect_scenes(video_file)
        orig_clip = VideoFileClip(video_file, verbose=False)
        frames = dict()
        for scene_id, scene in enumerate(scenes):
            frames[scene_id] = {
                'frames': []
            }

            start, end = scene
            subclip = orig_clip.subclip(start.frame_num / orig_clip.fps, end.frame_num / orig_clip.fps)
            
            for frame in subclip.iter_frames():
                frames[scene_id]['frames'].append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            first_frame = frames[scene_id]['frames'][0]
            cv2.imshow("Scene | Are you agree?", first_frame)
            key = cv2.waitKey(0)
            if key == 13 or chr(key & 255) == 'y':
                cv2.destroyAllWindows()
                print("Agreed")
                # Detecting faces
                current_frame_num = 1
                for frame in subclip.iter_frames():
                    frame_info = {
                        'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    }
                    faces = detector.detect(frame, face_det_tresh=0.2)
                    for face in faces:
                        image = cv2.cvtColor(face[0], cv2.COLOR_BGR2RGB)
                        cv2.imshow(f"Face | Are you agree? {current_frame_num} / {end.frame_num}", image)
                        key = cv2.waitKey(0)
                        if key == 13 or chr(key & 255) == 'y':
                            frame_info['face'] = {
                                'image': image,
                                'bbox': face[1]
                            }
                            cv2.destroyAllWindows()
                            break
                    frames[scene_id]['frames'].append(frame_info)
                    current_frame_num += 1
            else:
                cv2.destroyAllWindows()
                print("Not agreed")
                for frame in subclip.iter_frames():
                    frames[scene_id]['frames'].append({
                        'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    })
        cv2.destroyAllWindows()
        return frames
    
    def detect_scenes(self, video_file):
        videoManager = VideoManager([video_file])
        statsManager = StatsManager()
        sceneManager = SceneManager(statsManager)
        sceneManager.add_detector(ContentDetector())
        baseTimecode = videoManager.get_base_timecode()
        videoManager.set_downscale_factor()
        videoManager.start()
        sceneManager.detect_scenes(frame_source=videoManager)
        return sceneManager.get_scene_list(baseTimecode, start_in_scene=True)