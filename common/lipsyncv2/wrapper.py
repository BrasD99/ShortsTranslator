from typing import Any
from common.lipsyncv2.third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from common.lipsyncv2.third_part.GFPGAN.gfpgan import GFPGANer
from common.lipsyncv2.utils import audio
import torch
import cv2
from PIL import Image
import numpy as np
from common.lipsyncv2.third_part.face3d.extract_kp_videos import KeypointExtractor
from common.lipsyncv2.utils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect, load_model, options, split_coeff, \
                                  trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict
from common.lipsyncv2.third_part.face3d.util.load_mats import load_lm3d
from tqdm import tqdm
from common.lipsyncv2.third_part.face3d.util.preprocess import align_img
from scipy.io import loadmat
import argparse
from common.lipsyncv2.utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
import os
from pathlib import Path

class LipSync:
    def __init__(self, fps):
        self.fps = fps
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.enhancer = FaceEnhancement(base_dir='common/lipsyncv2/checkpoints', size=512, model='GPEN-BFR-512', \
            use_sr=False, sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=self.device)
        self.restorer = GFPGANer(model_path='common/lipsyncv2/checkpoints/GFPGANv1.3.pth', upscale=1, arch='clean', \
            channel_multiplier=2, bg_upsampler=None)
        self.base_name = 'test'
        self.net_recon = load_face3d_net('common/lipsyncv2/checkpoints/face3d_pretrain_epoch_20.pth', self.device)
        self.lm3d_std = load_lm3d('common/lipsyncv2/checkpoints/BFM')
        data = {
            'DNet_path': 'common/lipsyncv2/checkpoints/DNet.pt',
            'LNet_path': 'common/lipsyncv2/checkpoints/LNet.pth',
            'ENet_path': 'common/lipsyncv2/checkpoints/ENet.pth',
            'face3d_net_path': 'common/lipsyncv2/checkpoints/face3d_pretrain_epoch_20.pth',
            'exp_img': 'neutral',
            'fps': self.fps,
            'pads': [0, 20, 0, 0],
            'face_det_batch_size': 4,
            'LNet_batch_size': 4,
            'img_size': 384,
            'crop': [0, -1, 0, -1],
            'box': [-1, -1, -1, -1],
            'nosmooth': False,
            'static': False,
            'up_face': 'original',
            'one_shot': False,
            'without_rl1': False,
            'tmp_dir': 'temp',
            're_preprocess': False
        }

        self.args = argparse.Namespace(**data)
        self.D_Net, self.model = load_model(self.args, self.device)
        self.tmp_dir = 'temp'

    def process(self, full_frames, faces_dict, audio_filename):
        # Для лиц необходимо вычислить landmarks
        kp_extractor = KeypointExtractor()
        # Нужно конвертнуть изображения лиц в PIL
        faces_pil = [Image.fromarray(cv2.resize(face['image'], (256, 256))) for face in faces_dict.values()]

        current_folder = Path().absolute()
        temp_folder = os.path.join(current_folder, 'temp')

        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)

        lm = kp_extractor.extract_keypoint(faces_pil, temp_folder + '/' + self.base_name + '_landmarks.txt')

        video_coeffs = []

        for idx in tqdm(range(len(faces_pil)), desc="[Step 2] 3DMM Extraction In Video"):
            face = faces_pil[idx]
            W, H = face.size
            lm_idx = lm[idx].reshape([-1, 2])
            if np.mean(lm_idx) == -1:
                lm_idx = (self.lm3d_std[:, :2]+1) / 2.
                lm_idx = np.concatenate([lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
            else:
                lm_idx[:, -1] = H - 1 - lm_idx[:, -1]
            
            trans_params, im_idx, lm_idx, _ = align_img(face, lm_idx, self.lm3d_std)
            trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
            im_idx_tensor = torch.tensor(np.array(im_idx)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0) 
            with torch.no_grad():
                coeffs = split_coeff(self.net_recon(im_idx_tensor))
            pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
            pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'],\
                                pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
            video_coeffs.append(pred_coeff)
        
        semantic_npy = np.array(video_coeffs)[:,0]
        np.save(temp_folder + '/' + self.base_name + '_coeffs.npy', semantic_npy)

        expression = torch.tensor(loadmat(os.path.join(current_folder, 'common/lipsyncv2/checkpoints/expression.mat'))['expression_center'])[0]

        imgs = []

        for idx in tqdm(range(len(faces_pil)), desc="[Step 3] Stabilize the expression In Video"):
            source_img = trans_image(faces_pil[idx]).unsqueeze(0).to(self.device)
            semantic_source_numpy = semantic_npy[idx:idx+1]
            ratio = find_crop_norm_ratio(semantic_source_numpy, semantic_npy)
            coeff = transform_semantic(semantic_npy, idx, ratio).unsqueeze(0).to(self.device)
            coeff[:, :64, :] = expression[None, :64, None].to(self.device)
            with torch.no_grad():
                output = self.D_Net(source_img, coeff)
            img_stablized = np.uint8((output['fake_image'].squeeze(0).permute(1,2,0).cpu().clamp_(-1, 1).numpy() + 1 )/2. * 255)
            imgs.append(cv2.cvtColor(img_stablized, cv2.COLOR_RGB2BGR))
        
        np.save(temp_folder + '/' + self.base_name + '_stablized.npy', imgs)

        wav = audio.load_wav(audio_filename, 16000)
        mel = audio.melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_step_size, mel_idx_multiplier, i, mel_chunks = 16, 80./self.fps, 0, []
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        full_frames = full_frames[:len(mel_chunks)]

        imgs_enhanced = []
        for idx in tqdm(range(len(imgs)), desc='[Step 4] Reference Enhancement'):
            img = imgs[idx]
            pred, _, _ = self.enhancer.process(img, img, face_enhance=True, possion_blending=False)
            imgs_enhanced.append(pred)
        
        # To original dict
        id = 0
        for frame_num in faces_dict:
            faces_dict[frame_num]['image'] = imgs_enhanced[id]
            id += 1
        
        frame_h, frame_w = full_frames[0].shape[:-1]

        # Datagen...
        output_filename = 'result.avi'
        print(output_filename)

        out = cv2.VideoWriter(
            output_filename, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, (frame_w, frame_h)
        )

        gen = self.datagen(faces_dict, full_frames, mel_chunks)

        for i, (img_batch, mel_batch, bbox_batch, full_frame_batch, flags_batch) in \
            enumerate(tqdm(gen, desc='[Step 5] Lip Synthesis', total=int(np.ceil(float(len(full_frames)) / self.args.LNet_batch_size)))):
            
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
            
            with torch.no_grad():
                _, reference = torch.split(img_batch, 3, dim=1)
                preds, _ = self.model(mel_batch, img_batch, reference)
                preds = torch.clamp(preds, 0, 1)
            
            preds = preds.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            offset = 0
            for j, flag in enumerate(flags_batch):
                full_frame = full_frame_batch[j]
                if flag:
                    x1, y1, w, h = bbox_batch[offset]
                    pred = preds[offset]
                    pred = cv2.resize(pred.astype(np.uint8), (w, h))
                    xf = full_frame.copy()
                    full_frame[y1:y1+h, x1:x1+w] = pred
                    _, _, restored_img = self.restorer.enhance(
                        full_frame, has_aligned=False, only_center_face=True, paste_back=True)
                    mm = [0,   0,   0,   0,   0,   0,   0,   0,   0,  0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
                    mouse_mask = np.zeros_like(restored_img)
                    tmp_mask = self.enhancer.faceparser.process(restored_img[y1:y1+h, x1:x1+w], mm)[0]
                    mouse_mask[y1:y1+h, x1:x1+w] = cv2.resize(tmp_mask, (w, h))[:, :, np.newaxis] / 255.
                    height, width = full_frame.shape[:2]
                    restored_img, full_frame, full_mask = [cv2.resize(x, (512, 512)) for x in (restored_img, full_frame, np.float32(mouse_mask))]
                    img = Laplacian_Pyramid_Blending_with_mask(restored_img, full_frame, full_mask[:, :, 0], 10)
                    pp = np.uint8(cv2.resize(np.clip(img, 0 , 255), (width, height)))
                    pp, _, _ = self.enhancer.process(pp, xf, bbox=[y1, y1 + h, x1, x1 + w], face_enhance=False, possion_blending=True)
                    out.write(pp)
                    offset += 1
                else:
                    out.write(full_frame)

        out.release()

    def datagen(self, faces_dict, frames, mels):
        img_batch, ref_batch, mel_batch, bbox_batch, full_frame_batch, flags = [], [], [], [], [], []

        for idx in range(len(frames)):
            if idx in faces_dict:
                face = faces_dict[idx]['image']
                bbox = faces_dict[idx]['bbox']
                x1, y1, w, h = bbox
                orig_face = frames[idx][y1:y1+h, x1:x1+w]
                face = cv2.resize(face, (self.args.img_size, self.args.img_size))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                orig_face = cv2.resize(orig_face, (self.args.img_size, self.args.img_size))
                img_batch.append(orig_face)
                ref_batch.append(face)
                bbox_batch.append(bbox)
                mel_batch.append(mels[idx])
                flags.append(True)
            else:
                flags.append(False)

            full_frame_batch.append(frames[idx].copy())
        
            if len(img_batch) >= self.args.LNet_batch_size:
                img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
                
                img_masked = img_batch.copy()

                img_masked[:, self.args.img_size//2:] = 0
                img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.

                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, bbox_batch, full_frame_batch, flags

                img_batch, mel_batch, bbox_batch, full_frame_batch, ref_batch, flags  = [], [], [], [], [], []
            
        if len(img_batch) > 0:
            img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
            
            img_masked = img_batch.copy()

            img_masked[:, self.args.img_size//2:] = 0
            img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.

            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, bbox_batch, full_frame_batch, flags
