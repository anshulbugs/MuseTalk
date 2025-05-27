import os
import cv2
import torch
import glob
import pickle
import numpy as np
from tqdm import tqdm
import copy
import json
import shutil
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image_prepare_material, get_image_blending

def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

@torch.no_grad()
class WebRTCAvatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation, 
                 version="v15", extra_margin=10, parsing_mode="jaw", 
                 left_cheek_width=90, right_cheek_width=90):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.version = version
        self.extra_margin = extra_margin
        self.parsing_mode = parsing_mode
        self.left_cheek_width = left_cheek_width
        self.right_cheek_width = right_cheek_width
        
        # Set paths based on version
        if version == "v15":
            self.base_path = f"./results/{version}/avatars/{avatar_id}"
        else:  # v1
            self.base_path = f"./results/avatars/{avatar_id}"
            
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": version,
            "extra_margin": extra_margin,
            "parsing_mode": parsing_mode,
            "left_cheek_width": left_cheek_width,
            "right_cheek_width": right_cheek_width
        }
        
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        
        # Initialize face parser
        if version == "v15":
            self.fp = FaceParsing(
                left_cheek_width=left_cheek_width,
                right_cheek_width=right_cheek_width
            )
        else:
            self.fp = FaceParsing()
        
        self.init()

    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                print(f"Avatar {self.avatar_id} exists, recreating...")
                shutil.rmtree(self.avatar_path)
                
            print(f"Creating avatar: {self.avatar_id}")
            osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
            self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                raise ValueError(f"Avatar {self.avatar_id} does not exist, you should set preparation to True")

            # Load existing avatar info
            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)

            # Check if parameters changed
            if avatar_info.get('bbox_shift') != self.avatar_info['bbox_shift']:
                print(f"bbox_shift changed for {self.avatar_id}, recreating...")
                shutil.rmtree(self.avatar_path)
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
            else:
                # Load existing materials
                self.load_materials()

    def load_materials(self):
        """Load pre-computed materials"""
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
            
        input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)
        
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
            
        input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(input_mask_list)

    def prepare_material(self):
        """Prepare avatar materials"""
        print("Preparing data materials...")
        
        # Save avatar info
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        # Extract frames from video
        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"Copying files from {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
                
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        print("Extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        
        # We'll need to get VAE from the models passed to us
        # For now, we'll prepare the structure and load latents later
        input_latent_list = []
        
        # Process each frame
        for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
            if bbox == coord_placeholder:
                continue
                
            x1, y1, x2, y2 = bbox
            if self.version == "v15":
                y2 = y2 + self.extra_margin
                y2 = min(y2, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]
                
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # We'll store the crop frame and compute latents later when VAE is available
            input_latent_list.append(resized_crop_frame)

        # Create cycles
        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        # Process masks
        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            x1, y1, x2, y2 = self.coord_list_cycle[i]
            mode = self.parsing_mode if self.version == "v15" else "raw"
            
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=self.fp, mode=mode)

            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle.append(crop_box)
            self.mask_list_cycle.append(mask)

        # Save coordinates and masks
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        # Note: We'll compute and save latents separately when VAE is available
        print("Material preparation complete (latents will be computed when VAE is available)")

    def compute_latents(self, vae):
        """Compute latents using the provided VAE"""
        print(f"Computing latents for avatar {self.avatar_id}...")
        
        if hasattr(self, 'input_latent_list_cycle') and isinstance(self.input_latent_list_cycle[0], torch.Tensor):
            # Latents already computed
            return
            
        latent_list = []
        for crop_frame in self.input_latent_list_cycle:
            if isinstance(crop_frame, np.ndarray):
                latents = vae.get_latents_for_unet(crop_frame)
                latent_list.append(latents)
            else:
                latent_list.append(crop_frame)  # Already a tensor
                
        self.input_latent_list_cycle = latent_list
        
        # Save latents
        torch.save(self.input_latent_list_cycle, self.latents_out_path)
        print(f"Latents computed and saved for avatar {self.avatar_id}")