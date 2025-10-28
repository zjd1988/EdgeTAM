import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sam2.build_sam import build_sam2_video_predictor
from moviepy import ImageSequenceClip
from PIL import Image

checkpoint = "/data1/zhaojd-a/github_codes/EdgeTAM/checkpoints/edgetam.pt"
model_cfg = "configs/edgetam.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# video info
video_input_path = "/data1/zhaojd-a/github_codes/EdgeTAM/examples/01_dog.mp4"
video_ouput_path = "/data1/zhaojd-a/github_codes/EdgeTAM/result"

# obj info
box = [[450, 386], [706, 549]]
obj_id = 1

def show_mask(mask, obj_id=None, random_color=False, convert_to_image=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask = (mask * 255).astype(np.uint8)
    if convert_to_image:
        mask = Image.fromarray(mask, "RGBA")
    return mask


def get_video_frames_and_fps(video_path):
    # Open the video file
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print("Error: Could not open video.")
        return [], 0

    # Get the FPS of the video
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    all_frames = []
    frame_number = 0
    first_frame = None
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)

        # Store the first frame
        if frame_number == 0:
            first_frame = frame
        all_frames.append(frame)

        frame_number += 1
    return all_frames, video_fps

# get video frames and fps
video_frames, video_fps = get_video_frames_and_fps(video_input_path)

# tam process
with torch.inference_mode():
    state = predictor.init_state(video_input_path)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, box=box)

    output_frames = []
    # propagate the prompts to get masklets throughout the video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        video_segment = {
            out_obj_id: (masks[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(object_ids)
        }

        transparent_background = Image.fromarray(video_frames[frame_idx]).convert("RGBA")
        out_mask = video_segment[obj_id]
        mask_image = show_mask(out_mask)
        output_frame = Image.alpha_composite(transparent_background, mask_image)
        output_frame = np.array(output_frame)
        output_frames.append(output_frame)

    # Create a video clip from the image sequence
    clip = ImageSequenceClip(output_frames, fps=video_fps)
    # Write the result to a file
    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
    final_vid_output_path = f"output_video_{unique_id}.mp4"
    final_vid_output_path = os.path.join(video_ouput_path, final_vid_output_path)

    # Write the result to a file
    clip.write_videofile(final_vid_output_path, codec="libx264")