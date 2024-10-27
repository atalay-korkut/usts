import os
import cv2
import torch
import numpy as np
from PIL import Image
from SegTrackerSAM import SegTrackerSAM
from tool.transfer_tools import draw_outline, draw_points, mask2bbox
from seg_track_anything import tracking_objects_in_video, draw_mask
from model_args import segtracker_args, sam_args, aot_args
from seg_track_anything import aot_model2ckpt
from sammed.build_sam import sam_model_registry
from sammed.predictor_sammed import SammedPredictor
import argparse
from argparse import Namespace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = Namespace()
args.device = device
args.image_size = 256
args.sam_checkpoint = "/home/aorta-scan/Atalay/SAM_Med2D/pretrain_model/sam-med2d_b.pth"

def load_model(args):
    model = sam_model_registry["vit_b"](args).to(args.device)
    model.eval()
    predictor = SammedPredictor(model)
    return predictor

args.encoder_adapter = True
predictor_with_adapter = load_model(args)
args.encoder_adapter = False
predictor_without_adapter = load_model(args)

def run_sammed(input_image, selected_points, last_mask, adapter_type):
    predictor = predictor_with_adapter if adapter_type == "SAM-Med2D-B" else predictor_without_adapter
    image_pil = Image.fromarray(input_image)
    image = input_image
    H, W, _ = image.shape
    predictor.set_image(image)
    centers = np.array([a for a, b in selected_points])
    point_coords = centers
    point_labels = np.array([b for a, b in selected_points])

    masks, _, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=last_mask,
        multimask_output=True
    )


def get_meta_from_video(input_video):
    if input_video is None:
        return None, None, None, ""
    
    cap = cv2.VideoCapture(input_video)
    _, first_frame = cap.read()
    cap.release()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    return first_frame, first_frame, first_frame, ""

def get_meta_from_img_seq(input_img_seq):
    if input_img_seq is None:
        return None, None, None, ""
    
    file_name = input_img_seq.split('/')[-1].split('.')[0]
    file_path = f'./assets/{file_name}'
    if os.path.isdir(file_path):
        os.system(f'rm -r {file_path}')
    os.makedirs(file_path)
    os.system(f'unzip {input_img_seq} -d ./assets ')
    
    imgs_path = sorted([os.path.join(file_path, img_name) for img_name in os.listdir(file_path)])
    first_frame = imgs_path[0]
    first_frame = cv2.imread(first_frame)
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    return first_frame, first_frame, first_frame, ""

def SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
    with torch.cuda.amp.autocast():
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)
        Seg_Tracker.first_frame_mask = predicted_mask
    return Seg_Tracker

def load_model(args):
    model = sam_model_registry["vit_b"](args).to(args.device)
    model.eval()
    predictor = SammedPredictor(model)
    return predictor

args.encoder_adapter = True
predictor_with_adapter = load_model(args)
args.encoder_adapter = False
predictor_without_adapter = load_model(args)

def run_sammed(input_image, selected_points, last_mask, adapter_type):
    predictor = predictor_with_adapter if adapter_type == "SAM-Med2D-B" else predictor_without_adapter
    image_pil = Image.fromarray(input_image)
    image = input_image
    H, W, _ = image.shape
    predictor.set_image(image)
    centers = np.array([a for a, b in selected_points])
    point_coords = centers
    point_labels = np.array([b for a, b in selected_points])

    masks, _, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=last_mask,
        multimask_output=True
    )

def init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame):
    if origin_frame is None:
        return None, origin_frame, [[], []], ""

    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]
    aot_args["long_term_mem_gap"] = long_term_mem
    aot_args["max_len_long_term"] = max_len_long_term

    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    sam_args["generator_args"]["points_per_side"] = points_per_side

    Seg_Tracker = SegTrackerSAM(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()
    return Seg_Tracker, origin_frame, [[], []], ""

def sam_click_manual(Seg_Tracker, origin_frame, click_stack, coordinates_input, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side):
    try:
        coordinates = coordinates_input.split(";")
        for coord in coordinates:
            x, y, mode = map(int, coord.split(","))
            click_stack[0].append([x, y])
            click_stack[1].append(mode)
    except Exception as e:
        print(f"Error parsing coordinates: {e}")
        return Seg_Tracker, origin_frame, click_stack

    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)

    prompt = {
        "points_coord": click_stack[0],
        "points_mode": click_stack[1],
        "multimask": "True",
    }

    predicted_mask, masked_frame = Seg_Tracker.seg_acc_click(
        origin_frame=origin_frame,
        coords=np.array(prompt["points_coord"]),
        modes=np.array(prompt["points_mode"]),
        multimask=prompt["multimask"],
    )
    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)
    return Seg_Tracker, masked_frame, click_stack

def sam_bbox_manual(Seg_Tracker, origin_frame, bbox_coordinates_input, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side):
    try:
        coords = bbox_coordinates_input.split(";")
        x1, y1 = map(int, coords[0].split(","))
        x2, y2 = map(int, coords[1].split(","))
        bbox = [[x1, y1], [x2, y2]]
    except Exception as e:
        print(f"Error parsing coordinates: {e}")
        return Seg_Tracker, origin_frame

    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)

    predicted_mask, masked_frame = Seg_Tracker.seg_acc_bbox(origin_frame, bbox)
    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)
    return Seg_Tracker, masked_frame

def tracking_objects(Seg_Tracker, input_video, input_img_seq, fps):
    return tracking_objects_in_video(Seg_Tracker, input_video, input_img_seq, fps)

def cli_sammedtrack(input_video=None, input_img_seq=None, aot_model="r50_deaotl", long_term_mem=9999, max_len_long_term=9999, sam_gap=100, max_obj_num=255, points_per_side=16, point_mode="Positive", coordinates_input=None, bbox_coordinates_input=None, drawing_board=None, fps=8):
    Seg_Tracker = None
    origin_frame = None
    click_stack = [[], []]

    if input_video is not None:
        origin_frame, _, _, _ = get_meta_from_video(input_video)
    elif input_img_seq is not None:
        origin_frame, _, _, _ = get_meta_from_img_seq(input_img_seq)

    Seg_Tracker, origin_frame, click_stack, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)

    if coordinates_input is not None:
        Seg_Tracker, _, click_stack = sam_click_manual(Seg_Tracker, origin_frame, click_stack, coordinates_input, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side)

    if bbox_coordinates_input is not None:
        Seg_Tracker, _ = sam_bbox_manual(Seg_Tracker, origin_frame, bbox_coordinates_input, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side)

    if input_video or input_img_seq:
        output_video, output_mask = tracking_objects(Seg_Tracker, input_video, input_img_seq, fps)
        return output_video, output_mask

    return Seg_Tracker, origin_frame, click_stack

#point inputs x,y,mode (e.g., 100,100,1) ,The mode indicates whether the point is positive (1) or negative (0),
#bbox inputs x1,y1,x2,y2 (e.g., 100,100;200,200)
if __name__ == "__main__":
    output_video, output_mask = cli_sammedtrack(
        input_video="/home/aorta-scan/Atalay/case789.mp4",
        coordinates_input="100,100,1;200,200,0",
        #bbox_coordinates_input="100,100;200,200",
        fps=10
    )
    print("Tracking completed!")
