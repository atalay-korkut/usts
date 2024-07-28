import random
import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import torch
from sammed.build_sam import sam_model_registry
from sammed.predictor_sammed import SammedPredictor
from argparse import Namespace

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = Namespace()
args.device = device
args.image_size = 256
args.encoder_adapter = True
args.sam_checkpoint = "/home/aorta-scan/Atalay/SAM_Med2D/pretrain_model/sam-med2d_b.pth"

def load_model(args):
    model = sam_model_registry["vit_b"](args).to(args.device)
    model.eval()
    predictor = SammedPredictor(model)
    return predictor

predictor_with_adapter = load_model(args)
args.encoder_adapter = False
predictor_without_adapter = load_model(args)

# Extract the first frame from the video
def extract_first_frame(video_file):
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Couldn't read the first frame from the video.")
    cap.release()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    return frame
# Define these globally
colors = [(255, 0, 0), (0, 255, 0)]  # List of color tuples (red, green)
markers = [cv2.MARKER_CROSS, cv2.MARKER_TILTED_CROSS]  # List of marker types

def run_sammed(input_image, selected_points, last_mask, adapter_type):
    predictor = predictor_with_adapter if adapter_type == "SAM-Med2D-B" else predictor_without_adapter

    image_pil = Image.fromarray(input_image)
    H, W, _ = input_image.shape
    predictor.set_image(input_image)
    centers = np.array([a for a, b in selected_points])
    point_coords = centers
    point_labels = np.array([b for a, b in selected_points])

    masks, _, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=last_mask,
        multimask_output=True
    )

    mask_image = Image.new('RGBA', (W, H), color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)
    for mask in masks:
        draw_mask(mask, mask_draw, random_color=False)
    image_draw = ImageDraw.Draw(image_pil)

    draw_point(selected_points, image_draw)

    image_pil = image_pil.convert('RGBA')
    image_pil.alpha_composite(mask_image)
    last_mask = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float, device=device))
    return [(image_pil, mask_image), last_mask]

def draw_mask(mask, draw, random_color=False):
    color = (30, 144, 255, 153) if not random_color else (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    nonzero_coords = np.transpose(np.nonzero(mask))
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

def draw_point(point, draw, r=5):
    for pt, label in point:
        x, y = pt
        fill_color = 'green' if label == 1 else 'red'
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill_color)

# Gradio application setup
block = gr.Blocks()
with block:
    with gr.Row():
        gr.Markdown(
            '''# SAM-Med2D!🚀
            SAM-Med2D is an interactive segmentation model based on the SAM model for medical scenarios.'''
        )
        adapter_type = gr.Dropdown(["SAM-Med2D-B", "SAM-Med2D-B_w/o_adapter"], value='SAM-Med2D-B', label="Select Adapter")

    with gr.Tab(label='Video'):
        with gr.Row().style(equal_height=True):
            with gr.Column():
                video_upload = gr.Video(label='Upload a Video')
                input_image = gr.Image(type="numpy", label="First Frame")
                original_image = gr.State(value=None)
                selected_points = gr.State([])
                last_mask = gr.State(None)

                def store_video_and_extract_frame(video_file):
                    frame = extract_first_frame(video_file)
                    return frame, [], None

                video_upload.upload(
                    store_video_and_extract_frame,
                    [video_upload],
                    [input_image, selected_points, last_mask]
                )

                undo_button = gr.Button('Undo point')
                point_type = gr.Radio(['foreground_point', 'background_point'], label='Point Labels')
                button = gr.Button("Run!")

            gallery_sammed = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery").style(preview=True, grid=2, object_fit="scale-down")

    def get_point(img, sel_pix, point_type, evt: gr.SelectData):
        sel_pix.append((evt.index, 1 if point_type == 'foreground_point' else 0))
        for point, label in sel_pix:
            cv2.drawMarker(img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        return img

    input_image.select(
        get_point,
        [input_image, selected_points, point_type],
        [input_image]
    )

    def undo_points(orig_img, sel_pix):
        temp = orig_img.copy()
        if len(sel_pix) != 0:
            sel_pix.pop()
            for point, label in sel_pix:
                cv2.drawMarker(temp, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        return temp, None if isinstance(temp, np.ndarray) else np.array(temp), None

    undo_button.click(
        undo_points,
        [input_image, selected_points],
        [input_image, last_mask]
    )

    button.click(fn=run_sammed, inputs=[input_image, selected_points, last_mask, adapter_type], outputs=[gallery_sammed, last_mask])

block.launch(debug=True, share=True, show_error=True)
