import random
import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import torch
from argparse import Namespace
from sammed.build_sam import sam_model_registry
from sammed.predictor_sammed import SammedPredictor

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = Namespace()
args.device = device
args.image_size = 256
args.encoder_adapter = True
args.sam_checkpoint = "/home/aorta-scan/Atalay/SAM_Med2D/pretrain_model/sam-med2d_b.pth"

# Load models
def load_model(args):
    model = sam_model_registry["vit_b"](args).to(args.device)
    model.eval()
    predictor = SammedPredictor(model)
    return predictor

predictor_with_adapter = load_model(args)
args.encoder_adapter = False
predictor_without_adapter = load_model(args)

# Run segmentation
def run_sammed(input_image, selected_points, last_mask, adapter_type):
    predictor = predictor_with_adapter if adapter_type == "SAM-Med2D-B" else predictor_without_adapter
    image_pil = Image.fromarray(input_image)  # Convert numpy image to PIL image for drawing
    image = input_image
    H, W, _ = image.shape
    predictor.set_image(image)

    centers = np.array([a for a, b in selected_points])
    point_labels = np.array([b for a, b in selected_points])

    masks, _, logits = predictor.predict(
        point_coords=centers,
        point_labels=point_labels,
        mask_input=last_mask,
        multimask_output=True
    )

    mask_image = Image.new('RGBA', (W, H), color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)
    for mask in masks:
        draw_mask(mask, mask_draw, random_color=False)

    image_pil = image_pil.convert('RGBA')
    image_pil.alpha_composite(mask_image)
    last_mask = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float, device=device))
    return [(image_pil, mask_image), last_mask]

# Draw masks
def draw_mask(mask, draw, random_color=False):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153) if random_color else (30, 144, 255, 153)
    nonzero_coords = np.transpose(np.nonzero(mask))
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

# Draw points
def draw_point(points, draw, r=5):
    for point, label in points:
        x, y = point
        color = 'green' if label == 1 else 'red'
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

# Generate points grid
def generate_all_points(image, n_points_per_side=4):
    H, W, _ = image.shape
    x_coords = np.linspace(0, W - 1, n_points_per_side, dtype=int)
    y_coords = np.linspace(0, H - 1, n_points_per_side, dtype=int)
    point_grid = [(int(x), int(y)) for x in x_coords for y in y_coords]
    selected_points = [(point, 1) for point in point_grid]

    for point, _ in selected_points:
        cv2.drawMarker(image, point, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    return image, selected_points, None

# Undo the selected points
def undo_points(orig_img, sel_pix):
    temp = orig_img.copy()
    if len(sel_pix) != 0:
        sel_pix.pop()
        for point, label in sel_pix:
            cv2.drawMarker(temp, point, (0, 255, 0) if label == 1 else (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    return temp, None, None

# Process an example image
def process_example(img):
    return img, [], None

# Store uploaded image
def store_img(img):
    return img, [], None

# Get a point selected by the user
def get_point(img, sel_pix, point_type, evt: gr.SelectData):
    colors = [(0, 255, 0), (255, 0, 0)]
    markers = [cv2.MARKER_CROSS, cv2.MARKER_TILTED_CROSS]

    if point_type == 'foreground_point':
        sel_pix.append((evt.index, 1))
    elif point_type == 'background_point':
        sel_pix.append((evt.index, 0))
    else:
        sel_pix.append((evt.index, 1))
    for point, label in sel_pix:
        cv2.drawMarker(img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
    return img

# Gradio UI setup
block = gr.Blocks()
with block:
    with gr.Row():
        gr.Markdown(
            '''# SAM-Med2D!🚀
            SAM-Med2D is an interactive segmentation model based on the SAM model for medical scenarios. More info on [**GitHub**](https://github.com/uni-medical/SAM-Med2D/tree/main).
            '''
        )
        with gr.Row():
            adapter_type = gr.Dropdown(["SAM-Med2D-B", "SAM-Med2D-B_w/o_adapter"], value='SAM-Med2D-B', label="Select Adapter")

    with gr.Tab(label='Image'):
        with gr.Row().style(equal_height=True):
            with gr.Column():
                original_image = gr.State(value=None)
                input_image = gr.Image(type="numpy")
                selected_points = gr.State([])
                last_mask = gr.State(None)
                with gr.Column():
                    gr.Markdown('Click on the image to select points prompt.')
                    undo_button = gr.Button('Undo point')
                    everything_button = gr.Button('Everything')
                radio = gr.Radio(['foreground_point', 'background_point'], label='point labels')
                button = gr.Button("Run!")

            gallery_sammed = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery").style(preview=True, grid=2, object_fit="scale-down")

    input_image.upload(store_img, [input_image], [original_image, selected_points, last_mask])
    input_image.select(get_point, [input_image, selected_points, radio], [input_image])
    undo_button.click(undo_points, [original_image, selected_points], [input_image, last_mask])
    everything_button.click(generate_all_points, [input_image], [input_image, selected_points, last_mask])
    button.click(fn=run_sammed, inputs=[original_image, selected_points, last_mask, adapter_type], outputs=[gallery_sammed, last_mask])

block.launch(debug=True, share=True, show_error=True)
