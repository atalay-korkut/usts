import gradio as gr
import numpy as np
from PIL import Image
import torch
from sammed.build_sam import sam_model_registry
from sammed.predictor_sammed import SammedPredictor
from argparse import Namespace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = Namespace()
args.device = device
args.image_size = 256
args.sam_checkpoint = "/home/aorta-scan/Atalay/SAM_Med2D/pretrain_model/sam-med2d_b.pth"

# Load SAM models
def load_model(args):
    model = sam_model_registry["vit_b"](args).to(args.device)
    model.eval()
    return SammedPredictor(model)

args.encoder_adapter = True
predictor_with_adapter = load_model(args)
args.encoder_adapter = False
predictor_without_adapter = load_model(args)

# Run SAM segmentation
def run_sammed(input_image, selected_points, adapter_type):
    predictor = predictor_with_adapter if adapter_type == "SAM-Med2D-B" else predictor_without_adapter
    image_pil = Image.fromarray(input_image)
    H, W, _ = input_image.shape
    predictor.set_image(input_image)
    centers = np.array([a for a, b in selected_points])
    point_coords = centers
    point_labels = np.array([b for a, b in selected_points])

    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )

    # Draw the mask on the image
    mask_image = Image.new('RGBA', (W, H), color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)
    for mask in masks:
        draw_mask(mask, mask_draw)

    image_pil = image_pil.convert('RGBA')
    image_pil.alpha_composite(mask_image)

    return image_pil

# Utility functions for drawing
def draw_mask(mask, draw):
    nonzero_coords = np.transpose(np.nonzero(mask))
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=(30, 144, 255, 153))

# Gradio app
def seg_track_app():
    with gr.Blocks() as app:
        gr.Markdown(
            '''
            <div style="text-align:center;">
                <span style="font-size:2em; font-weight:bold;">Simple SAM-Track App</span>
            </div>
            '''
        )
        
        click_stack = gr.State([[], []])  # Store clicks
        input_first_frame = gr.Image(label='Segment result of first frame', interactive=True)
        adapter_type = gr.Radio(
            choices=["SAM-Med2D-B", "No Adapter"],
            value="SAM-Med2D-B",
            label="Adapter Type"
        )

        # Function to handle clicks
        def sam_click(image, clicks, adapter):
            if not clicks:
                return image
            click_stack[0].append([clicks[-1][0], clicks[-1][1]])
            click_stack[1].append(1 if clicks[-1][2] == 'Positive' else 0)
            return run_sammed(np.array(image), list(zip(click_stack[0], click_stack[1])), adapter)

        input_first_frame.select(
            fn=sam_click,
            inputs=[input_first_frame, gr.State([[], []]), adapter_type],
            outputs=input_first_frame
        )

        app.launch(debug=True)

if __name__ == "__main__":
    seg_track_app()
