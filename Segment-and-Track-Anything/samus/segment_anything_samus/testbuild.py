import build_sam_us as samus
import modeling

class Args:
    def __init__(self):
        self.encoder_input_size = 1024  # Set based on your image size
        self.encoder_type = "ViT"  # Example encoder type
        self.patch_size = self.encoder_input_size // 32  # Derive from encoder input size
        self.mlp_ratio = 4  # Default MLP ratio
        self.window_size = 14  # Default window size
        # Other necessary fields or parameters
args = Args()  # Create an instance with default values


model = samus.build_samus_vit_b(args,checkpoint="checkpoints/SAMUS__399.pth")

print(model.eval())