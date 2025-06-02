from model.ddpm import DDPM
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from skimage.color import lab2rgb, rgb2lab
import os
from datetime import datetime

class TestDDPM:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DDPM()  # Use default model configuration
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        self.size = 128
        
    def preprocess_image(self, image_path):
        """Convert input image to grayscale and prepare it for the model"""
        # Load and convert to RGB
        img = Image.open(image_path).convert("RGB")
        
        # Resize to model's expected size
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size), Image.BICUBIC),
        ])
        img = transform(img)
        
        # Convert to numpy array and then to LAB
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        
        # Extract L channel and normalize to [-1, 1]
        L = img_lab[:, :, 0]
        L = torch.from_numpy(L).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        L = (L / 50.0) - 1.0  # Normalize to [-1, 1]
        
        # Store original L for later use
        original_L = img_lab[:, :, 0].copy()
        
        return L.to(self.device), original_L
    
    def colorize(self, image_path, output_path=None):
        """Colorize a grayscale image and save the result"""
        # Preprocess the image
        L, original_L = self.preprocess_image(image_path)
        
        # Generate colored image
        with torch.no_grad():
            # Create input tensor with L channel and zero ab channels
            zeros_ab = torch.zeros(1, 2, self.size, self.size, device=self.device)
            x = torch.cat([L, zeros_ab], dim=1)  # [B, 3, H, W] tensor with L and zero ab
            
            # Custom sampling process with 1000 steps
            n_steps = 1000
            device = next(self.model.parameters()).device
            
            # Initialize with noise
            ab = torch.randn(1, 2, self.size, self.size, device=device)
            x = torch.cat([L, ab], dim=1)
            
            # Gradually denoise
            for i in range(n_steps):
                t = torch.ones(1, device=device, dtype=torch.long) * (n_steps - i - 1)
                # Get model prediction
                ab_pred = self.model.net(x, t)
                # Update ab channels
                x = torch.cat([L, ab_pred], dim=1)
            
            fake_ab = x[:, 1:, :, :]  # Extract final ab channels
            
            # Create final LAB image
            fake_lab = np.zeros((self.size, self.size, 3))
            fake_lab[:, :, 0] = original_L  # Use original L channel
            fake_lab[:, :, 1:] = fake_ab[0].cpu().numpy().transpose(1, 2, 0) * 110.0  # Denormalize and add predicted ab
        
        # Convert back to RGB
        fake_rgb = lab2rgb(fake_lab)
        fake_rgb = (fake_rgb * 255).astype(np.uint8)
        
        # Create PIL Image and resize to 256x256
        result_img = Image.fromarray(fake_rgb)
        result_img = result_img.resize((256, 256), Image.BICUBIC)
        
        # Save the result
        if output_path:
            result_img.save(output_path)
        
        return result_img

    def get_output_path(self, input_path):
        """Generate output path and create directory if it doesn't exist"""
        # Get directory and filename from input path
        input_filename = os.path.basename(input_path)
        name, ext = os.path.splitext(input_filename)
        
        # Create output directory
        output_dir = os.path.join('imgs/colored', 'ddpm')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename with timestamp
        output_filename = f"{name}_ddpm_colored_{ext}"
        
        return os.path.join(output_dir, output_filename)

def main():
    # Example usage
    ddpm = TestDDPM(model_path='checkpoints/ddpm.pt')
    
    # Test on a single image
    input_image = "imgs/grey/004.jpg"
    
    # Generate output path automatically
    output_image = ddpm.get_output_path(input_image)
    
    # Colorize the image
    colored_image = ddpm.colorize(input_image, output_image)
    print(f"Colored image saved to {output_image}")

if __name__ == "__main__":
    main()