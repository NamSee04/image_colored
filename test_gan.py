from model.gan import MainModel, init_model, build_res_unet
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
from skimage.color import lab2rgb, rgb2lab
import os
from fastai.vision.learner import create_body
from torchvision.models import resnet18
from fastai.vision.models.unet import DynamicUnet

SIZE = 256

class TestGAN:
    def __init__(self, model_path=None, net_G=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if net_G is None:
            self.model = MainModel()
        else:
            self.model = MainModel(net_G=net_G)
        
        if model_path and os.path.exists(model_path):
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Handle DataParallel state dict
            if all(k.startswith('module.') for k in state_dict.keys()):
                # Remove 'module.' prefix if it exists
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            # Load the state dict into the model
            self.model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: No model path provided or model file not found")
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((SIZE, SIZE), Image.BICUBIC),
        ])
    
    def preprocess_image(self, image_path):
        """Preprocess image for the model"""
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        return L.unsqueeze(0).to(self.device)  # Add batch dimension
    
    def colorize(self, image_path, output_path=None):
        """Colorize a grayscale image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
            
        # Create output directory if needed
        if output_path is None:
            output_dir = os.path.join('imgs/colored', 'gan')
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, filename)
        
        # Preprocess image
        L = self.preprocess_image(image_path)
        
        # Generate color
        with torch.no_grad():
            self.model.net_G.eval()
            fake_color = self.model.net_G(L)
            
        # Convert to image
        fake_color = fake_color.cpu().squeeze(0) * 110.  # Scale back to LAB range
        L = L.cpu().squeeze(0) * 50. + 50.  # Scale back to LAB range
        
        fake_im = torch.cat([L, fake_color], dim=0).numpy()
        fake_im = np.transpose(fake_im, (1, 2, 0))
        fake_im = lab2rgb(fake_im)
        fake_im = (fake_im * 255).astype(np.uint8)
        
        # Convert to PIL Image and resize
        img = Image.fromarray(fake_im)
        # Get original image size
        original_img = Image.open(image_path)
        original_size = original_img.size
        # Resize to original size
        img = img.resize(original_size, Image.BICUBIC)
        
        # Save the colored image
        img.save(output_path)
        print(f"Saved colored image to: {output_path}")
        return output_path

def main():
    # Example usage
    model_path = 'check_points/final_gan.pt'
    
    def build_res_unet(n_input=1, n_output=2, size=256):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = resnet18(pretrained=True)
        body = create_body(resnet, n_in=n_input, cut=-2)
        net_G = DynamicUnet(body, n_output, (size, size)).to(device)
        return net_G
    
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    test_gan = TestGAN(model_path=model_path, net_G=net_G)
    
    # Process an image
    input_image = "imgs/grey/006.jpg"  # Use the grayscale image we created
    output_image = test_gan.colorize(input_image)
    print(f"\nSuccessfully colored image: {input_image}")
    print(f"Output saved to: {output_image}")

if __name__ == "__main__":
    main()