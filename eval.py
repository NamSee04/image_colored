import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from torchvision import transforms
from scipy import linalg
from tqdm import tqdm
from PIL import Image
import os
from skimage.color import lab2rgb

from model.gan import MainModel
from dataset.Dataset import ColorizationDataset, make_dataloaders

class FIDEvaluator:
    def __init__(self, device):
        self.device = device
        # Load Inception model
        self.inception_model = inception_v3(pretrained=True, transform_input=False)
        self.inception_model.fc = torch.nn.Identity()  # Remove final classification layer
        self.inception_model = self.inception_model.to(device)
        self.inception_model.eval()
        
        # Define transforms for Inception model
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def get_features(self, images):
        """Extract features from images using Inception model"""
        features = []
        with torch.no_grad():
            for img in images:
                img = img.to(self.device)
                feature = self.inception_model(img)
                features.append(feature.cpu().numpy())
        return np.concatenate(features, axis=0)
    
    def calculate_statistics(self, features):
        """Calculate mean and covariance matrix of features"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, mu1, sigma1, mu2, sigma2):
        """Calculate FID score between two distributions"""
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
    
    def lab_to_rgb(self, L, ab):
        """Convert LAB to RGB"""
        L = (L + 1.) * 50.
        ab = ab * 110.
        Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
        rgb_imgs = []
        for img in Lab:
            img_rgb = np.clip(lab2rgb(img), 0, 1)
            rgb_imgs.append(img_rgb)
        return np.array(rgb_imgs)

def evaluate_fid(model_path, test_paths, batch_size=32, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = MainModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Initialize FID evaluator
    fid_evaluator = FIDEvaluator(device)
    
    # Create dataloader
    test_dataloader = make_dataloaders(
        batch_size=batch_size,
        paths=test_paths,
        split='val'
    )
    
    real_features = []
    fake_features = []
    
    print("Calculating FID score...")
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            # Get real images
            L = data['L'].to(device)
            ab = data['ab'].to(device)
            
            # Generate fake images
            fake_ab = model.net_G(L)
            
            # Convert to RGB
            real_rgb = fid_evaluator.lab_to_rgb(L, ab)
            fake_rgb = fid_evaluator.lab_to_rgb(L, fake_ab)
            
            # Convert to tensor and normalize for Inception
            real_tensor = torch.stack([fid_evaluator.transform(Image.fromarray((img * 255).astype(np.uint8))) 
                                     for img in real_rgb])
            fake_tensor = torch.stack([fid_evaluator.transform(Image.fromarray((img * 255).astype(np.uint8))) 
                                     for img in fake_rgb])
            
            # Extract features
            real_features.append(fid_evaluator.get_features(real_tensor))
            fake_features.append(fid_evaluator.get_features(fake_tensor))
    
    # Concatenate all features
    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)
    
    # Calculate statistics
    mu_real, sigma_real = fid_evaluator.calculate_statistics(real_features)
    mu_fake, sigma_fake = fid_evaluator.calculate_statistics(fake_features)
    
    # Calculate FID score
    fid_score = fid_evaluator.calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    
    return fid_score

if __name__ == "__main__":
    # Example usage
    model_path = "path/to/your/model.pth"
    test_paths = ["path/to/test/images/*.jpg"]  # List of paths to test images
    
    fid_score = evaluate_fid(model_path, test_paths)
    print(f"FID Score: {fid_score:.2f}")
