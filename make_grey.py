from PIL import Image
import os

def convert_to_grey(image_path):
    """Convert an image to grayscale and save it"""
    # Create output directory if it doesn't exist
    grey_dir = os.path.join('imgs', 'grey')
    os.makedirs(grey_dir, exist_ok=True)
    
    # Get filename and create output path
    filename = os.path.basename(image_path)
    output_path = os.path.join(grey_dir, filename)
    
    # Open and convert image to grayscale
    img = Image.open(image_path).convert('L')
    
    # Save the grayscale image
    img.save(output_path)
    print(f"Saved grayscale image to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Example usage
    input_image = "imgs/true/001.jpg"  # Change this to your input image path
    convert_to_grey(input_image)