import torch
import torch.nn as nn
import time
from torchvision import transforms
from PIL import Image
import ai3  # Assuming ai3 is the library where your custom conv2d is implemented


# Define a simple model with a single convolutional layer
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)


def load_image(image_path, transform):
    """
    Load an image from the given path and apply the specified transformations.
    """
    image = Image.open(image_path).convert("RGB")  # Ensure it's a 3-channel image
    return transform(image)


if __name__ == "__main__":
    # Define the image preprocessing transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to Tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize
        ]
    )

    # Initialize the simple CNN
    model = SimpleCNN(
        in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
    )

    # Swap the convolution operation with your custom implementation
    ai3.swap_conv2d(model)  # Assuming no 'algorithm' argument is needed

    # List of image paths
    image_paths = [
        "./virat.png",
        "./dhoni.png",
        "./ak.png",
        "./vj.png",
        "./messi.png",
        "./ronaldo.png",
    ]

    # Ensure the images exist and are valid
    for image_path in image_paths:
        try:
            Image.open(image_path).verify()
        except Exception as e:
            print(f"Invalid image file {image_path}: {e}")
            image_paths.remove(image_path)

    total_time = 0

    for image_path in image_paths:
        try:
            # Load and preprocess the image
            input_image = load_image(image_path, transform)
            input_tensor = input_image.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        # Measure the time taken for the convolution
        start_time = time.time()
        output = model(input_tensor)
        end_time = time.time()

        total_time += end_time - start_time

    avg_time = total_time / len(image_paths)
    print(f"Average matrix multiplication time for 6 images: {avg_time} seconds")
