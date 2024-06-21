import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
from flask import Flask, send_file
from io import BytesIO

# Use the 'Agg' backend for Matplotlib
matplotlib.use('Agg')

# Define the Generator and Discriminator classes
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# Initialize the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
z_dim = 64
image_dim = 28 * 28 * 1

# Initialize models
generator = Generator(z_dim, image_dim).to(device)
discriminator = Discriminator(image_dim).to(device)

# Load pre-trained weights
generator.load_state_dict(torch.load('generator_weights_50.pth', map_location=device))
discriminator.load_state_dict(torch.load('discriminator_weights_50.pth', map_location=device))

app = Flask(__name__)

@app.route('/app', methods=['GET'])
def generate_image():
    # Generate a new image from the GAN
    noise = torch.randn(1, z_dim).to(device)
    fake_image = generator(noise).view(28, 28).cpu().detach().numpy()

    # Plot the image and save to a BytesIO object
    fig, ax = plt.subplots()
    ax.imshow(fake_image, cmap='gray')
    ax.axis('off')

    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='PNG')
    img_bytes.seek(0)
    plt.close(fig)

    return send_file(img_bytes, mimetype='image/png')

if __name__ == '__main__':
    app.run( host='0.0.0.0', port=5000, debug=True)

