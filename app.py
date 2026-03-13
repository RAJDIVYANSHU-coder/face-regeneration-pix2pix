import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os 

# -----------------------------
# DEVICE (GPU if available)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# MODEL ARCHITECTURE
# -----------------------------

def down_block(in_c, out_c, normalize=True):
    layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)

def up_block(in_c, out_c):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU()
    )

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = down_block(3, 64, normalize=False)
        self.d2 = down_block(64, 128)
        self.d3 = down_block(128, 256)
        self.d4 = down_block(256, 512)

        self.u1 = up_block(512, 256)
        self.u2 = up_block(512, 128)
        self.u3 = up_block(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):

        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        u1 = self.u1(d4)
        u1 = torch.cat([u1, d3], 1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, d2], 1)

        u3 = self.u3(u2)
        u3 = torch.cat([u3, d1], 1)

        return self.final(u3)

# -----------------------------
# LOAD MODEL
# -----------------------------

model = Generator().to(device)

model_path = "pix2pix_epoch_45.pth" #Change the file as per required *.pth file.

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully")
else:
    print("Model file not found. Please train the model using pix2pix.ipynb")
model.eval()

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# -----------------------------
# TENSOR → IMAGE
# -----------------------------

def tensor_to_image(tensor):

    tensor = tensor.squeeze().detach().cpu()
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.numpy()

    tensor = np.transpose(tensor,(1,2,0))
    tensor = (tensor * 255).astype(np.uint8)

    return tensor

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------

def regenerate_face(image):

    image = np.array(image)

    # get width of combined image
    h, w, _ = image.shape

    # crop LEFT HALF (masked face)
    masked_face = image[:, :w//2]

    masked_face = Image.fromarray(masked_face)

    masked_face = transform(masked_face).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(masked_face)

    return tensor_to_image(output)

# -----------------------------
# GRADIO UI
# -----------------------------

interface = gr.Interface(
    fn=regenerate_face,
    inputs=gr.Image(type="pil", label="Upload Combined Image [Masked | Real]"),
    outputs=gr.Image(label="Generated Face"),
    title="Face Regeneration using Pix2Pix",
    description="Upload an image from the dataset (masked | real). The system automatically extracts the masked face and reconstructs the full face."
)

interface.launch()