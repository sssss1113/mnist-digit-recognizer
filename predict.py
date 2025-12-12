import torch
from model import Net
from torchvision import transforms
from PIL import Image, ImageOps
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

net = Net().to(DEVICE)
net.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=DEVICE))
net.eval()

img_path = sys.argv[1] if len(sys.argv) > 1 else "images/one.png"

# 1. Load grayscale
img = Image.open(img_path).convert("L")

# 2. Invert to match MNIST (white digit on black)
img = ImageOps.invert(img)

# 3. Resize WITH aspect ratio to max 20×20 (MNIST style)
img.thumbnail((20, 20), Image.LANCZOS)

# 4. Create 28×28 black canvas and paste centered
canvas = Image.new("L", (28, 28), 0)
x = (28 - img.width) // 2
y = (28 - img.height) // 2
canvas.paste(img, (x, y))

# 5. Normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img_tensor = transform(canvas).unsqueeze(0).to(DEVICE)

output = net(img_tensor)
pred = torch.argmax(output, 1).item()

print(f"Predicted digit for {img_path}: {pred}")
