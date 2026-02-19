import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from model import simplenet

# Paths
image_path = "Test/Good/000.png"
checkpoint_path = "SimpleNetV2_checkpoints/run_3/Checkpoints/checkpoint_epoch_99.pth"
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import simplenet

# Parameters
image_height = 320
image_width = 320
patch_grid = image_height // 8  # 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load image
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

image_np = cv2.resize(image_np, (image_width, image_height))
image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# Preprocess
transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0).to(device)

# Load model
model_backbone = simplenet.resnet18().to(device)
model_g = simplenet.Generator(image_height, image_width, in_ch=128+256, out_ch=128+256).to(device)
model_d = simplenet.Discriminator(in_ch=128+256, out_ch=128+256).to(device)

ckpt = torch.load(checkpoint_path, map_location=device)
model_g.load_state_dict(ckpt['model_g'])
model_d.load_state_dict(ckpt['model_d'])
model_backbone.eval()
model_g.eval()
model_d.eval()

# Forward pass
with torch.no_grad():
    features = model_backbone(input_tensor)
    adapted_features = model_g(features)
    scores = -model_d(adapted_features)

# Convert scores
scores = scores.view(patch_grid, patch_grid).cpu().numpy()

# Normalize
scores_normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)

# Resize to full image
heatmap = cv2.resize(scores_normalized, (image_width, image_height))
heatmap_uint8 = np.uint8(255 * heatmap)

# Apply colormap
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

# Overlay on image
overlay = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)

# Save or display
cv2.imwrite("anomaly_heatmap.png", heatmap_color)
cv2.imwrite("anomaly_overlay.png", overlay)

print("✅ Saved: 'anomaly_heatmap.png' and 'anomaly_overlay.png'")
