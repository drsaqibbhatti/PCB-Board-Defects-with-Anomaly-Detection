import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from collections import OrderedDict
from anoDataset import anoDataset
from simplenet.simplenet import PDN_S, Generator, Discriminator

# ------------------- Config -------------------

checkpoint_path = 'SimpleNetV2_checkpoints/run_8/checkpoint_epoch_175.pth'
test_data_path = 'dataset/PCB_Boxes(New)/Test/anomaly'
output_dir = 'SimpleNetV2_checkpoints/run_8/test_outputs'
feature_size = 384
image_height = 256
image_width = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(output_dir, exist_ok=True)

# ------------------- Load Model -------------------

model = PDN_S(with_bn=False, last_kernel_size=feature_size).to(device)
model_g = Generator(in_ch=128 + 256 + 256 + feature_size, out_ch=feature_size).to(device)
model_d = Discriminator(in_ch=feature_size, out_ch=1).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model_g.load_state_dict(checkpoint['model_g'])
model_d.load_state_dict(checkpoint['model_d'])

model.eval()
model_g.eval()
model_d.eval()

for param in model.parameters():
    param.requires_grad = False

# ------------------- Dataset -------------------

transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor()
])

evalDataset = anoDataset(path=test_data_path, transform=transform)
evalLoader = torch.utils.data.DataLoader(evalDataset, batch_size=1, shuffle=False, num_workers=0)

# ------------------- Inference Loop -------------------

with torch.no_grad():
    for idx, input in enumerate(evalLoader):
        input = input.to(device)

        # --- Extract multi-level features ---
        f1, f2, f3, f4 = model(input)
        f1 = F.interpolate(f1, size=f4.shape[2:], mode='bilinear', align_corners=False)
        f2 = F.interpolate(f2, size=f4.shape[2:], mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=f4.shape[2:], mode='bilinear', align_corners=False)
        multi_feat = torch.cat([f1, f2, f3, f4], dim=1)

        # --- Anomaly map computation ---
        gen_feats = model_g(multi_feat)
        scores = -model_d(gen_feats)

        upsampled_map = F.interpolate(scores, size=(image_height, image_width), mode='bilinear', align_corners=False)
        anomaly_map = upsampled_map[0][0].detach().cpu().numpy()

        # --- Normalize anomaly map ---
        clip_max = np.percentile(anomaly_map, 99.5)
        anomaly_map = np.clip(anomaly_map, 0, clip_max)
        anomaly_map = anomaly_map / (clip_max + 1e-8)
        anomaly_map = (anomaly_map * 255).astype(np.uint8)

        # --- Save output ---
        save_path = os.path.join(output_dir, f"anomaly_output_{idx:04d}.png")
        cv2.imwrite(save_path, anomaly_map)

        print(f"Saved: {save_path}")
