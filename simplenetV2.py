import torch
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
from collections import OrderedDict
import torchvision.models as models


from anoDataset import anoDataset


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


# class ResNet18(nn.Module):
#     def __init__(self, pretrained_path=None):
#         super(ResNet18, self).__init__()

#         # Load a pretrained resnet18 model
#         resnet = models.resnet18(pretrained=False)

#         if pretrained_path:
#             state_dict = torch.load(pretrained_path, map_location=device)
#             resnet.load_state_dict(state_dict)
            
#         # Use the early layers
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool

#         self.layer1 = resnet.layer1  # Output channels: 64
#         self.layer2 = resnet.layer2  # Output channels: 128
#         self.layer3 = resnet.layer3  # Output channels: 256
#         self.layer4 = resnet.layer4  # Output channels: 512

#     def forward(self, x):
#         # Initial layers
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         # Feature layers
#         f1 = self.layer1(x)  # (B, 64, H/4, W/4)
#         f2 = self.layer2(f1)  # (B, 128, H/8, W/8)
#         f3 = self.layer3(f2)  # (B, 256, H/16, W/16)
#         f4 = self.layer4(f3)  # (B, 512, H/32, W/32)
#         return  f1, f2, f3, f4


class PDN_S(torch.nn.Module):

    def __init__(self, last_kernel_size=768, with_bn=False) -> None:
        super().__init__()
        # Layer Name Stride Kernel Size Number of Kernels Padding Activation
        # Conv-1 1×1 4×4 128 3 ReLU
        # AvgPool-1 2×2 2×2 128 1 -
        # Conv-2 1×1 4×4 256 3 ReLU
        # AvgPool-2 2×2 2×2 256 1 -
        # Conv-3 1×1 3×3 256 1 ReLU
        # Conv-4 1×1 4×4 384 0 -
        self.with_bn = with_bn
        self.conv1 = torch.nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(256, last_kernel_size, kernel_size=4, stride=1, padding=0)
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        if self.with_bn:
            self.bn1 = torch.nn.BatchNorm2d(128)
            self.bn2 = torch.nn.BatchNorm2d(256)
            self.bn3 = torch.nn.BatchNorm2d(256)
            self.bn4 = torch.nn.BatchNorm2d(last_kernel_size)


    def forward(self, x):
        f1 = F.relu(self.bn1(self.conv1(x))) if self.with_bn else F.relu(self.conv1(x))
        f1 = self.avgpool1(f1)

        f2 = F.relu(self.bn2(self.conv2(f1))) if self.with_bn else F.relu(self.conv2(f1))
        f2 = self.avgpool2(f2)

        f3 = F.relu(self.bn3(self.conv3(f2))) if self.with_bn else F.relu(self.conv3(f2))
        f4 = self.conv4(f3)
        f4 = self.bn4(f4) if self.with_bn else f4

        return f4





# class Generator(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, in_ch * 2, kernel_size=1),
#             nn.PReLU(inplace=True),
#             nn.Conv2d(in_ch * 2, in_ch * 2, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_ch * 2, out_ch, kernel_size=1)
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_normal_(m.weight)

#     def forward(self, x):
#         return self.net(x)

class Generator(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True),
            # torch.nn.PReLU(),
            # torch.nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=True)
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = self.net(x)
        return x


# class Discriminator(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(Discriminator, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, in_ch*2, kernel_size=1),
#             #nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
            
#             nn.Conv2d(in_ch*2, in_ch*2, kernel_size=1),
#             #nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(in_ch*2, out_ch, kernel_size=1),  # final output: [B, 1, H, W]
#             #nn.Tanh()  # output range: [-1, 1]
#         )

#         # Xavier Initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_normal_(m.weight)

#     def forward(self, x):
#         return self.net(x)



class Discriminator(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Discriminator, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Conv2d(in_ch, int(out_ch), kernel_size=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(int(out_ch), 1, kernel_size=1, bias=False),
            # nn.Sigmoid()
        )

        # Xavier Initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        # out = self.net(x)
        # out_cpu = out.detach().cpu().numpy()  # Move to CPU and numpy safely
        # print("MinDiscriminator:", np.min(out_cpu))
        # print("MaxDiscriminator:", np.max(out_cpu))
        return self.net(x)
    

class ComputeLoss:
    def __init__(self, device, noise_std=0.015, penalize_generator=1.1):
        super().__init__()
        self.device = device
        self.noise_std = noise_std
        self.penalize_generator= penalize_generator

    def __call__(self, features, model_g, model_d):
        true_feats = model_g(features)

        # noise_drop = torch.zeros(features.shape, dtype=torch.float)
        # for channel in range(features.shape[1]):
        #     noise_drop[:, channel, :, :] = np.random.uniform(-self.noise_std, self.noise_std)
        noise_drop = torch.randn_like(true_feats) * self.noise_std
        
        #recon_loss = F.mse_loss(true_feats, features)
        
        #gaussian_random = torch.zeros(features.shape, dtype=torch.int)
        #for channel in range(features.shape[1]):
        #    gaussian_random[:, channel, :, :] = np.random.randint(0, 2)

        noise_drop = noise_drop.to(self.device)


        fake_feats = (noise_drop + true_feats)

        true_scores = model_d(true_feats)
        fake_scores = model_d(fake_feats)

        th = 0.5
        p_true = (true_scores.detach() >= th).sum() / len(true_scores)
        p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
        # true_loss = torch.clip(-true_scores + th, min=0)
        # fake_loss = torch.clip(fake_scores + th, min=0)
        true_loss = F.binary_cross_entropy_with_logits(true_scores, torch.ones_like(true_scores))
        fake_loss = F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))

        #true_loss = F.relu(th - true_scores)
        #fake_loss = F.relu(fake_scores + th)

        # print("True scores:", true_scores.detach().cpu().numpy()[:5])
        # print("Fake scores:", fake_scores.detach().cpu().numpy()[:5])
        # print("True Score Stats:", true_scores.min().item(), true_scores.max().item(), true_scores.mean().item())
        # print("Fake Score Stats:", fake_scores.min().item(), fake_scores.max().item(), fake_scores.mean().item())

        loss = true_loss.mean() + self.penalize_generator* fake_loss.mean()

        #lamb=0.1
        #total_loss= loss + lamb*recon_loss
        # print(">>> True feature stats:")
        # print("Mean:", true_feats.mean().item(), "Std:", true_feats.std().item())

        # print(">>> Noise stats:")
        # print("Mean:", noise_drop.mean().item(), "Std:", noise_drop.std().item())

        # print(">>> noisy feature stats:")
        # print("Mean:", fake_feats.mean().item(), "Std:", fake_feats.std().item())

        return loss, true_loss.mean(), fake_loss.mean(), true_scores.detach(), fake_scores.detach()





class ComputeLoss2:
    def __init__(self, device):
        super().__init__()
        self.device = device

    def __call__(self, features, model_g, model_d):
        true_feats = model_g(features)

        indices = torch.randint(0, 1, torch.Size([true_feats.shape[0]]))
        one_hot = F.one_hot(indices, num_classes=1)  # (N, K)
        noise = torch.stack([torch.normal(0, 0.015, true_feats.shape)], dim=1)  # (N, K, C)
        noise = (noise.to(self.device) * one_hot.to(self.device).unsqueeze(-1)).sum(1)
        fake_feats = true_feats + noise

        scores = model_d(torch.cat([true_feats, fake_feats]))
        true_scores = scores[:len(true_feats)]
        fake_scores = scores[len(fake_feats):]

        th = 0.5
        p_true = (true_scores.detach() >= th).sum() / len(true_scores)
        p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
        true_loss = F.relu(th - true_scores)
        fake_loss = F.relu(fake_scores + th)

        loss = true_loss.mean() + fake_loss.mean() 

        return loss, true_loss.mean(), fake_loss.mean(), true_scores.detach(), fake_scores.detach()



def save_score_histograms(epoch, true_scores, fake_scores, save_dir="score_histograms"):
    os.makedirs(run_path, exist_ok=True)

    true_scores = true_scores.flatten()
    fake_scores = fake_scores.flatten()

    plt.figure(figsize=(8, 5))
    plt.hist(true_scores, bins=50, alpha=0.6, label='True Scores', color='green', density=True)
    plt.hist(fake_scores, bins=50, alpha=0.6, label='Fake Scores', color='red', density=True)
    plt.title(f"Score Distribution - Epoch {epoch}")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(run_path, f"epoch_{epoch:03d}.png")
    plt.savefig(save_path)
    plt.close()

    return save_path



############################################################################################################




#hyper parameter
feature_size = 384 
image_width = 448
image_height = 448

epochs = 1000
batch_size = 15
learningRate_d = 2E-4
learningRate_g = learningRate_d / 2
weightDecay = 1E-5
#hyper parameter



#Dataset Loader
transform = transforms.Compose([
    transforms.Resize(size=(image_height, image_width)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
trianDataset = anoDataset(path="dataset/PCB_Boxes(New)/train/good", transform=transform)
trainLoader = torch.utils.data.DataLoader(trianDataset, batch_size=batch_size, shuffle=True, num_workers=0)
#
evalDataset = anoDataset(path="dataset/PCB_Boxes(New)/Test/anomaly", transform=transform)
evalLoader = torch.utils.data.DataLoader(evalDataset, batch_size=1, shuffle=True, num_workers=0)
# Setup save path
base_dir = "SimpleNetV2_checkpoints"
os.makedirs(base_dir, exist_ok=True)

# Auto-increment run ID
existing_runs = sorted([d for d in os.listdir(base_dir) if d.startswith("run_")])
run_id = len(existing_runs) + 1
run_path = os.path.join(base_dir, f"run_{run_id}")
os.makedirs(run_path, exist_ok=True)




# Create CSV logger
csv_path = os.path.join(run_path, "loss_log.csv")
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Total Loss", "True Loss", "Fake Loss"])


# Loss trackers
loss_history = []
true_loss_history = []
fake_loss_history = []

checkpoint_path= "backbones/resnet18-f37072fd.pth"
#model = ResNet18(pretrained_path=checkpoint_path)
model = PDN_S(with_bn=False, last_kernel_size=feature_size)

model = model.to(device)




# Load raw state dict
#raw_state_dict = torch.load(checkpoint_path, map_location=device)

# # Strip the "pdn." prefix from all keys
# new_state_dict = OrderedDict()
# for k, v in raw_state_dict.items():
#     new_key = k.replace("pdn.", "")  # Remove "pdn." from keys
#     new_state_dict[new_key] = v

# model.load_state_dict(new_state_dict)
# model.eval()


for param in model.parameters():
    param.requires_grad = False

#multi_input_channels = 128 + 256 + 256 + feature_size  # f1 + f2 + f3 + f4
#multi_input_channels = 64 + 128 + 256 + feature_size  # 960 channels
f4_channel_size=384
model_g = Generator(in_ch=f4_channel_size, out_ch=feature_size)
model_g = model_g.to(device)
model_g.train()



model_d = Discriminator( in_ch=feature_size, out_ch=feature_size)
model_d = model_d.to(device)
model_d.train()


criterion  = ComputeLoss(device=device)


totalBatchSize = len(trainLoader)
print('total batch size = ', totalBatchSize)

optimizer_d = torch.optim.Adam(model_d.parameters(), learningRate_d, weight_decay=weightDecay)
optimizer_g = torch.optim.AdamW(model_g.parameters(), learningRate_g)

for epoch in range(1, epochs):
    

    model_d.train()
    model_g.train()

    totalLoss = 0
    totalFakeLoss = 0
    totalTrueLoss = 0
    



    currentBatch = 0
    for input in trainLoader:

        optimizer_d.zero_grad()
        optimizer_g.zero_grad()
        gpu_input = input.to(device)

        #f1, f2, f3, f4 = model(gpu_input)
        f4= model(gpu_input)
        # with torch.no_grad():
        #     # Resize f1–f3 to match f4 spatial size
        #     f1 = F.interpolate(f1, size=f4.shape[2:], mode='nearest')
        #     f2 = F.interpolate(f2, size=f4.shape[2:], mode='nearest')
        #     f3 = F.interpolate(f3, size=f4.shape[2:], mode='nearest')

        # multi_feat = torch.cat([f1, f2, f3, f4], dim=1)

        loss, trueLoss, fakeLoss, truescore, fakescore = criterion(f4, model_g, model_d)

        loss.backward()

        optimizer_g.step()
        optimizer_d.step()


        totalLoss += (loss.item() / totalBatchSize)
        totalFakeLoss += (fakeLoss.item() / totalBatchSize)
        totalTrueLoss += (trueLoss.item() / totalBatchSize)

        currentBatch += 1
        #print('current batch status = ', currentBatch, ' / ', totalBatchSize)
        #print('batch loss=', loss.item(), ', fake loss=', fakeLoss.item(), ', true loss=', trueLoss.item())
        
        # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'model_g': model_g.state_dict(),
        'model_d': model_d.state_dict(),
        'optimizer_d': optimizer_d.state_dict(),
        'optimizer_g': optimizer_g.state_dict()
    }
    torch.save(checkpoint, os.path.join(run_path, f"checkpoint_epoch_{epoch}.pth"))

    # Track and plot losses
    loss_history.append(totalLoss)
    true_loss_history.append(totalTrueLoss)
    fake_loss_history.append(totalFakeLoss)

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Total Loss', linewidth=2)
    plt.plot(true_loss_history, label='True Loss', linestyle='--')
    plt.plot(fake_loss_history, label='Fake Loss', linestyle='-.')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Trend per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_path, 'loss_curve.png'))
    plt.close()

    # Append to CSV
    with open(csv_path, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, totalLoss, totalTrueLoss, totalFakeLoss])

    print('****** epoch = ', epoch)
    print('****** avg totall loss=', totalLoss, ', avg fake loss=', totalFakeLoss, ', avg true loss=', totalTrueLoss)


    save_score_histograms(epoch, truescore.cpu().numpy(), fakescore.cpu().numpy())
    normal_scores = []
    model.eval()
    with torch.no_grad():
        for input in trainLoader:  # or a separate good-only val set
            gpu_input = input.to(device)
            #f1, f2, f3, f4 = model(gpu_input)
            f4 = model(gpu_input)

            # # Resize to match f4 size
            # f1 = F.interpolate(f1, size=f4.shape[2:], mode='nearest')
            # f2 = F.interpolate(f2, size=f4.shape[2:], mode='nearest')
            # f3 = F.interpolate(f3, size=f4.shape[2:], mode='nearest')

            # # Fuse features
            # multi_feat = torch.cat([f1, f2, f3, f4], dim=1)
            
            score = -model_d(model_g(f4))
            image_score = score.view(score.size(0), -1).max(dim=1)[0]
            normal_scores.append(image_score.cpu().numpy())

    normal_scores = np.concatenate(normal_scores)
    threshold = np.percentile(normal_scores, 93)  # 95% of normal images fall below this
    print(f"Suggested threshold: {threshold:.4f}")
    #threshold=2.5
    with torch.no_grad():
        model.eval()
        model_g.eval()
        model_d.eval()
        for input in evalLoader:


            gpu_input = input.to(device)
            #f1, f2, f3, f4 = model(gpu_input)
            f4= model(gpu_input)

            # # Resize to match f4 size
            # f1 = F.interpolate(f1, size=f4.shape[2:], mode='nearest')
            # f2 = F.interpolate(f2, size=f4.shape[2:], mode='nearest')
            # f3 = F.interpolate(f3, size=f4.shape[2:], mode='nearest')

            # # Fuse features
            # multi_feat = torch.cat([f1, f2, f3, f4], dim=1)
            
            #output = model_d(features)

            #true_feats = model_g(features)
            
            # Compute scores
            scores = -model_d(model_g(f4)) 

            # Compute image-level anomaly score (e.g., max)
            image_anomaly_score = scores.view(scores.size(0), -1).max(dim=1)[0]

            print(f"Anomaly Score: {image_anomaly_score.item():.4f}")
            if image_anomaly_score.item() > threshold:
                print(f"Anomaly Score: {image_anomaly_score.item():.4f}:Anomalous Image")
            else:
                print(f"Anomaly Score: {image_anomaly_score.item():.4f}:Normal Image")



            input_image = input[0].permute(1, 2, 0).detach().cpu().numpy()
            input_image = (input_image * 255).astype(np.uint8)

            #output_image = scores[0][0].detach().cpu().numpy()
            upsampled_map = F.interpolate(scores, size=(image_height, image_width), mode='bilinear', align_corners=False)
            output_image = upsampled_map[0][0].detach().cpu().numpy()


            # Pixel-wise thresholding for visualization

            #anomaly_map = upsampled_map[0][0].detach().cpu()
            clip_max = np.percentile(output_image, 99.5)
            anomaly_map = np.clip(output_image, 0, clip_max)
            anomaly_map = (anomaly_map / (clip_max + 1e-8)) * 255
            anomaly_map = anomaly_map.astype(np.uint8)
            
            
            pixel_threshold_value = 150  # You can tune this
            _, binary_map = cv2.threshold(anomaly_map, pixel_threshold_value, 255, cv2.THRESH_BINARY)


            #clip_max = np.percentile(output_image, 99.5)
            #output_image = np.clip(output_image, 0,clip_max)
            min_value = np.min(output_image)
            max_value = np.max(output_image)
            
            output_image = (output_image - min_value) / (max_value - min_value)


            #output_image = (output_image / (clip_max + 1e-8))
            output_image = (output_image * 255).astype(np.uint8)  # Convert to [0, 255] uint8
            # output_image = cv2.resize(output_image, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
            # output_image = cv2.equalizeHist(output_image)
            #output_image = cv2.applyColorMap(output_image, cv2.COLORMAP_JET)




            # Show third window for pixel-wise thresholded visualization
            cv2.namedWindow("pixelwise_thresholded", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("pixelwise_thresholded", image_width, image_height)
            cv2.imshow("pixelwise_thresholded", binary_map)



            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('output', image_width, image_height)
            cv2.imshow('output', output_image)


            cv2.namedWindow("input", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('input', image_width, image_height)
            cv2.imshow('input', input_image)
            cv2.waitKey(100)
