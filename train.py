import os
import random
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import date
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from PIL import Image
from model import simplenet
from utils.util import *
from utils.dataloader import SimpleNetDataset



def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed and processes
    setup_seed()
    setup_multi_processes()
    imageWidth=512
    imageHeight=512
    epochs=50
    # Load backbone (pretrained)
    model_backbone = simplenet.resnet18() 
    state_dict = torch.load('/backbones/resnet18-5c106cde.pth')

    # Filter to avoid loading unmatched keys if needed
    model_dict = model_backbone.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(state_dict)
    model_backbone.load_state_dict(model_dict)
    for p in model_backbone.parameters():
        p.requires_grad_(False)
    model_backbone.to(device)

    # Initialize Generator and Discriminator
    # filters = 128 * model_backbone.fn.expansion + 256 * model_backbone.fn.expansion
    filters = 128 * 1 + 256 * 1  # Since torchvision ResNet uses `BasicBlock` (expansion = 1)

    model_d = simplenet.Discriminator(filters, filters).to(device)
    model_g = simplenet.Generator(image_height=imageHeight,image_width=imageWidth, in_ch=filters, out_ch=filters).to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model_backbone = DataParallel(model_backbone)
        model_g = DataParallel(model_g)
        model_d = DataParallel(model_d)

    # Optimizers
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=2e-4, weight_decay=1e-5)
    optimizer_g = torch.optim.AdamW(model_g.parameters(), lr=1e-4)



    transform = transforms.Compose([
        transforms.Resize((imageHeight, imageWidth), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
    
    dataPathTrain= "\dataset\\PCB_Boxes(New)\\train\\good"
    dataPathTest= "\dataset\\PCB_Boxes(New)\\test\\anomaly"
    train_dataset = SimpleNetDataset(dataPathTrain, transform)
    test_dataset = SimpleNetDataset(dataPathTest, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=2, pin_memory=True)
    # Criterion
    criterion = ComputeLoss(device)

    # Run directory setup
    base_dir = 'trained_models'
    run_name = 'SimpleNetTrainerV1_test'
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    existing_runs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
    run_id = len(existing_runs) + 1
    run_path = os.path.join(run_dir, f"run_{run_id}")
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(os.path.join(run_path, 'Checkpoints'), exist_ok=True)

    # CSV logger
    log_file = os.path.join(run_path, 'training_log.csv')
    metric_writer = pd.DataFrame(columns=['Epoch', 'Train Loss', 'F1', 'ACC', 'AUC', 'ROC_AUC'])

    best_score = 0
    for epoch in range(epochs):
        model_backbone.eval()
        model_d.train()
        model_g.train()

        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for samples, _, _ in pbar:
            samples = samples.to(torch.float).to(device)
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()


            with torch.no_grad():
                features = model_backbone(samples)
            
            # print("Feature sizes from backbone:")
            # for i, feat in enumerate(features):
            #     print(f" - Layer {i}: {feat.shape}")
            loss, true_score, fake_score = criterion(features, model_g, model_d)
            loss.backward()
            optimizer_d.step()
            optimizer_g.step()

            train_loss += loss.item() * samples.size(0)
            pbar.set_postfix(loss=loss.item(), true=true_score.item(), fake=fake_score.item())

        train_loss /= len(train_loader.dataset)


        model_backbone_eval = copy.deepcopy(model_backbone.module if isinstance(model_backbone, DataParallel) else model_backbone)
        model_backbone_eval.eval()
        model_g.eval()
        model_d.eval()

        scores = []
        labels = []

        for samples, targets, filenames in tqdm(test_loader, desc=f"Testing Epoch {epoch}"):
            samples = samples.to(torch.float).to(device)
            labels.extend(targets.cpu().numpy().tolist())
            shape = samples.shape[0]

            with torch.no_grad():
                features = model_backbone_eval(samples)
                features = model_g(features)

                patch_scores = image_scores = -model_d(features)
                patch_scores = patch_scores.cpu().numpy()
                image_scores = image_scores.cpu().numpy()

                # Reshape to [B, -1, ...] if needed
                image_scores = image_scores.reshape(shape, -1, *image_scores.shape[1:])
                image_scores = image_scores.reshape(*image_scores.shape[:2], -1)

                # Pool over all dimensions until 1D per sample remains
                was_numpy = False
                if isinstance(image_scores, np.ndarray):
                    was_numpy = True
                    image_scores = torch.from_numpy(image_scores)
                while image_scores.ndim > 2:
                    image_scores = torch.max(image_scores, dim=-1).values
                if image_scores.ndim == 2:
                    image_scores = torch.max(image_scores, dim=1).values
                if was_numpy:
                    image_scores = image_scores.numpy()

                scores.extend(list(image_scores))

        scores = np.squeeze(np.array(scores))
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
        
        # ✅ Add this to inspect what the model is outputting
        print("Sample outputs (normalized anomaly scores):", scores)
        print("Sample targets (ground truth labels):", labels)
        ########## Dynamic threshold selection based on best F1 ###############
        best_f1 = 0
        best_threshold = 0.5
        for t in np.linspace(0, 1, 100):
            preds = (scores > t).astype(int)
            f1 = metrics.f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        # Final predictions with best threshold
        final_preds = (scores > best_threshold).astype(int)

        # Compute final metrics
        roc_auc = metrics.roc_auc_score(labels, scores)
        precision, recall, _ = metrics.precision_recall_curve(labels, scores)
        auc = metrics.auc(recall, precision)
        acc = metrics.accuracy_score(labels, final_preds)
        print(f"🧪 Optimal Threshold: {best_threshold:.3f}")
        ###################################################################  
              
        #roc_auc, auc, f1, acc = compute_metrics(scores, labels) FIXED THRESHOLD
        
        print(f"Test Results - F1: {f1:.3f}, ACC: {acc:.3f}, ROC_AUC: {roc_auc:.3f}, AUC: {auc:.3f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_g': model_g.state_dict(),
            'model_d': model_d.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
        }
        torch.save(checkpoint, os.path.join(run_path, 'Checkpoints', f'checkpoint_epoch_{epoch}.pth'))

        # Save best model
        score = fitness(np.array([roc_auc, auc, f1, acc]))
        if score > best_score:
            best_score = score
            torch.save(checkpoint, os.path.join(run_path, 'best_model.pth'))
            print(f"[Epoch {epoch}] Best model saved with fitness score: {best_score:.4f}")

        # Log metrics
        metric_writer.loc[len(metric_writer)] = [epoch, train_loss, f1, acc, auc, roc_auc]
        metric_writer.to_csv(log_file, index=False)

    print("Training finished. Best fitness:", best_score)








if __name__ == '__main__':
    train()
    
    run_base_path = '/trained_models/SimpleNetTrainerV1'
    latest_run_path = get_latest_run_path(run_base_path)
    if latest_run_path:
        plot_and_save_training_logs(os.path.join(latest_run_path, 'training_log.csv'))
    else:
        print("No run folder found to plot logs.")