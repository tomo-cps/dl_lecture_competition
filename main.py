import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.datasets import ThingsMEGDataset, PreprocessdThingsMEGDataset
from src.models import BasicConvClassifier, EEGNetClassifier
from src.utils import set_seed

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    # subject_id = args.subject_id  # Assuming subject_id is provided in the config

    # train_set = ThingsMEGDataset("train", args.data_dir, subject_id=subject_id)
    # train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    # val_set = ThingsMEGDataset("val", args.data_dir, subject_id=subject_id)
    # val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    # test_set = ThingsMEGDataset("test", args.data_dir, subject_id=subject_id)
    # test_loader = torch.utils.data.DataLoader(
    #     test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    # )
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    # model = BasicConvClassifier(
    #     train_set.num_classes, train_set.seq_len, train_set.num_channels
    # ).to(args.device)
    
    model = EEGNetClassifier(
       num_classes=train_set.num_classes, 
       in_channels=train_set.num_channels, 
       input_window_samples=train_set.seq_len
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    early_stopping_patience = 20
    early_stopping_counter = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        mean_train_loss = np.mean(train_loss)
        mean_train_acc = np.mean(train_acc)
        mean_val_loss = np.mean(val_loss)
        mean_val_acc = np.mean(val_acc)

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {mean_train_loss:.3f} | train acc: {mean_train_acc:.3f} | val loss: {mean_val_loss:.3f} | val acc: {mean_val_acc:.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": mean_train_loss, "train_acc": mean_train_acc, "val_loss": mean_val_loss, "val_acc": mean_val_acc})
        
        # Update the learning rate
        scheduler.step(mean_val_loss)

        if mean_val_acc > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = mean_val_acc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
