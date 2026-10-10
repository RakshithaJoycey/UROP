"""
adapt.py
--------
Stage 2 + 3: Domain adaptation of MST++ to lesion skin, then melanoma classification.

Pipeline
--------
Phase A – Domain Adaptation (fine-tune MST++ on synthetic lesion HSI)
  • Loads the best MST++ checkpoint from train.py
  • Generates synthetic melanoma / benign lesion HSI patches from face data
  • Fine-tunes with BioOpticalLoss, using a lower LR and partial freezing
    (encoder frozen for first N epochs, then unfrozen)
  • Saves adapted model checkpoint

Phase B – Classifier Training
  • Freezes the adapted MST++ encoder
  • Trains MelanomaClassifier on the reconstructed HSI features
  • Evaluates: Accuracy, AUC-ROC, Sensitivity, Specificity

Usage:
    python src/adapt.py --zip ../face_ms.zip

Optional (if you have ISIC images):
    python src/adapt.py --zip ../face_ms.zip --isic_dir ../data/isic
"""

import os
import sys
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))

from dataset        import build_datasets
from mst_plus_plus  import MSTpp, count_parameters as count_params
from bio_loss       import BioOpticalLoss
from lesion_dataset import build_lesion_patches, LesionDataset, load_isic_images
from classifier     import MelanomaClassifier, ReconstructAndClassify


# ── args ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Domain adaptation + melanoma classification")
    p.add_argument("--zip",           default="../face_ms.zip")
    p.add_argument("--isic_dir",      default=None,            help="Optional ISIC data dir")
    p.add_argument("--checkpoints",   default="../checkpoints")
    p.add_argument("--outputs",       default="../outputs")
    p.add_argument("--patch_size",    type=int,   default=64)
    p.add_argument("--base_ch",       type=int,   default=32)
    p.add_argument("--num_blocks",    type=int,   default=2)
    p.add_argument("--num_heads",     type=int,   default=1)
    # domain adaptation
    p.add_argument("--adapt_epochs",  type=int,   default=30)
    p.add_argument("--adapt_lr",      type=float, default=5e-5)
    p.add_argument("--freeze_epochs", type=int,   default=5,
                   help="Epochs to keep encoder frozen before full fine-tune")
    p.add_argument("--n_melanoma",    type=int,   default=80)
    p.add_argument("--n_benign",      type=int,   default=80)
    # classifier
    p.add_argument("--clf_epochs",    type=int,   default=30)
    p.add_argument("--clf_lr",        type=float, default=1e-3)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


# ── Phase A: domain adaptation ────────────────────────────────────────────────

def adapt_model(args, model, hsi_patches, hsi_stats, device):
    print("\n" + "=" * 60)
    print("Phase A: Domain Adaptation (lesion skin fine-tuning)")
    print("=" * 60)

    # build lesion dataset
    if args.isic_dir:
        rgb_p, hsi_p, labels = load_isic_images(
            args.isic_dir, patch_size=args.patch_size, seed=args.seed
        )
        if rgb_p is None:
            print("  ISIC dir specified but no images found — falling back to synthetic")
            hsi_p, labels = build_lesion_patches(
                hsi_patches, args.n_melanoma, args.n_benign, args.seed
            )
            rgb_p = None
    else:
        hsi_p, labels = build_lesion_patches(
            hsi_patches, args.n_melanoma, args.n_benign, args.seed
        )
        rgb_p = None

    dataset = LesionDataset(hsi_p, labels, hsi_stats, rgb_patches=rgb_p)

    # 80/20 split
    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=8,  shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=8,  shuffle=False, num_workers=0)
    print(f"  Adaptation train: {n_train}  val: {n_val}")

    criterion = BioOpticalLoss(lambda_chrom=0.05, lambda_smooth=0.01, lambda_mono=0.01).to(device)

    # start with encoder frozen
    def freeze_encoder(m, frozen: bool):
        for name, param in m.named_parameters():
            if "stem" in name or "enc" in name:
                param.requires_grad = not frozen

    freeze_encoder(model, frozen=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.adapt_lr, weight_decay=1e-4
    )

    best_val   = float("inf")
    history    = []
    adapt_ckpt = os.path.join(args.checkpoints, "adapted_model.pth")

    print(f"\n{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'Frozen':>7}")
    print("-" * 40)

    for epoch in range(1, args.adapt_epochs + 1):
        # unfreeze encoder after freeze_epochs
        if epoch == args.freeze_epochs + 1:
            freeze_encoder(model, frozen=False)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.adapt_lr * 0.5, weight_decay=1e-4
            )
            print(f"         ↑ Encoder unfrozen at epoch {epoch}")

        frozen_str = "yes" if epoch <= args.freeze_epochs else "no"

        # train
        model.train()
        t_loss = 0.0
        for rgb, hsi, _ in train_loader:   # ignore labels in adaptation
            rgb, hsi = rgb.to(device), hsi.to(device)
            pred = model(rgb)
            loss, _ = criterion(pred, hsi)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * rgb.size(0)
        t_loss /= n_train

        # validate
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for rgb, hsi, _ in val_loader:
                rgb, hsi = rgb.to(device), hsi.to(device)
                pred = model(rgb)
                loss, _ = criterion(pred, hsi)
                v_loss += loss.item() * rgb.size(0)
        v_loss /= n_val

        history.append({"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss})
        print(f"{epoch:>6}  {t_loss:>10.5f}  {v_loss:>10.5f}  {frozen_str:>7}")

        if v_loss < best_val:
            best_val = v_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    best_val,
                "model_type":  "MSTpp_adapted",
            }, adapt_ckpt)
            print(f"         ✓ Adapted model saved (val={best_val:.5f})")

    with open(os.path.join(args.outputs, "adapt_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # reload best adapted weights
    ckpt = torch.load(adapt_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nPhase A done. Best adaptation val loss: {best_val:.5f}")
    return model, dataset, labels


# ── Phase B: classifier training ─────────────────────────────────────────────

def train_classifier(args, model, dataset, device):
    print("\n" + "=" * 60)
    print("Phase B: Melanoma Classifier Training")
    print("=" * 60)

    # freeze MST++ — classifier uses frozen HSI features
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    classifier = MelanomaClassifier(in_bands=31, feature_dim=128, dropout=0.3).to(device)
    print(f"  Classifier parameters: {count_params(classifier):,}")

    # 80/20 split (same indices as before for fairness)
    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.clf_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0
    clf_ckpt     = os.path.join(args.checkpoints, "classifier.pth")
    history      = []

    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Acc':>10}")
    print("-" * 35)

    for epoch in range(1, args.clf_epochs + 1):
        # train
        classifier.train()
        t_loss = 0.0
        for rgb, hsi, labels in train_loader:
            rgb, hsi, labels = rgb.to(device), hsi.to(device), labels.to(device)
            with torch.no_grad():
                pred_hsi = model(rgb)
            logits = classifier(pred_hsi)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * rgb.size(0)
        t_loss /= n_train
        scheduler.step()

        # validate
        classifier.eval()
        correct, total = 0, 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for rgb, hsi, labels in val_loader:
                rgb, hsi, labels = rgb.to(device), hsi.to(device), labels.to(device)
                pred_hsi = model(rgb)
                logits   = classifier(pred_hsi)
                probs    = torch.softmax(logits, dim=1)[:, 1]
                preds    = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_acc = correct / total
        history.append({"epoch": epoch, "train_loss": t_loss, "val_acc": val_acc})
        print(f"{epoch:>6}  {t_loss:>12.5f}  {val_acc:>10.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":            epoch,
                "classifier_state": classifier.state_dict(),
                "val_acc":          best_val_acc,
            }, clf_ckpt)
            print(f"         ✓ Best classifier saved (acc={best_val_acc:.4f})")

    # ── final evaluation ──
    classifier.eval()
    all_preds, all_probs, all_labels_list = [], [], []
    with torch.no_grad():
        for rgb, hsi, labels in val_loader:
            rgb, hsi, labels = rgb.to(device), hsi.to(device), labels.to(device)
            pred_hsi = model(rgb)
            logits   = classifier(pred_hsi)
            probs    = torch.softmax(logits, dim=1)[:, 1]
            preds    = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels_list.extend(labels.cpu().numpy().tolist())

    metrics = _compute_clf_metrics(all_labels_list, all_preds, all_probs)

    print("\n=== Classifier Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}")

    with open(os.path.join(args.outputs, "clf_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    results_path = os.path.join(args.outputs, "classifier_results.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nClassifier results → {results_path}")

    return classifier, metrics


def _compute_clf_metrics(y_true, y_pred, y_prob) -> dict:
    """Accuracy, sensitivity, specificity, AUC (trapezoidal)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy    = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)   # recall / TPR
    specificity = tn / (tn + fp + 1e-8)
    precision   = tp / (tp + fp + 1e-8)
    f1          = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)

    # AUC via trapezoidal rule
    thresholds  = np.linspace(0, 1, 100)
    tprs, fprs  = [], []
    for thr in thresholds:
        p   = (y_prob >= thr).astype(int)
        tpr = ((p == 1) & (y_true == 1)).sum() / (y_true.sum() + 1e-8)
        fpr = ((p == 1) & (y_true == 0)).sum() / ((y_true == 0).sum() + 1e-8)
        tprs.append(float(tpr))
        fprs.append(float(fpr))
    auc = float(np.trapz(tprs[::-1], fprs[::-1]))

    return {
        "accuracy":    float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision":   float(precision),
        "f1_score":    float(f1),
        "auc_roc":     float(auc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.outputs,     exist_ok=True)

    # ── load base healthy-skin data ──
    print("\nLoading healthy-skin dataset …")
    _, _, hsi_stats, _, hsi_full = build_datasets(
        zip_path=args.zip, patch_size=args.patch_size, seed=args.seed
    )

    # extract all HSI patches from full image for lesion synthesis
    from dataset import make_patches
    import zipfile, io
    from PIL import Image

    # re-load rgb+hsi for patch extraction
    with zipfile.ZipFile(args.zip, "r") as z:
        names = z.namelist()
        rgb_name = [n for n in names if n.lower().endswith(".bmp")][0]
        with z.open(rgb_name) as f:
            rgb_arr = np.array(Image.open(io.BytesIO(f.read())).convert("RGB"))
        band_names = sorted([n for n in names if n.lower().endswith(".png") and "face_ms_" in n.lower()])
        bands = []
        for bn in band_names:
            with z.open(bn) as f:
                bands.append(np.array(Image.open(io.BytesIO(f.read()))).astype(np.float32))
        hsi_arr = np.stack(bands, axis=-1)

    _, hsi_patches_all = make_patches(rgb_arr, hsi_arr, patch_size=args.patch_size)

    # ── load MST++ checkpoint ──
    ckpt_path = os.path.join(args.checkpoints, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run train.py first."
        )

    model = MSTpp(
        in_channels=3, out_channels=31,
        base_ch=args.base_ch, num_blocks=args.num_blocks, num_heads=args.num_heads
    ).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded MST++ from epoch {ckpt['epoch']}  (val_loss={ckpt['val_loss']:.5f})")

    # ── Phase A: domain adaptation ──
    model, lesion_dataset, labels = adapt_model(
        args, model, hsi_patches_all, hsi_stats, device
    )

    # ── Phase B: classifier ──
    classifier, metrics = train_classifier(args, model, lesion_dataset, device)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Sensitivity : {metrics['sensitivity']:.4f}")
    print(f"  Specificity : {metrics['specificity']:.4f}")
    print(f"  AUC-ROC     : {metrics['auc_roc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
