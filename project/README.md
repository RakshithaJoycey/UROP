# Physics-Informed RGB → HSI Reconstruction with Domain Adaptation

Full implementation of the proposed framework:
- **Stage 1**: MST++ backbone for RGB → 31-band HSI reconstruction on healthy facial skin
- **Stage 2**: Bio-optical physics-informed loss (melanin, HbO2, Hb constraints)
- **Stage 3**: Domain adaptation to pathological (lesion) skin
- **Stage 4**: Melanoma vs benign classification on reconstructed HSI

---

## Project Structure

```
project/
├── src/
│   ├── mst_plus_plus.py    MST++ spectral transformer backbone
│   ├── bio_loss.py         Physics-informed loss (chromophore + smoothness + monotonicity)
│   ├── lesion_dataset.py   Synthetic lesion HSI generation + ISIC support
│   ├── classifier.py       Spectral feature extractor + melanoma classifier
│   ├── dataset.py          Face HSI data loading, patching, normalisation
│   ├── train.py            Stage 1+2: MST++ training with bio-optical loss
│   ├── adapt.py            Stage 3+4: Domain adaptation + classifier training
│   ├── evaluate.py         RMSE / SAM / PSNR / SSIM evaluation
│   └── visualize.py        Visualisation figures
├── checkpoints/            Saved model weights + stats
├── outputs/                Results, history, figures
├── data/
│   └── isic/               (Optional) Real ISIC dermoscopy images
│       ├── melanoma/
│       └── benign/
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Stage 1 + 2: Train MST++ with bio-optical loss

```bash
python src/train.py --zip ../face_ms.zip
```

Key flags:
```
--epochs        50      max training epochs
--lr            2e-4    AdamW learning rate
--base_ch       32      MST++ channel width (doubles per stage)
--num_blocks    2       spectral transformer blocks per stage
--lambda_chrom  0.05    chromophore consistency loss weight
--lambda_smooth 0.01    spectral smoothness loss weight
--lambda_mono   0.01    melanin monotonicity loss weight
```

### Evaluate reconstruction

```bash
python src/evaluate.py --zip ../face_ms.zip
```

Reports: RMSE, SAM (degrees), PSNR (dB), SSIM

### Visualise

```bash
python src/visualize.py --zip ../face_ms.zip
```

### Stage 3 + 4: Domain adaptation + classification

```bash
# Synthetic lesion data (no external data needed)
python src/adapt.py --zip ../face_ms.zip

# With real ISIC images (place in data/isic/melanoma/ and data/isic/benign/)
python src/adapt.py --zip ../face_ms.zip --isic_dir ../data/isic
```

Key flags:
```
--adapt_epochs   30     domain adaptation epochs
--adapt_lr       5e-5   fine-tune learning rate
--freeze_epochs  5      epochs to keep encoder frozen before full fine-tune
--n_melanoma     80     synthetic melanoma patches to generate
--n_benign       80     synthetic benign patches to generate
--clf_epochs     30     classifier training epochs
```

---

## Architecture: MST++

Multi-Scale Spectral Transformer — computes attention across the **spectral
dimension** rather than spatial, which is efficient (O(C²HW)) and naturally
captures inter-band correlations critical for physically consistent reconstruction.

```
RGB (3,H,W)
    ↓ stem conv
    ↓ Encoder: 3× [downsample + SpectralTransformerBlocks]
    ↓ Bottleneck: SpectralTransformerBlocks
    ↓ Decoder: 3× [upsample + skip connection + SpectralTransformerBlocks]
    ↓ 1×1 head
HSI (31,H,W)
```

~6.4M parameters at base_ch=32.

---

## Bio-Optical Loss

Three physics-informed regularisation terms added to MSE:

| Term | What it enforces |
|------|-----------------|
| Chromophore consistency | Predicted spectrum should be explainable by melanin + HbO2 + Hb basis (Beer-Lambert) |
| Spectral smoothness | Penalises rapid oscillations between adjacent bands (unphysical artefacts) |
| Melanin monotonicity | Melanin absorption decreases monotonically 400→700 nm |

---

## Domain Adaptation Strategy

1. Load MST++ pre-trained on healthy face skin
2. Freeze encoder, fine-tune decoder on lesion HSI for N epochs
3. Unfreeze encoder, continue with lower LR
4. Train spectral classifier on frozen adapted MST++ features

Lesion HSI is synthesised by applying spatially-varying melanin and
haemoglobin absorption perturbations to healthy face patches, simulating
the spectral signatures of melanoma and benign naevi.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Pixel-level reconstruction error |
| SAM | Spectral Angle Mapper (degrees) — spectral shape fidelity |
| PSNR (dB) | Signal quality, >30 dB = good |
| SSIM | Structural similarity per band |
| Accuracy | Classification accuracy |
| Sensitivity | True positive rate (melanoma detection) |
| Specificity | True negative rate |
| AUC-ROC | Area under ROC curve |
