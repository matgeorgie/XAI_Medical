"""
MedAI Vision — Full Dataset Training
======================================
Backbone  : EfficientNet-B0 (pretrained ImageNet)
Strategy  : Phase 1 — frozen backbone, train classifier (3 epochs)
            Phase 2 — unfreeze full network, fine-tune end-to-end (12 epochs)
Images    : ALL available images, no cap
Epochs    : 15 max per model (early stop patience=4)
Batch     : 32
LR        : Phase1=1e-3, Phase2=1e-4 (fine-tune)

Expected time on CPU (full dataset ~5000+ images):
  Lung Model  : 60-120 min
  Bone Model  : 60-120 min
  Router      : 20-40  min
  TOTAL       : ~2.5-5 hours (leave it running)

Run:  python train_cnn.py
"""

import os, sys, pickle, warnings, random, time
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from models_cnn import LungModel, BoneModel, ModalityRouter

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE    = 32
EPOCHS_P1     = 3       # Phase 1: frozen backbone — just warm up the classifier
EPOCHS_P2     = 12      # Phase 2: full fine-tune
PATIENCE      = 4       # early stop patience (epochs with no val improvement)
LR_P1         = 1e-3    # Phase 1 lr — fast convergence for new classifier layer
LR_P2         = 1e-4    # Phase 2 lr — small lr for fine-tuning pretrained weights
WEIGHT_DECAY  = 1e-4

print(f"\nDevice        : {DEVICE}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Strategy      : Phase1 (frozen, {EPOCHS_P1} ep) → Phase2 (full, {EPOCHS_P2} ep)")
print(f"Data cap      : ALL images (no limit)")
print(f"Estimated     : 2.5-5 hours on CPU\n")

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}
SKIP      = {'__macosx', '__pycache__', '.ds_store'}

# ── Augmentation (aggressive for full training) ──────────────────────────────
TRAIN_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Dataset ───────────────────────────────────────────────────────────────────
class XRayDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths = paths; self.labels = labels; self.tf = tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try:    return self.tf(Image.open(self.paths[i]).convert('RGB')), self.labels[i]
        except: return self.tf(Image.new('RGB', (224, 224), 0)), self.labels[i]

# ── Scan data folder ──────────────────────────────────────────────────────────
def scan():
    cn=[]; ci=[]; bn=[]; bf=[]; seen=set()
    CN  = {'normal'}
    CI  = {'pneumonia', 'infection', 'infected'}
    BN  = {'not fractured', 'notfractured', 'not_fractured'}
    BF  = {'fractured', 'fracture'}
    print(f"Scanning {DATA_DIR} ...")
    for root, dirs, files in os.walk(DATA_DIR):
        rel    = os.path.relpath(root, DATA_DIR).lower()
        if any(s in rel for s in SKIP): dirs[:] = []; continue
        folder = os.path.basename(root).lower().strip()
        parent = os.path.basename(os.path.dirname(root)).lower().strip()
        imgs   = [os.path.join(root, f) for f in files
                  if os.path.splitext(f)[1].lower() in VALID_EXT]
        if not imgs: continue
        new = [p for p in imgs
               if (k := os.path.normpath(p).lower()) not in seen and not seen.add(k)]
        if not new: continue
        short = root.replace(DATA_DIR, '').strip(os.sep)
        if   folder in CI or parent in CI:
            ci.extend(new); print(f"  [CHEST-INFECT] {short} ({len(new)})")
        elif folder in CN and 'bone' not in rel and 'fractur' not in rel:
            cn.extend(new); print(f"  [CHEST-NORMAL] {short} ({len(new)})")
        elif folder in BF or parent in BF:
            bf.extend(new); print(f"  [BONE-FRAC   ] {short} ({len(new)})")
        elif folder in BN or parent in BN:
            bn.extend(new); print(f"  [BONE-NORMAL ] {short} ({len(new)})")

    cn = list(dict.fromkeys(cn)); ci = list(dict.fromkeys(ci))
    bn = list(dict.fromkeys(bn)); bf = list(dict.fromkeys(bf))
    print(f"\n  ChestN:{len(cn)}  ChestI:{len(ci)}  BoneN:{len(bn)}  BoneF:{len(bf)}\n")
    return cn, ci, bn, bf

# ── Make data loaders — NO cap, all images ────────────────────────────────────
def make_loaders(p0, p1, names, cap=None):
    random.seed(42)
    if cap and len(p0) > cap: p0 = random.sample(p0, cap)
    if cap and len(p1) > cap: p1 = random.sample(p1, cap)

    # Balance classes — upsample minority to match majority
    max_n = max(len(p0), len(p1))
    if len(p0) < max_n:
        p0 = p0 + random.choices(p0, k=max_n - len(p0))
    if len(p1) < max_n:
        p1 = p1 + random.choices(p1, k=max_n - len(p1))

    paths  = p0 + p1
    labels = [0] * len(p0) + [1] * len(p1)
    Xtr, Xva, ytr, yva = train_test_split(
        paths, labels, test_size=0.15, random_state=42, stratify=labels)

    counts = np.bincount(ytr)
    wts    = [1.0 / counts[y] for y in ytr]

    tl = DataLoader(XRayDataset(Xtr, ytr, TRAIN_TF), batch_size=BATCH_SIZE,
                    sampler=WeightedRandomSampler(wts, len(wts)), num_workers=0,
                    pin_memory=False)
    vl = DataLoader(XRayDataset(Xva, yva, VAL_TF), batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=0)

    print(f"  Train: {len(Xtr)}  Val: {len(Xva)}  "
          f"Classes: {dict(zip(names, np.bincount(ytr).tolist()))}")
    return tl, vl

# ── Freeze/unfreeze helpers ────────────────────────────────────────────────────
def freeze_all(model):
    net = getattr(model, 'net', None) or getattr(model, 'backbone', None)
    if net is None: return
    for p in net.parameters(): p.requires_grad = False
    for p in net.classifier.parameters(): p.requires_grad = True
    t = sum(p.numel() for p in net.parameters() if p.requires_grad)
    a = sum(p.numel() for p in net.parameters())
    print(f"  [Phase 1] Trainable: {t:,} / {a:,}  ({t/a*100:.1f}%) — classifier only")

def unfreeze_all(model):
    net = getattr(model, 'net', None) or getattr(model, 'backbone', None)
    if net is None: return
    for p in net.parameters(): p.requires_grad = True
    t = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  [Phase 2] Trainable: {t:,}  — full network")

# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, opt, crit, train=True):
    model.train() if train else model.eval()
    loss_sum = 0; ok = 0; tot = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if train:
                opt.zero_grad()
                out  = model(imgs)
                loss = crit(out, labels)
                loss.backward(); opt.step()
                loss_sum += loss.item()
            else:
                out = model(imgs)
            ok  += (out.argmax(1) == labels).sum().item()
            tot += labels.size(0)
            if train:
                print(f"    batch {i+1}/{len(loader)}  "
                      f"loss={loss_sum/(i+1):.3f}  acc={ok/tot*100:.1f}%   ", end='\r')
    return ok / tot

def train_one(model, tl, vl, names, mname):
    model = model.to(DEVICE)
    crit  = nn.CrossEntropyLoss()
    path  = os.path.join(MODEL_DIR, f'{mname}.pth')
    best  = 0.0; no_imp = 0; t0 = time.time()
    all_p = []; all_l = []

    # ── Phase 1: frozen backbone ──────────────────────────────────────────────
    print(f"\n  --- Phase 1: warm up classifier ({EPOCHS_P1} epochs) ---")
    freeze_all(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt    = optim.Adam(params, lr=LR_P1, weight_decay=WEIGHT_DECAY)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_P1)

    for ep in range(EPOCHS_P1):
        tr_acc = run_epoch(model, tl, opt, crit, train=True)
        va_acc = run_epoch(model, vl, opt, crit, train=False)
        sched.step()
        sv = ''
        if va_acc > best:
            best = va_acc; torch.save(model.state_dict(), path); sv = ' ✓'; no_imp = 0
        else: no_imp += 1
        print(f"  P1 Epoch {ep+1}/{EPOCHS_P1}  train={tr_acc*100:.1f}%  "
              f"val={va_acc*100:.1f}%  best={best*100:.1f}%{sv}  "
              f"[{(time.time()-t0)/60:.1f}min]")

    # ── Phase 2: full fine-tune ───────────────────────────────────────────────
    print(f"\n  --- Phase 2: full fine-tune ({EPOCHS_P2} epochs, lr={LR_P2}) ---")
    unfreeze_all(model)
    opt   = optim.Adam(model.parameters(), lr=LR_P2, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_P2, eta_min=1e-6)
    no_imp = 0

    for ep in range(EPOCHS_P2):
        tr_acc = run_epoch(model, tl, opt, crit, train=True)
        va_acc = run_epoch(model, vl, opt, crit, train=False)
        sched.step()
        sv = ''
        if va_acc > best:
            best = va_acc; torch.save(model.state_dict(), path); sv = ' ✓'
            no_imp = 0
            # collect predictions on val for final report
            model.eval(); all_p = []; all_l = []
            with torch.no_grad():
                for imgs, labels in vl:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    all_p.extend(model(imgs).argmax(1).cpu().numpy())
                    all_l.extend(labels.cpu().numpy())
        else:
            no_imp += 1
        print(f"  P2 Epoch {ep+1}/{EPOCHS_P2}  train={tr_acc*100:.1f}%  "
              f"val={va_acc*100:.1f}%  best={best*100:.1f}%{sv}  "
              f"[{(time.time()-t0)/60:.1f}min]")
        if no_imp >= PATIENCE:
            print(f"  Early stop (no improvement for {PATIENCE} epochs)")
            break

    elapsed = (time.time() - t0) / 60
    print(f"\n  Best val accuracy : {best*100:.1f}%  ({elapsed:.1f} min total)")
    if all_l:
        print(classification_report(all_l, all_p, target_names=names, digits=3))
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return model, best

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t_start = time.time()
    print("=" * 60)
    print("  MedAI Vision — Full Dataset Training")
    print("  All images · Two-phase · EfficientNet-B0")
    print("=" * 60)

    cn, ci, bn, bf = scan()
    if not cn or not ci: print("ERROR: No chest data found"); sys.exit(1)
    if not bn or not bf: print("ERROR: No bone data found");  sys.exit(1)

    # ── Lung model ────────────────────────────────────────────────────────────
    print(f"\n>>> [1/3] LUNG MODEL  (elapsed: {(time.time()-t_start)/60:.1f}min)")
    print(f"  Using ALL {len(cn)} normal + {len(ci)} infection images")
    tl, vl = make_loaders(cn, ci, LungModel.CLASSES)
    lm, la = train_one(LungModel(), tl, vl, LungModel.CLASSES, 'lung_model')

    # ── Bone model ────────────────────────────────────────────────────────────
    print(f"\n>>> [2/3] BONE MODEL  (elapsed: {(time.time()-t_start)/60:.1f}min)")
    print(f"  Using ALL {len(bn)} normal + {len(bf)} fracture images")
    tb, vb = make_loaders(bn, bf, BoneModel.CLASSES)
    bm, ba = train_one(BoneModel(), tb, vb, BoneModel.CLASSES, 'bone_model')

    # ── Router ────────────────────────────────────────────────────────────────
    print(f"\n>>> [3/3] ROUTER  (elapsed: {(time.time()-t_start)/60:.1f}min)")
    random.seed(42)
    # Router uses all images from both modalities — no cap
    cs = cn + ci
    bs = bn + bf
    print(f"  Using ALL {len(cs)} chest + {len(bs)} bone images")
    tr_l, vr_l = make_loaders(cs, bs, ['chest', 'bone'])
    rm, ra = train_one(ModalityRouter(), tr_l, vr_l, ['chest', 'bone'], 'router')

    # ── Save manifest ─────────────────────────────────────────────────────────
    manifest = {
        'model_type'       : 'cnn_dual',
        'backbone'         : 'EfficientNet-B0',
        'training'         : 'full_dataset',
        'lung_classes'     : LungModel.CLASSES,
        'bone_classes'     : BoneModel.CLASSES,
        'lung_accuracy'    : la,
        'bone_accuracy'    : ba,
        'router_accuracy'  : ra,
        'device'           : DEVICE,
        'lung_model_path'  : os.path.join(MODEL_DIR, 'lung_model.pth'),
        'bone_model_path'  : os.path.join(MODEL_DIR, 'bone_model.pth'),
        'router_model_path': os.path.join(MODEL_DIR, 'router.pth'),
    }
    with open(os.path.join(MODEL_DIR, 'manifest.pkl'), 'wb') as f:
        pickle.dump(manifest, f)

    total = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"  LUNG   : {la*100:.1f}%")
    print(f"  BONE   : {ba*100:.1f}%")
    print(f"  ROUTER : {ra*100:.1f}%")
    print(f"  Total time : {total:.0f} minutes  ({total/60:.1f} hours)")
    print(f"{'='*60}")
    print("\n  *** TRAINING COMPLETE — run: python app.py ***\n")