# DeepFER: shared code for the notebook, the Streamlit app and the live webcam script.
# The notebook writes this file, so the app and the webcam script use exactly the same steps.
import copy
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG = 48                                           # every image in the dataset is 48x48 grayscale
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device():
    """NVIDIA GPU on Colab, the Apple GPU (MPS) on a Mac, otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    """Wait for the GPU to finish, so timings are real."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def log_line(msg):
    """Print a training line. If the DEEPFER_LOG variable names a file, the line is also added to it,
    which makes it easy to follow a long training run from outside the notebook."""
    import os
    print(msg)
    if os.environ.get("DEEPFER_LOG"):
        with open(os.environ["DEEPFER_LOG"], "a") as fh:
            fh.write(msg + "\n")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- data
def load_images(root):
    """Read every jpg under root/<folder>/<emotion>/ into a table plus one uint8 array (n, 48, 48)."""
    rows, pix = [], []
    for folder in ("train", "validation"):
        for label in CLASSES:
            for f in sorted(Path(root, folder, label).glob("*.jpg")):
                img = Image.open(f)
                rows.append((f"{folder}/{label}/{f.name}", folder, label, img.size[0], img.size[1], img.mode))
                pix.append(np.array(img.convert("L").resize((IMG, IMG))))
    df = pd.DataFrame(rows, columns=["file", "folder", "label", "width", "height", "mode"])
    return df, np.stack(pix)


def hog_lbp_features(X, use_clahe=False):
    """Handcrafted features for the classic baseline: HOG (edge directions in 8x8 cells, 900 numbers)
    plus LBP texture histograms in a 4x4 grid of regions (160 numbers)."""
    from skimage.feature import hog, local_binary_pattern
    out = np.empty((len(X), 1060), dtype=np.float32)
    for k, a in enumerate(X):
        if use_clahe:
            a = equalize(a)
        h = hog(a, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        lbp = local_binary_pattern(a, P=8, R=1, method="uniform").astype(int)
        hist = [np.bincount(lbp[r:r + 12, c:c + 12].ravel(), minlength=10) / 144 for r in range(0, 48, 12) for c in range(0, 48, 12)]
        out[k] = np.concatenate([h, np.concatenate(hist)])
    return out


def image_stats(X, chunk=4000):
    """Simple numbers per image: brightness, contrast, sharpness, edge strength by face region and more."""
    out = []
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk].astype(np.float32)
        gy, gx = np.gradient(x, axis=(1, 2))
        edge = np.hypot(gx, gy)
        lap = x[:, :-2, 1:-1] + x[:, 2:, 1:-1] + x[:, 1:-1, :-2] + x[:, 1:-1, 2:] - 4 * x[:, 1:-1, 1:-1]
        flat = []
        for a in X[s:s + chunk]:
            h = np.bincount(a.ravel(), minlength=256)
            m = h.argmax()
            flat.append(h[max(0, m - 3):m + 4].sum() / a.size)
        p = np.stack([np.bincount(a.ravel(), minlength=256) for a in X[s:s + chunk]]) / (IMG * IMG)
        ent = -(p * np.log2(np.where(p > 0, p, 1))).sum(1)
        out.append(pd.DataFrame({
            "brightness": x.mean((1, 2)),
            "contrast": x.std((1, 2)),
            "sharpness": lap.var((1, 2)),
            "entropy": ent,
            "edge_eyes": edge[:, 10:24, 6:42].mean((1, 2)),     # band with the brows and eyes
            "edge_nose": edge[:, 22:32, 14:34].mean((1, 2)),
            "edge_mouth": edge[:, 31:45, 10:38].mean((1, 2)),   # band with the mouth
            "asymmetry": np.abs(x - x[:, :, ::-1]).mean((1, 2)),
            "flat_share": flat,
            "hash": [hashlib.md5(a.tobytes()).hexdigest() for a in X[s:s + chunk]],
        }))
    return pd.concat(out, ignore_index=True)


def clean_dataset(df, non_face_files, blank_contrast=5.0):
    """Mark the rows to drop, in this order: blank images, non-face images, copies with conflicting
    labels, training copies of a test image, and repeated copies inside the same folder."""
    df = df.copy()
    df["drop_reason"] = ""
    df.loc[df["contrast"] < blank_contrast, "drop_reason"] = "blank image"
    df.loc[(df["drop_reason"] == "") & df["file"].isin(set(non_face_files)), "drop_reason"] = "not a face"
    keep = df["drop_reason"] == ""
    n_labels = df[keep].groupby("hash")["label"].transform("nunique")
    df.loc[n_labels[n_labels > 1].index, "drop_reason"] = "same image, different labels"
    keep = df["drop_reason"] == ""
    test_hashes = set(df.loc[keep & (df["folder"] == "validation"), "hash"])
    leak = keep & (df["folder"] == "train") & df["hash"].isin(test_hashes)
    df.loc[leak, "drop_reason"] = "copy of a test image"
    keep = df["drop_reason"] == ""
    dup = keep & df[keep].duplicated(["folder", "hash"]).reindex(df.index, fill_value=False)
    df.loc[dup, "drop_reason"] = "duplicate in the same folder"
    return df


# ---------------------------------------------------------------- augmentation on the GPU
def augment(x, rot=15, scale=0.12, shift=0.1, flip=True, bright=0.2, contrast=0.3, erase_p=0.25):
    """Random flip, rotation, zoom, shift, lighting change and a cut-out patch, for a whole batch at once.
    x: (B, 1, 48, 48) floats in [0, 1] on any device. Every image gets its own random values."""
    B, dev = x.size(0), x.device
    u = lambda *s: torch.rand(*s, device=dev) * 2 - 1           # uniform in [-1, 1]
    ang = u(B) * math.radians(rot)
    sc = 1 + u(B) * scale
    fl = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0) if flip else torch.ones(B, device=dev)
    t = u(B, 2) * shift
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * fl, -sin, t[:, 0]], 1),
                         torch.stack([sin * fl, cos, t[:, 1]], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False),
                      padding_mode="border", align_corners=False)
    m = x.mean((2, 3), keepdim=True)
    x = ((x - m) * (1 + u(B, 1, 1, 1) * contrast) + m + u(B, 1, 1, 1) * bright).clamp(0, 1)
    if erase_p > 0:                                              # random erasing: hide a patch
        H = x.size(-1)
        h = (torch.rand(B, device=dev) * 0.25 + 0.1) * H
        w = (torch.rand(B, device=dev) * 0.25 + 0.1) * H
        y0 = torch.rand(B, device=dev) * (H - h)
        x0 = torch.rand(B, device=dev) * (H - w)
        r = torch.arange(H, device=dev).float()
        my = (r.view(1, H) >= y0.view(B, 1)) & (r.view(1, H) < (y0 + h).view(B, 1))
        mx = (r.view(1, H) >= x0.view(B, 1)) & (r.view(1, H) < (x0 + w).view(B, 1))
        mask = (my.view(B, H, 1) & mx.view(B, 1, H)) & (torch.rand(B, 1, 1, device=dev) < erase_p)
        x = torch.where(mask.unsqueeze(1), torch.full_like(x, 0.5), x)
    return x


# ---------------------------------------------------------------- models
class InputAdapter(nn.Module):
    """Every saved model takes the same input: grayscale 48x48 in [0, 1]. This layer resizes it,
    copies it to 3 channels for the ImageNet models and normalises it."""

    def __init__(self, size, rgb, mean, std):
        super().__init__()
        self.size, self.rgb = size, rgb
        c = 3 if rgb else 1
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, c, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, c, 1, 1))

    def forward(self, x):
        if self.size != x.size(-1):
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        if self.rgb:
            x = x.expand(-1, 3, -1, -1)
        return (x - self.mean) / self.std


class SqueezeExcite(nn.Module):
    """Channel attention: the block learns which feature maps matter for the current face."""

    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // r, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(c // r, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class ResBlock(nn.Module):
    """Two 3x3 convolutions with batch norm, squeeze-excitation and a shortcut connection."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), SqueezeExcite(cout))
        self.skip = nn.Identity() if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class DeepFERNet(nn.Module):
    """Custom CNN for 48x48 faces: a stem at full resolution, then three residual stages that halve
    the size (48 -> 24 -> 12 -> 6) while the channels grow (64 -> 128 -> 256 -> 512)."""

    def __init__(self, n_classes=7, width=64, dropout=0.4):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(1, w // 2, 3, 1, 1, bias=False), nn.BatchNorm2d(w // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(w // 2, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(ResBlock(w, 2 * w, 2), ResBlock(2 * w, 2 * w))
        self.stage2 = nn.Sequential(ResBlock(2 * w, 4 * w, 2), ResBlock(4 * w, 4 * w))
        self.stage3 = ResBlock(4 * w, 8 * w, 2)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(8 * w, n_classes))

    def forward(self, x):
        return self.head(self.stage3(self.stage2(self.stage1(self.stem(x)))))


def build_model(name, pretrained=True, dropout=0.3, size=128, pixel_mean=0.5, pixel_std=0.25):
    """deepfer_cnn (from scratch), resnet18 or efficientnet_b0 (ImageNet weights, new 7-class head)."""
    import torchvision
    if name == "deepfer_cnn":
        net = DeepFERNet(len(CLASSES), dropout=max(dropout, 0.4))
        return nn.Sequential(InputAdapter(IMG, False, [pixel_mean], [pixel_std]), net)
    if name == "resnet18":
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=w)
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, len(CLASSES)))
    elif name == "efficientnet_b0":
        w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.efficientnet_b0(weights=w)
        net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.classifier[1].in_features, len(CLASSES)))
    else:
        raise ValueError(name)
    return nn.Sequential(InputAdapter(size, True, IMAGENET_MEAN, IMAGENET_STD), net)


def head_and_body(model):
    """Split the parameters into the new classifier head and the rest (the pre-trained body)."""
    net = model[1]
    head = net.head if isinstance(net, DeepFERNet) else (net.fc if hasattr(net, "fc") else net.classifier)
    head_ids = {id(p) for p in head.parameters()}
    return list(head.parameters()), [p for p in net.parameters() if id(p) not in head_ids]


def cam_layer(model):
    """The last convolution block, used by Grad-CAM."""
    net = model[1]
    if isinstance(net, DeepFERNet):
        return net.stage3
    return net.layer4 if hasattr(net, "layer4") else net.features[-1]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- training
def to_batch(xu8, device):
    """uint8 (B, 48, 48) -> float (B, 1, 48, 48) in [0, 1]."""
    return xu8.to(device).float().div(255).unsqueeze(1)


def amp_ok(device):
    return device.type in ("cuda", "mps")


def amp_dtype(device):
    """Half precision for speed: bfloat16 on the Apple GPU (same number range as float32, so no loss scaling
    is needed), float16 with loss scaling on an NVIDIA GPU."""
    return torch.bfloat16 if device.type == "mps" else torch.float16


@torch.no_grad()
def predict_logits(model, X, device, batch=512, tta=False):
    """Raw scores for a uint8 tensor of faces. tta=True also scores the mirrored face and averages."""
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        x = to_batch(X[s:s + batch], device)
        with torch.autocast(device.type, dtype=amp_dtype(device), enabled=amp_ok(device)):
            z = model(x).float()
            if tta:
                z = (z + model(torch.flip(x, dims=[3])).float()) / 2
        out.append(z.cpu())
    return torch.cat(out)


def scores(y, pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return {"accuracy": accuracy_score(y, pred),
            "macro precision": precision_score(y, pred, average="macro", zero_division=0),
            "macro recall": recall_score(y, pred, average="macro", zero_division=0),
            "macro F1": f1_score(y, pred, average="macro", zero_division=0),
            "weighted F1": f1_score(y, pred, average="weighted", zero_division=0)}


def train_model(model, Xtr, ytr, Xva, yva, device, epochs=30, batch=128, lr=1e-3, lr_body=None,
                weight_decay=1e-4, label_smoothing=0.1, class_weights=None, patience=6,
                freeze_body=False, aug=None, seed=42, log=log_line):
    """Mini-batch training with AdamW, a one-cycle learning rate (warm-up then cosine decay),
    mixed precision on the GPU (bfloat16 on Apple, float16 on NVIDIA), GPU augmentation and early stopping
    on validation macro F1.
    The weights of the best epoch are put back at the end."""
    set_seed(seed)
    model = model.to(device)
    head, body = head_and_body(model)
    for p in body:
        p.requires_grad_(not freeze_body)
    groups = [{"params": head, "lr": lr}]
    if not freeze_body:
        groups.append({"params": body, "lr": lr_body or lr})
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    steps = math.ceil(len(Xtr) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=epochs * steps,
                                                pct_start=0.15, div_factor=10, final_div_factor=100)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    w = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=device)
    ytr = ytr.to(Xtr.device)
    aug = aug or {}
    best, best_state, wait, hist = -1, None, 0, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs + 1):
        model.train()
        if freeze_body:                      # keep the frozen body's batch-norm statistics fixed
            for m in model[1].modules():
                if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        t0, tot, n = time.time(), 0.0, 0
        order = torch.randperm(len(Xtr), generator=g).to(Xtr.device)
        for s in range(0, len(Xtr), batch):
            idx = order[s:s + batch]
            x = augment(to_batch(Xtr[idx], device), **aug)
            yb = ytr[idx].to(device)
            with torch.autocast(device.type, dtype=amp_dtype(device), enabled=amp_ok(device)):
                loss = F.cross_entropy(model(x).float(), yb, weight=w, label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        zva = predict_logits(model, Xva, device)
        va_loss = F.cross_entropy(zva, yva).item()
        sc = scores(yva.numpy(), zva.argmax(1).numpy())
        hist.append({"epoch": ep, "train loss": tot / n, "val loss": va_loss, "val accuracy": sc["accuracy"],
                     "val macro F1": sc["macro F1"], "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0})
        log(f"epoch {ep:2d}  train loss {tot / n:.3f}  val loss {va_loss:.3f}  val acc {sc['accuracy']:.3f}  "
            f"val macro F1 {sc['macro F1']:.3f}  ({time.time() - t0:.0f}s)")
        if sc["macro F1"] > best + 1e-4:
            best, wait = sc["macro F1"], 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            wait += 1
            if wait >= patience:
                log(f"early stop: no gain for {patience} epochs, best val macro F1 {best:.3f}")
                break
    model.load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad_(True)
    return model, pd.DataFrame(hist)


def fit_temperature(logits, y):
    """Temperature scaling: one number that makes the probabilities honest (calibrated) on validation."""
    grid = np.linspace(0.5, 3.0, 51)
    nll = [F.cross_entropy(logits / t, y).item() for t in grid]
    return float(grid[int(np.argmin(nll))])


def save_model(model, path, name, extra=None):
    torch.save({"name": name, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "extra": extra or {}}, path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ex = ck.get("extra", {})
    model = build_model(ck["name"], pretrained=False, size=ex.get("size", 128),
                        pixel_mean=ex.get("pixel_mean", 0.5), pixel_std=ex.get("pixel_std", 0.25))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ex


# ---------------------------------------------------------------- explainability
def grad_cam(model, x, target=None):
    """Grad-CAM heat map: which parts of the face pushed the model towards the class.
    x: (B, 1, 48, 48) on the model's device. Returns maps (B, 48, 48) in [0, 1] and the classes used."""
    model.eval()
    layer, store = cam_layer(model), {}

    def keep(module, inputs, out):
        store["a"] = out
        out.register_hook(lambda grad: store.__setitem__("g", grad))

    hook = layer.register_forward_hook(keep)
    x = x.clone().requires_grad_(True)
    z = model(x)
    cls = z.argmax(1) if target is None else torch.as_tensor(target, device=x.device).view(-1)
    model.zero_grad()
    z.gather(1, cls.view(-1, 1)).sum().backward()
    hook.remove()
    a, g = store["a"], store["g"]
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / (cam.amax((1, 2), keepdim=True) + 1e-8)
    return cam.detach().cpu().numpy(), cls.cpu().numpy()


# ---------------------------------------------------------------- real time: faces in a photo or a video frame
class FaceFinder:
    """OpenCV's Haar cascade face detector (it ships with OpenCV, so nothing to download)."""

    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def find(self, gray, scale_to=480, min_face=40):
        """Boxes (x, y, w, h) of the faces in a grayscale image. Large images are shrunk first for speed."""
        f = min(1.0, scale_to / max(gray.shape))
        small = self.cv2.resize(gray, None, fx=f, fy=f) if f < 1 else gray
        boxes = self.det.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(int(min_face * f), int(min_face * f)))
        return [tuple(int(v / f) for v in b) for b in boxes] if len(boxes) else []


def crop_face(gray, box, scale=1.03, shift=0.02):
    """Cut the face out the same way the training faces were cut and make it 48x48. Measured on enlarged
    test faces: the dataset crop is almost exactly OpenCV's face box, 3% larger and 2% further right and down."""
    import cv2
    x, y, w, h = box
    size = int(round(w * scale))
    x0, y0 = int(round(x + shift * w)), int(round(y + shift * w))
    pad = size                                              # pad with edge pixels in case the face touches the border
    big = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    face = big[y0 + pad:y0 + pad + size, x0 + pad:x0 + pad + size]
    return cv2.resize(face, (IMG, IMG), interpolation=cv2.INTER_AREA)


def equalize(img):
    """CLAHE histogram equalisation: evens out dark or flat lighting, tile by tile."""
    import cv2
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(img)


class EmotionPredictor:
    """The deployed model: one or more saved networks (averaged), mirrored-face averaging and temperature,
    exactly as chosen in the notebook and written to models/model_card.json."""

    def __init__(self, model_dir="models", device="cpu"):
        import json
        self.card = json.loads(Path(model_dir, "model_card.json").read_text())
        self.device = torch.device(device)          # the CPU by default: fast enough for a few faces and works everywhere
        self.models = [load_model(Path(model_dir, f), self.device)[0] for f in self.card["members"]]
        self.temperature = self.card["temperature"]
        self.tta = self.card["tta"]

    @torch.no_grad()
    def logits(self, x):
        z = 0
        for m in self.models:
            z = z + m(x).float()
            if self.tta:
                z = z + m(torch.flip(x, dims=[3])).float()
        return z / (len(self.models) * (2 if self.tta else 1))

    def predict(self, faces):
        """Probabilities (n, 7) for a list or array of 48x48 uint8 faces."""
        if len(faces) == 0:
            return np.zeros((0, len(CLASSES)))
        x = to_batch(torch.from_numpy(np.stack(faces)), self.device)
        return torch.softmax(self.logits(x) / self.temperature, 1).cpu().numpy()

    def explain(self, face):
        """Grad-CAM heat map of the first network for one 48x48 face."""
        cam, _ = grad_cam(self.models[0], to_batch(torch.from_numpy(face[None]), self.device))
        return cam[0]


@torch.no_grad()
def predict_faces(model, faces, device, temperature=1.0):
    """Emotion probabilities for a list of 48x48 uint8 faces."""
    if len(faces) == 0:
        return np.zeros((0, len(CLASSES)))
    x = to_batch(torch.from_numpy(np.stack(faces)), device)
    model.eval()
    return torch.softmax(model(x).float() / temperature, 1).cpu().numpy()


def analyse_image(gray, finder, predictor):
    """The real-time pipeline for one photo or video frame: find faces, crop them like the training faces,
    and score them all in one batch. Returns the boxes, the 48x48 faces and the probabilities."""
    boxes = finder.find(gray)
    faces = [crop_face(gray, b) for b in boxes]
    return boxes, faces, predictor.predict(faces)
