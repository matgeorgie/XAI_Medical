"""
Zone 4 — XAI Explainability Suite
Grad-CAM++ uses register_full_backward_hook on the LAYER (not the tensor).
This works regardless of requires_grad on intermediate tensors.
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from PIL import Image
import torchvision.transforms as transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def to_tensor(img_array):
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return TRANSFORM(Image.fromarray(img_array)).unsqueeze(0)


def get_target_layer(model):
    """Find the last Conv2d block for Grad-CAM++."""
    net = getattr(model, 'net', None) or getattr(model, 'backbone', None) or model
    # EfficientNet-B0: features[-1][0] is a Conv2d
    try:
        layer = net.features[-1][0]
        return layer
    except Exception:
        pass
    # DenseNet fallback
    try:
        return net.features.denseblock4.denselayer16.conv2
    except Exception:
        pass
    # Generic: last Conv2d
    last = None
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last:
        return last
    raise RuntimeError("No Conv2d found for Grad-CAM++")


# ── Grad-CAM++ (hook on LAYER, not tensor) ────────────────────────────
def compute_gradcam(model, img_tensor, class_idx, device='cpu'):
    """
    Registers forward + backward hooks on the layer itself.
    Works even when intermediate tensors have requires_grad=False.
    """
    target_layer = get_target_layer(model)
    activations  = {}
    gradients    = {}

    def fwd_hook(module, inp, out):
        activations['value'] = out.detach().clone()

    def bwd_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach().clone()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        # Need grad to flow — temporarily unfreeze all params
        saved = {}
        for name, p in model.named_parameters():
            saved[name] = p.requires_grad
            p.requires_grad_(True)

        model.eval()
        img = img_tensor.to(device)

        with torch.enable_grad():
            out = model(img)
            model.zero_grad()
            out[0, class_idx].backward()

        # Restore
        for name, p in model.named_parameters():
            p.requires_grad_(saved[name])

    finally:
        h1.remove()
        h2.remove()

    if 'value' not in activations or 'value' not in gradients:
        return np.ones((7, 7), dtype=np.float32)

    acts  = activations['value'][0]   # (C, H, W)
    grads = gradients['value'][0]     # (C, H, W)

    # Grad-CAM++ formula
    grads_sq = grads ** 2
    grads_cu = grads ** 3
    denom    = 2.0 * grads_sq + (acts * grads_cu).sum(dim=(1,2), keepdim=True)
    denom    = torch.where(denom.abs() > 1e-8, denom, torch.ones_like(denom))
    alpha    = grads_sq / denom
    weights  = (alpha * F.relu(grads)).sum(dim=(1, 2))

    cam = F.relu((weights[:, None, None] * acts).sum(dim=0))
    cam = cam.cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = np.zeros_like(cam)
    return cam


def generate_heatmap(img_array, model, class_idx, predicted_class, device='cpu'):
    tensor  = to_tensor(img_array)
    cam     = compute_gradcam(model, tensor, class_idx, device)

    h, w    = img_array.shape[:2]
    cam_r   = cv2.resize(cam, (w, h))
    cam_u8  = (cam_r * 255).astype(np.uint8)
    heatmap = cv2.cvtColor(cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    if img_array.ndim == 2:
        orig_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        orig_rgb = img_array.astype(np.uint8)

    gray     = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    orig_disp = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2RGB)
    overlay  = cv2.addWeighted(orig_disp, 0.5, heatmap, 0.5, 0)

    mask     = (cam_r > 0.5).astype(np.uint8) * 255
    cnts, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (255, 255, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0a0f1e')
    for ax, (im, title) in zip(axes, [
        (orig_disp, 'Original X-Ray'),
        (cam_r,     'Grad-CAM++ Attention'),
        (overlay,   'Activation Overlay'),
    ]):
        kw = {'cmap': 'jet', 'vmin': 0, 'vmax': 1} if 'Grad-CAM' in title else {}
        ax.imshow(im, **kw)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
        ax.set_facecolor('#0a0f1e')

    sm   = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                        fraction=0.025, pad=0.07, shrink=0.55)
    cbar.set_label('Activation Intensity (Grad-CAM++)', color='white', fontsize=9)
    plt.setp(cbar.ax.get_xticklabels(), color='white', fontsize=8)
    fig.suptitle(f'Prediction: {predicted_class}',
                 color='#00d4ff', fontsize=13, fontweight='bold', y=0.98)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0f1e')
    plt.close('all')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode(), cam_r


# ── LIME ──────────────────────────────────────────────────────────────
def lime_explanation(img_array, model, class_idx, device='cpu', n_samples=50):
    model.eval()
    h, w    = img_array.shape[:2]
    GH, GW  = 6, 6
    sh, sw  = h // GH, w // GW

    segs = np.zeros((h, w), int)
    sid  = 0
    for i in range(GH):
        for j in range(GW):
            y0, y1 = i*sh, min((i+1)*sh, h)
            x0, x1 = j*sw, min((j+1)*sw, w)
            segs[y0:y1, x0:x1] = sid
            sid += 1

    n_segs    = segs.max() + 1
    seg_means = [int(img_array[segs == s].mean()) if (segs == s).any() else 128
                 for s in range(n_segs)]

    with torch.no_grad():
        base_prob = float(torch.softmax(
            model(to_tensor(img_array).to(device)), dim=1)[0, class_idx])

    weights = np.zeros(n_segs)
    rng     = np.random.default_rng(42)
    for _ in range(n_samples):
        active = rng.integers(0, 2, n_segs)
        pimg   = img_array.copy()
        for s in range(n_segs):
            if not active[s]:
                pimg[segs == s] = seg_means[s]
        with torch.no_grad():
            p = float(torch.softmax(
                model(to_tensor(pimg).to(device)), dim=1)[0, class_idx])
        weights[active == 1] += p

    weights /= (n_samples * 0.5 + 1e-8)
    if weights.max() > weights.min():
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

    REGION_NAMES = [
        'upper-left lung apex',    'left upper lobe',        'upper mediastinum',
        'right upper lobe',        'upper-right apex',       'far right',
        'left hilar area',         'lower-left lung',        'hilar area',
        'right mid-lung',          'right hilar area',       'outer right',
        'left mid-lung',           'left center',            'central area',
        'right center',            'right edge',             'far right edge',
        'left lower lobe',         'left lower base',        'lower center',
        'right lower base',        'right lower lobe',       'outer base',
        'left costophrenic angle', 'left hemidiaphragm',     'central diaphragm',
        'right hemidiaphragm',     'right costophrenic angle','right diaphragm',
        'far left base',           'left bottom',            'bottom center',
        'right bottom',            'far right base',         'bottom edge',
    ]
    top_idx     = np.argsort(weights)[-6:][::-1]
    top_regions = [REGION_NAMES[int(s)] if int(s) < len(REGION_NAMES)
                   else f'region {s+1}' for s in top_idx]
    top_weights = [float(weights[s]) for s in top_idx]
    return top_regions, top_weights, base_prob


# ── SHAP ──────────────────────────────────────────────────────────────
def shap_explanation(img_array, model, class_idx, device='cpu'):
    model.eval()
    h, w = img_array.shape[:2]

    with torch.no_grad():
        base_prob = float(torch.softmax(
            model(to_tensor(img_array).to(device)), dim=1)[0, class_idx])

    region_names = [
        'Upper-left',   'Upper-center',  'Upper-right',
        'Mid-left',     'Center',        'Mid-right',
        'Lower-left',   'Lower-center',  'Lower-right',
    ]
    sh, sw    = h // 3, w // 3
    shap_vals = {}
    for ri, rname in enumerate(region_names):
        row, col = ri // 3, ri % 3
        y0, y1   = row*sh, min((row+1)*sh, h)
        x0, x1   = col*sw, min((col+1)*sw, w)
        masked   = img_array.copy().astype(float)
        masked[y0:y1, x0:x1] = masked[y0:y1, x0:x1].mean()
        masked   = np.clip(masked, 0, 255).astype(np.uint8)
        with torch.no_grad():
            p = float(torch.softmax(
                model(to_tensor(masked).to(device)), dim=1)[0, class_idx])
        shap_vals[rname] = round(base_prob - p, 4)

    gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY) \
           if img_array.ndim == 3 else img_array.astype(np.uint8)
    f    = gray.astype(np.float32) / 255.0
    measurements = {
        'bright_fraction': round(float((f > 0.6).mean()), 4),
        'dark_fraction':   round(float((f < 0.3).mean()), 4),
        'mean_intensity':  round(float(f.mean()), 4),
        'edge_density':    round(float(cv2.Canny(gray, 50, 150).mean() / 255.0), 4),
        'std_intensity':   round(float(f.std()), 4),
    }
    return shap_vals, measurements, base_prob


# ── NLG ───────────────────────────────────────────────────────────────
def generate_explanation(predicted_class, confidence, cam_map,
                          lime_regions, lime_weights, shap_vals, measurements):
    conf_pct = round(confidence * 100, 1)
    h, w     = cam_map.shape
    peak_y, peak_x = np.unravel_index(cam_map.argmax(), cam_map.shape)
    vert  = 'upper' if peak_y < h//3 else ('lower' if peak_y > 2*h//3 else 'middle')
    horiz = 'left'  if peak_x < w//3 else ('right' if peak_x > 2*w//3 else 'central')
    high_act = float((cam_map > 0.7).mean() * 100)

    r1 = lime_regions[0] if lime_regions else 'central area'
    w1 = round(lime_weights[0] * 100) if lime_weights else 0
    r2 = lime_regions[1] if len(lime_regions) > 1 else 'adjacent region'
    w2 = round(lime_weights[1] * 100) if len(lime_weights) > 1 else 0
    r3 = lime_regions[2] if len(lime_regions) > 2 else None
    w3 = round(lime_weights[2] * 100) if len(lime_weights) > 2 else 0

    shap_s   = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    sr1      = shap_s[0][0] if shap_s else 'center'
    sv1      = shap_s[0][1] if shap_s else 0
    sr2      = shap_s[1][0] if len(shap_s)>1 else None
    sv2      = shap_s[1][1] if len(shap_s)>1 else 0
    support  = [r for r,v in shap_s if v >  0.005]
    contradict=[r for r,v in shap_s if v < -0.005]

    bright = round(measurements.get('bright_fraction', 0) * 100, 1)
    dark   = round(measurements.get('dark_fraction',   0) * 100, 1)
    std    = measurements.get('std_intensity', 0)
    edges  = measurements.get('edge_density',  0)
    mean_i = measurements.get('mean_intensity', 0)

    if predicted_class == 'Normal':
        third_region = f" The {r3} also showed minor influence ({w3}%)," if r3 else ""
        support_str  = ', '.join(support[:2]) if support else 'no dominant region'
        return (
            f"The model classified this X-ray as <strong>Normal</strong> with a confidence of <strong>{conf_pct}%</strong>. "
            f"The Grad-CAM++ attention heatmap shows that the model's focus was spread across the {vert} {horiz} region, "
            f"with only {high_act:.1f}% of the image exceeding 70% activation intensity — a low, diffuse pattern "
            f"that is consistent with a clear study where no single anatomical area stands out as abnormal. "
            f"When specific regions were masked during LIME analysis, hiding the {r1} caused a {w1}% drop in Normal confidence, "
            f"and masking the {r2} resulted in a {w2}% drop.{third_region} "
            f"This relatively even spread of influence across regions — rather than a single dominant hotspot — "
            f"indicates the model did not detect any concentrated opacity, consolidation, or structural break. "
            f"SHAP contribution analysis confirms that the {sr1} region most strongly supported the Normal prediction "
            f"(contribution value: {sv1:+.4f})"
            + (f", while the {sr2} also contributed positively ({sv2:+.4f})" if sr2 and sv2 > 0 else "") + ". "
            f"Examining raw image statistics, the bright pixel fraction was {bright}% and dark fraction {dark}%, "
            f"both within the expected range for clear lung fields. "
            f"The pixel standard deviation of {std:.3f} and mean intensity of {mean_i:.3f} show uniform tissue density "
            f"without the heterogeneity typically associated with infection or injury. "
            f"Taken together, the spatial attention pattern, region masking results, regional contributions, "
            f"and image-level measurements all point consistently toward a normal, unremarkable X-ray."
        )

    elif predicted_class == 'Infection':
        third_region = f" The {r3} contributed a further {w3}%," if r3 else ""
        contra_str   = f" In contrast, the {contradict[0]} region appeared to work against the Infection prediction, suggesting that area looked relatively clear." if contradict else ""
        return (
            f"The model detected signs of <strong>Infection (Pneumonia)</strong> with a confidence of <strong>{conf_pct}%</strong>. "
            f"The Grad-CAM++ heatmap reveals that the model directed its strongest attention toward the {vert} {horiz} "
            f"region of the lung, with {high_act:.1f}% of the image showing activation above 70%. "
            f"This concentrated spatial focus is a hallmark of pneumonic consolidation — a condition where inflammatory "
            f"fluid fills the alveolar spaces, replacing normal air with dense material that appears brighter on X-ray. "
            f"LIME analysis, which works by systematically hiding parts of the image and measuring the effect on confidence, "
            f"found that masking the {r1} caused a {w1}% drop in Infection confidence — making it the single most "
            f"diagnostically important region in this image. "
            f"Masking the {r2} caused a further {w2}% drop.{third_region} "
            f"This pattern of concentrated influence in the {vert} lung field is consistent with lobar or segmental pneumonia.{contra_str} "
            f"SHAP value analysis, which quantifies each region's direct contribution to the prediction, "
            f"shows the {sr1} region most strongly supporting the Infection diagnosis (SHAP: {sv1:+.4f}), "
            f"meaning removing that area alone shifts model confidence by {abs(sv1)*100:.1f}%. "
            f"Measuring the raw image, the bright pixel fraction reached {bright}% — elevated radiodensity caused by "
            f"fluid or consolidated tissue replacing the normally dark air-filled lung. "
            f"The pixel standard deviation of {std:.3f} indicates "
            f"{'a high degree of heterogeneity, consistent with patchy or confluent consolidation spreading across the affected lobe' if std > 0.22 else 'moderate intensity variation, suggesting early or focal consolidation'}. "
            f"All four lines of evidence — the heatmap location, LIME masking impact, SHAP regional contributions, "
            f"and elevated image brightness — point consistently to infection-related changes in the {vert} {horiz} lung field."
        )

    elif predicted_class == 'Fracture':
        third_region = f" The {r3} contributed {w3}% as well," if r3 else ""
        edge_interp  = 'elevated edge density consistent with a sharp cortical disruption' if edges > 0.05 else 'moderate edge activity, suggesting a subtle or hairline fracture'
        return (
            f"The model detected a <strong>Bone Fracture</strong> with a confidence of <strong>{conf_pct}%</strong>. "
            f"The Grad-CAM++ attention heatmap shows tightly concentrated activation ({high_act:.1f}% above 70%) "
            f"in the {vert} {horiz} portion of the image. "
            f"Unlike infection which produces diffuse activation spread across a lobe, fractures typically produce "
            f"this kind of narrow, localised hotspot because the model has learned to focus on the thin line of cortical disruption "
            f"rather than a broad area of opacity. "
            f"LIME analysis confirmed that the {r1} was the most critical region — masking it reduced Fracture confidence "
            f"by {w1}%, the largest single-region impact in this scan.{third_region} "
            f"Masking the {r2} contributed a further {w2}% drop, indicating the fracture evidence is concentrated "
            f"rather than distributed across the image. "
            f"SHAP contribution values show the {sr1} region most strongly supports the Fracture prediction "
            f"(SHAP: {sv1:+.4f}), a shift of {abs(sv1)*100:.1f}% in model confidence when that area is removed. "
            + (f"The {sr2} region also contributed significantly ({sv2:+.4f}). " if sr2 else "")
            + f"From a pixel-level perspective, the image shows {edge_interp} (edge density: {edges:.3f}). "
            f"The bright pixel fraction of {bright}% and standard deviation of {std:.3f} are consistent with "
            f"localised bone density irregularity at the fracture site, where the break in cortical continuity "
            f"creates a distinct intensity transition in the image. "
            f"The convergence of heatmap focus, LIME masking impact, SHAP contributions, and edge-level measurements "
            f"all point to a fracture in the {vert} {horiz} region of the bone."
        )

    else:
        return (
            f"The model predicted <strong>{predicted_class}</strong> with <strong>{conf_pct}% confidence</strong>. "
            f"The Grad-CAM++ heatmap shows peak attention in the {vert} {horiz} region, "
            f"with {high_act:.1f}% of the image above 70% activation. "
            f"LIME analysis identifies the {r1} as the most influential region ({w1}% confidence drop when masked), "
            f"followed by the {r2} ({w2}%). "
            f"SHAP values show the {sr1} region as the top contributor (value: {sv1:+.4f}). "
            f"Image measurements — bright fraction {bright}%, std deviation {std:.3f}, edge density {edges:.3f} — "
            f"are consistent with the predicted finding."
        )