"""Inspect the learned cross-resolution fusion gate parameters."""
import torch

ckpt = torch.load(
    '/media/eamurciab/TOSHIBA_EXT1/PhD/SR/checkpoints/multires_multioutput_schedPer_t0.5_adaptive_direct_learnedUps_full/best_model.pt',
    map_location='cpu', weights_only=False
)
state = ckpt['model_state_dict']

print("=== CROSS-RESOLUTION FUSION PARAMETERS ===\n")

keys_of_interest = [
    'fusion_10m_from_20m.0.weight',
    'fusion_10m_from_20m.0.bias',
    'fusion_20m_from_10m.0.weight',
    'fusion_20m_from_10m.0.bias',
    'fusion_gate.0.weight',
    'fusion_gate.0.bias',
]

for k in keys_of_interest:
    if k in state:
        w = state[k]
        print(f"{k}:")
        print(f"  shape={list(w.shape)} mean={w.mean():.4f} std={w.std():.4f} min={w.min():.4f} max={w.max():.4f}")
    else:
        print(f"{k}: NOT FOUND")

print()

# Analyze the gate bias
b = state.get('fusion_gate.0.bias')
if b is not None:
    s = torch.sigmoid(b)
    print(f"Gate sigmoid(bias) -- default gate values:")
    print(f"  mean={s.mean():.4f} std={s.std():.4f} min={s.min():.4f} max={s.max():.4f}")
    total = s.numel()
    below = (s < 0.3).sum().item()
    mid = ((s >= 0.3) & (s <= 0.7)).sum().item()
    above = (s > 0.7).sum().item()
    print(f"  <0.3 (mostly ignore cross-res): {below}/{total} ({100*below/total:.1f}%)")
    print(f"  0.3-0.7 (active mixing):        {mid}/{total} ({100*mid/total:.1f}%)")
    print(f"  >0.7 (heavy cross-res use):      {above}/{total} ({100*above/total:.1f}%)")

# Check weight magnitude comparison between the two transform modules
w10 = state.get('fusion_10m_from_20m.0.weight')
w20 = state.get('fusion_20m_from_10m.0.weight')
if w10 is not None and w20 is not None:
    print(f"\nTransform weight L2 norms:")
    print(f"  10m_from_20m: {w10.norm():.4f}")
    print(f"  20m_from_10m: {w20.norm():.4f}")
    # Cosine similarity between the two transforms
    cos = torch.nn.functional.cosine_similarity(w10.flatten().unsqueeze(0), w20.flatten().unsqueeze(0))
    print(f"  Cosine similarity between transforms: {cos.item():.4f}")
    print(f"  (0 = orthogonal/independent, 1 = identical)")

# Also check residual scales
for name in ['residual_scale_10m', 'residual_scale_20m']:
    if name in state:
        print(f"\n{name}: {state[name].item():.6f}")
