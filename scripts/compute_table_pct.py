#!/usr/bin/env python3
"""Compute percentage improvements for the temporal comparison table."""
import numpy as np

bl = {'psnr': 32.23, 'ssim': 0.8324, 'rmse': 0.0494, 'ergas': 12.74, 'sid': 0.06521}
higher = {'psnr', 'ssim'}

block1 = {
    'Temporal':          {'psnr': 35.56, 'ssim': 0.8939, 'rmse': 0.0282, 'ergas': 7.10, 'sid': 0.03828},
    'MultiOut SchedPer': {'psnr': 35.81, 'ssim': 0.8952, 'rmse': 0.0259, 'ergas': 6.74, 'sid': 0.03387},
    'SchedPer Adaptive': {'psnr': 35.92, 'ssim': 0.8959, 'rmse': 0.0256, 'ergas': 6.66, 'sid': 0.03274},
}
block1_fail = {
    'BestFrame':     {'psnr': 27.35, 'ssim': 0.4720, 'rmse': 0.0669, 'ergas': 20.37, 'sid': 0.13187},
    'Median':        {'psnr': 27.65, 'ssim': 0.4798, 'rmse': 0.0670, 'ergas': 20.15, 'sid': 0.10613},
    'MultiOutput':   {'psnr': 26.31, 'ssim': 0.6807, 'rmse': 0.1015, 'ergas': 16.28, 'sid': 7.12063},
    'SchedPer Meta': {'psnr': 25.88, 'ssim': 0.4543, 'rmse': 0.1137, 'ergas': 20.52, 'sid': 0.16768},
}
block2 = {
    'BestFrame':          {'psnr': 34.57, 'ssim': 0.8812, 'rmse': 0.0364, 'ergas': 7.79, 'sid': 0.05025},
    'Median':             {'psnr': 33.78, 'ssim': 0.8671, 'rmse': 0.0432, 'ergas': 9.08, 'sid': 0.06231},
    'MultiOutput':        {'psnr': 35.45, 'ssim': 0.8913, 'rmse': 0.0266, 'ergas': 6.91, 'sid': 0.03344},
    'MultiOut SchedPer':  {'psnr': 35.47, 'ssim': 0.8896, 'rmse': 0.0268, 'ergas': 6.95, 'sid': 0.04326},
    'SchedPer Adaptive':  {'psnr': 36.07, 'ssim': 0.8970, 'rmse': 0.0252, 'ergas': 6.56, 'sid': 0.03501},
}
block2_fail = {
    'Temporal': {'psnr': 27.83, 'ssim': 0.5955, 'rmse': 0.0610, 'ergas': 17.81, 'sid': 7.49396},
}

def pct(val, metric):
    b = bl[metric]
    if metric in higher:
        return (val - b) / abs(b) * 100
    else:
        return (b - val) / abs(b) * 100

metrics = ['psnr', 'ssim', 'rmse', 'ergas', 'sid']

def show(name, vals, failed=False):
    parts = []
    for m in metrics:
        p = pct(vals[m], m)
        parts.append(f'{m.upper()}: {p:+.1f}%')
    tag = ' [FAILED]' if failed else ''
    print(f'  {name:<22} {"  ".join(parts)}{tag}')

print('=== BLOCK I: Direct ===')
for n, v in block1_fail.items():
    show(n, v, True)
for n, v in block1.items():
    show(n, v)
print('  --- Mean (converged) ---')
for m in metrics:
    vals = [pct(v[m], m) for v in block1.values()]
    print(f'    Mean {m.upper()}: {np.mean(vals):+.1f}%')

print('\n=== BLOCK II: Residual ===')
for n, v in block2_fail.items():
    show(n, v, True)
for n, v in block2.items():
    show(n, v)
print('  --- Mean (converged) ---')
for m in metrics:
    vals = [pct(v[m], m) for v in block2.values()]
    print(f'    Mean {m.upper()}: {np.mean(vals):+.1f}%')
