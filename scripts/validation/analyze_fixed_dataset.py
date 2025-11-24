import pickle, numpy as np
from pathlib import Path
p=Path('fixed_dataset/training_samples.pkl')
with p.open('rb') as f:
    samples=pickle.load(f)
print('Loaded', len(samples), 'samples')
count_test=0
for i,s in enumerate(samples):
    hr = s.hr_sequence
    vals = hr[0,0,0,:]
    if np.allclose(vals, [300,500,700,600], atol=30):
        count_test+=1
    if i<5:
        print(f'sample{i} hr first pixel: {vals}, hr min/max: {hr.min()}/{hr.max()}')
print('test fallback count:', count_test, 'out of', len(samples))
