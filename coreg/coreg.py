import arosics
from arosics import COREG
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def img_kernel(img, min, max):
    img_i = img.copy()
    kernel_i = img_i[:..., min:max, min:max]
    return kernel_i

def coregister_images(ref, tar, r_bmatch=4, s_bmatch=4, max_shift=1, nodata=(0,0)):

    """Coregister target image to reference image using AROSICS."""

    coreg = COREG(ref,tar,
                fmt_out= 'GTiff', r_b4match=r_bmatch, s_b4match=s_bmatch, max_shift=max_shift, nodata=nodata)
    coreg.calculate_spatial_shifts()
    coreg.correct_shifts()

    src_coreg = coreg.corrected_data
    ref = coreg.ref_data 

    return coreg, src_coreg

def harmonize_images(ref, coreg, kernel_size, verbose=False, verbose_img=False):
    """Harmonize source image to reference image using min-max matching."""
    
    def minmax_match(src_band, ref_band, eps=1e-5):
        src_min, src_max = np.nanpercentile(src_band, (2, 98))
        ref_min, ref_max = np.nanpercentile(ref_band, (2, 98))
        norm = (src_band - src_min) / max((src_max - src_min), eps)
        return norm * max((ref_max - ref_min), eps) + ref_min
    
    # Transpose to (bands, height, width)
    coreg = np.transpose(coreg, (1, 2, 0))
    ref = np.transpose(ref, (1, 2, 0))
    
    # Crop to minimum dimensions
    min_rows = min(ref.shape[1], coreg.shape[1])
    min_cols = min(ref.shape[2], coreg.shape[2])
    ref_cropped = ref[:, :min_rows, :min_cols]
    coreg_cropped = coreg[:, :min_rows, :min_cols]
    
    # Initialize full harmonized image
    harmonized_full = np.zeros_like(coreg_cropped)
    
    # Process each kernel and merge
    num_kernels = (min_cols + kernel_size - 1) // kernel_size  # ceiling division
    
    for j in range(kernel_size, min_cols + kernel_size, kernel_size):
        start_col = j - kernel_size
        end_col = min(j, min_cols)
        
        # Extract kernel regions
        img_kernel_ = coreg_cropped[:, :, start_col:end_col]
        ref_kernel_ = ref_cropped[:, :, start_col:end_col]
        
        # Harmonize each band in the kernel
        harmonized_kernel = np.zeros_like(img_kernel_)
        for i in range(img_kernel_.shape[0]):
            harmonized_kernel[i, :, :] = minmax_match(img_kernel_[i, :, :], ref_kernel_[i, :, :])
        
        # Merge kernel into full harmonized image
        harmonized_full[:, :, start_col:end_col] = harmonized_kernel
        
        if verbose:
            print(f"Processed kernel {start_col}:{end_col} ({(end_col/min_cols)*100:.1f}%)")
    
    # Visualization (if needed)
    if verbose_img:
        # Create RGB composite (B, G, R = bands 0, 1, 2 in typical order)
        rgb = np.transpose(coreg_cropped[[2, 1, 0], :, :], (1, 2, 0))
        rgb_harm = np.transpose(harmonized_full[[2, 1, 0], :, :], (1, 2, 0))
        
        # Apply 2–98% contrast stretch per channel
        rgb_stretched = np.zeros_like(rgb, dtype=np.float32)
        for k in range(3):
            band = rgb[..., k]
            p2, p98 = np.nanpercentile(band, (2, 98))
            stretched = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
            rgb_stretched[..., k] = stretched
        
        rgb_harm_stretched = np.zeros_like(rgb_harm, dtype=np.float32)
        for k in range(3):
            band_harm = rgb_harm[..., k]
            p2_harm, p98_harm = np.nanpercentile(band_harm, (2, 98))
            stretched_harm = np.clip((band_harm - p2_harm) / (p98_harm - p2_harm + 1e-8), 0, 1)
            rgb_harm_stretched[..., k] = stretched_harm
        
        # Display side-by-side comparison

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb_stretched)
        axes[0].set_title('Original Coregistered')
        axes[0].axis('off')
        
        axes[1].imshow(rgb_harm_stretched)
        axes[1].set_title('Harmonized')
        axes[1].axis('off')
        
        axes[2].imshow(np.transpose(ref_cropped[[2, 1, 0], :, :], (1, 2, 0)))
        axes[2].set_title('Reference')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Calculate metrics
    if verbose:
        mse = np.mean((ref_cropped - harmonized_full)**2)
        if mse > 0:
            PSNR = 10 * np.log10((np.max(ref_cropped)**2) / mse)
        else:
            PSNR = float('inf')
        
        print(f"PSNR: {PSNR:.2f} dB")
        
        # SSIM calculation (if arosics is available)
        try:
            
            SSIM = ssim(ref_cropped, harmonized_full, 
                       data_range=ref_cropped.max() - ref_cropped.min(), 
                       channel_axis=0)
            print(f"SSIM: {SSIM:.4f}")
        except ImportError:
            print("SSIM calculation requires scikit-image")
    
    return harmonized_full