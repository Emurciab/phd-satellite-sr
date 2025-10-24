import arosics
from arosics import COREG
import rasterio
import numpy as np
import matplotlib.pyplot as plt


tif_paths = ['/content/drive/MyDrive/PhD/FASABE2025/Corregistered/correg_PlanetTrial_DL250625.tif', #2020coreg
            '/content/drive/MyDrive/PlanetSTA2c3c4_0214_2019_DL250625.tif', #ref
            '/content/drive/MyDrive/PlanetSTA2c3c4_0204_2020_DL250625.tif'] #src or tar

def coreg(ref_path, tar_path, out_path, verbose=False, tif_paths=tif_paths, verbose_img=False,**kwargs):

    coreg = COREG(ref_path, tar_path, path_out= out_path,
                fmt_out= 'GTiff', r_b4match=4, s_b4match=4, max_shift=1, nodata=(0,0))
    coreg.calculate_spatial_shifts()
    coreg.correct_shifts()
    coreg.save_corrected_image()

    def minmax_match(src_band, ref_band, eps=1e-5):
        src_min, src_max = np.nanpercentile(src_band, (2, 98))
        ref_min, ref_max = np.nanpercentile(ref_band, (2, 98))
        norm = (src_band - src_min) / max((src_max - src_min), eps)
        return norm * max((ref_max - ref_min), eps) + ref_min


    with rasterio.open(tif_paths[0]) as src_coreg: #2020Coreg
        nodata = src_coreg.nodata if src_coreg.nodata is not None else -32768
        src_coreg = src_coreg.read([1,2,3,4])
        src_coreg = np.where(src_coreg == nodata, np.nan, src_coreg)

    with rasterio.open(tif_paths[0]) as src_nocor: #2020
        nodata_nc = src_nocor.nodata if src_nocor.nodata is not None else -32768
        src_nocor = src_nocor.read([1,2,3,4])
        src_nocor = np.where(src_nocor == nodata_nc, np.nan, src_nocor)

    with rasterio.open(tif_paths[1]) as ref: #2019
        nodata = ref.nodata if ref.nodata is not None else -32768
        ref = ref.read([1,2,3,4])
        ref = np.where(ref == nodata, np.nan, ref)

    min_rows = min(ref.shape[1], src_coreg.shape[1])
    min_cols = min(ref.shape[2], src_coreg.shape[2])

    ref_cropped = ref[:, :min_rows, :min_cols]
    src_coreg_cropped = src_coreg[:, :min_rows, :min_cols]

    diff_nir = ref_cropped[3, :, :] - src_coreg_cropped[3, :, :]  # Band 4 = NIR

    if verbose_img:
        plt.imshow(np.clip(diff_nir, -1000, 1000), cmap='RdBu')
        plt.colorbar()
        plt.title("NIR Difference: Ref (2019) - Coregistered (2020 Coreg)")
        plt.show()

    min_rows_nc = min(ref.shape[1], src_nocor.shape[1])
    min_cols_nc = min(ref.shape[2], src_nocor.shape[2])

    ref_cropped_nc = ref[:, :min_rows_nc, :min_cols_nc]
    src_nocor_cropped = src_nocor[:, :min_rows_nc, :min_cols_nc]

    diff_nir_nc = ref_cropped_nc[3, :, :] - src_nocor_cropped[3, :, :]  # Band 4 = NIR



    src_coreg = np.transpose(src_coreg, (1, 2, 0))


    ref = np.transpose(ref, (1, 2, 0))
    

    harmonized = np.zeros_like(src_coreg)
    for i in range(src_coreg.shape[-1]):
        harmonized[..., i] = minmax_match(src_coreg[..., i], ref[..., i])

    # Create RGB composite (assume B, G, R = bands 1, 2, 3)
    rgb = src_coreg[..., [2, 1, 0]]

    rgb_harm = harmonized[..., [2, 1, 0]]

    # Apply 2–98% contrast stretch per channel
    rgb_stretched = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        band = rgb[..., i]
        p2, p98 = np.nanpercentile(band, (2, 98))
        stretched = np.clip((band - p2) / (p98 - p2), 0, 1)
        rgb_stretched[..., i] = stretched

    # Apply 2–98% contrast stretch per channel
    rgb_harm_stretched = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        band_harm = rgb_harm[..., i]
        p2_harm, p98_harm = np.nanpercentile(band_harm, (2, 98))
        stretched_harm = np.clip((band_harm - p2_harm) / (p98_harm - p2_harm), 0, 1)
        rgb_harm_stretched[..., i] = stretched_harm

    if verbose:
        print('x_shift_map',coreg.x_shift_map, 'y_shift_map',coreg.y_shift_map, 'x_shift_px', coreg.x_shift_px, 'y_shift_px', coreg.y_shift_px, 'mean shift', coreg.shift.mean())
        print('Shift reliability',coreg.shift_reliability)
        print(src_coreg.shape)
        print(ref.shape)

    if verbose_img:
        plt.imshow(np.clip(diff_nir_nc, -1000, 1000), cmap='RdBu')
        plt.colorbar()
        plt.title("NIR Difference: Ref (2019) - No Coregistered (2020)")
        plt.show()      
        # Display
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_stretched)
        plt.title("RGB")
        plt.axis("off")
        plt.show()  
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_harm_stretched)
        plt.title("RGB Harmonized")
        plt.axis("off")
        plt.show()


        plt.figure(figsize=(10, 4))
        for i, color in enumerate(['Red', 'Green', 'Blue']):
            plt.subplot(1, 3, i+1)
            plt.hist(src_coreg[..., i].ravel(), bins=100, alpha=0.5, label='Before', color='gray')
            plt.hist(harmonized[..., i].ravel(), bins=100, alpha=0.5, label='After', color='blue')
            plt.title(f'{color} Band')
            plt.legend()
        plt.tight_layout()
        plt.show()

    return coreg, harmonized, src_coreg,