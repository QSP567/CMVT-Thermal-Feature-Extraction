"""
CMVT thermal feature extraction.

This module contains only the feature-calculation logic used to construct the
1,612-dimensional handcrafted thermal feature library reported in the study.

Input
-----
Eight normalized ROI temperature matrices for one participant:
    front_left_lower, front_left_upper, front_right_lower, front_right_upper,
    back_left_lower,  back_left_upper,  back_right_lower,  back_right_upper.

Anatomical convention:
    upper = calf
    lower = ankle-foot

Feature structure
-----------------
    1,352 ROI-level features = 8 ROIs x (136 + 33)
      116 CMVT-oriented features = 4 calf ROIs x 23 + 24 cross-ROI
      144 patient-level relational features = 120 + 24
    ---------------------------------------------------------------
    1,612 total features

For CMVT presence detection, the first 1,468 features (1,352 + 116) form the
feature representation. The full 1,612 features include the additional 144
patient-level relational features used for the three-class and four-class tasks.
"""

from typing import Dict

import numpy as np
from scipy.fft import fft2, fftshift
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    sobel,
)
from scipy.signal import find_peaks
from scipy.stats import kurtosis, ks_2samp, skew, wasserstein_distance
from skimage import filters
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from skimage.measure import label as sk_label
from skimage.measure import regionprops
from skimage.morphology import binary_erosion
from skimage.transform import resize

# Dimensional accounting used in the study.
EXPECTED_136_COUNT = 136
EXPECTED_70_FUNCTION_ACTUAL_COUNT = 33
EXPECTED_PER_ROI_FEATURES = 169
EXPECTED_LEGACY_TOTAL_FEATURES = 1352

CMVT_PRIMARY_PART = "upper"
CMVT_REFERENCE_PART = "lower"
CMVT_EPS = 1e-10
EXPECTED_CMVTLITE_PER_UPPER_ROI = 23
EXPECTED_CMVTLITE_UPPER_TOTAL = 92
EXPECTED_CMVTLITE_CROSS = 24
EXPECTED_CMVTLITE_TOTAL = 116
EXPECTED_BASE_TOTAL_FEATURES = 1468

RELV2_EPS = 1e-10
EXPECTED_RELV2_GENERAL = 120
EXPECTED_RELV2_BILATERAL = 24
EXPECTED_RELV2_TOTAL = 144
EXPECTED_TOTAL_FEATURES = 1612

RELV2_GENERAL_DESCRIPTORS = [
    "base_mean",
    "base_var",
    "base_grad",
    "base_contrast",
    "hotspot_contrast",
    "GLCM_contrast",
    "GLCM_homogeneity",
    "HF_ratio",
    "Orient_entropy",
    "dispersion_index",
]

RELV2_BILATERAL_CMVTDESCRIPTORS = [
    "valid_top10_mean",
    "localcontrast_s7_p95",
    "localcontrast_s15_top1mean",
    "hotring_deltaT",
    "hotring_area_fraction",
    "hotspot_persistence_auc",
]


class ThermalFeatureExtractor:
    """Feature calculations used in the reported 1,612-feature representation."""

    def extract_136d_features(self, data, prefix=""):
        """
        Extract 136 texture, spectral, directional, and related descriptors.

        Parameters:
            data: NumPy array containing the ROI temperature matrix.
            prefix: String prefix used for feature names.

        Returns:
            Dictionary containing 136 features.
        """
        features = {}

        # Downsample large matrices to accelerate feature computation.
        if data.shape[0] > 128 or data.shape[1] > 128:
            data = resize(data, (128, 128), preserve_range=True, anti_aliasing=True)

        # ============ 1. Edge and gradient features (12 dimensions) ============
        sobel_x = sobel(data, axis=0)
        sobel_y = sobel(data, axis=1)
        sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        features[f'{prefix}sobel_count'] = np.sum(sobel_mag > np.percentile(sobel_mag, 75))
        features[f'{prefix}sobel_mean'] = np.mean(sobel_mag)
        features[f'{prefix}sobel_var'] = np.var(sobel_mag)

        try:
            canny = filters.canny(data, sigma=1.0)
            features[f'{prefix}canny_count'] = np.sum(canny)
            features[f'{prefix}canny_density'] = np.sum(canny) / canny.size
            features[f'{prefix}canny_mean'] = np.mean(data[canny]) if np.any(canny) else 0
        except:
            features[f'{prefix}canny_count'] = 0
            features[f'{prefix}canny_density'] = 0
            features[f'{prefix}canny_mean'] = 0

        edge_map = sobel_mag > np.percentile(sobel_mag, 75)
        features[f'{prefix}edge_density'] = np.sum(edge_map) / edge_map.size

        edge_angles = np.arctan2(sobel_y, sobel_x)
        hist, _ = np.histogram(edge_angles[edge_map], bins=16, range=(-np.pi, np.pi))
        hist = hist / (np.sum(hist) + 1e-10)
        features[f'{prefix}edge_entropy'] = -np.sum(hist * np.log(hist + 1e-10))

        features[f'{prefix}gradient_mean'] = np.mean(sobel_mag)
        features[f'{prefix}gradient_var'] = np.var(sobel_mag)

        grad_hist, _ = np.histogram(sobel_mag, bins=32)
        grad_hist = grad_hist / (np.sum(grad_hist) + 1e-10)
        features[f'{prefix}grad_hist_mean'] = np.mean(grad_hist)
        features[f'{prefix}grad_hist_entropy'] = -np.sum(grad_hist * np.log(grad_hist + 1e-10))

        # ============ 2. Texture features (10 dimensions) ============
        data_normalized = ((data - data.min()) / (data.max() - data.min() + 1e-10) * 255).astype(np.uint8)

        try:
            glcm = graycomatrix(data_normalized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            features[f'{prefix}GLCM_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
            features[f'{prefix}GLCM_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
            features[f'{prefix}GLCM_energy'] = graycoprops(glcm, 'energy')[0, 0]
            features[f'{prefix}GLCM_entropy'] = -np.sum(glcm * np.log(glcm + 1e-10))
        except:
            features[f'{prefix}GLCM_contrast'] = 0
            features[f'{prefix}GLCM_homogeneity'] = 0
            features[f'{prefix}GLCM_energy'] = 0
            features[f'{prefix}GLCM_entropy'] = 0

        try:
            lbp = local_binary_pattern(data_normalized, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
            lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-10)
            features[f'{prefix}LBP_entropy'] = -np.sum(lbp_hist * np.log(lbp_hist + 1e-10))
            features[f'{prefix}LBP_uniformity'] = np.sum(lbp_hist ** 2)
            features[f'{prefix}LBP_variance'] = np.var(lbp)
        except:
            features[f'{prefix}LBP_entropy'] = 0
            features[f'{prefix}LBP_uniformity'] = 0
            features[f'{prefix}LBP_variance'] = 0

        try:
            gabor_real, gabor_imag = gabor(data, frequency=0.1)
            features[f'{prefix}gabor_mean'] = np.mean(np.abs(gabor_real))
            features[f'{prefix}gabor_var'] = np.var(np.abs(gabor_real))
            features[f'{prefix}gabor_dir_strength'] = np.std(gabor_real)
        except:
            features[f'{prefix}gabor_mean'] = 0
            features[f'{prefix}gabor_var'] = 0
            features[f'{prefix}gabor_dir_strength'] = 0

        # ============ 3. Frequency-domain features (26 dimensions) ============
        fft = fft2(data)
        fft_shifted = fftshift(fft)
        magnitude_spectrum = np.abs(fft_shifted)

        rows, cols = magnitude_spectrum.shape
        center_row, center_col = rows // 2, cols // 2
        radius = min(center_row, center_col)

        y, x = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)

        # High-frequency features.
        hf_mask = dist_from_center > radius / 2
        hf_spectrum = magnitude_spectrum[hf_mask]

        features[f'{prefix}HF_energy'] = np.sum(hf_spectrum ** 2)
        features[f'{prefix}HF_ratio'] = np.sum(hf_spectrum ** 2) / (np.sum(magnitude_spectrum ** 2) + 1e-10)

        hf_hist, _ = np.histogram(hf_spectrum, bins=32)
        hf_hist = hf_hist / (np.sum(hf_hist) + 1e-10)
        features[f'{prefix}HF_entropy'] = -np.sum(hf_hist * np.log(hf_hist + 1e-10))
        features[f'{prefix}HF_centroid'] = np.mean(hf_spectrum) if np.sum(hf_spectrum) > 0 else 0
        features[f'{prefix}HF_skew'] = skew(hf_spectrum.flatten()) if len(hf_spectrum) > 0 else 0
        features[f'{prefix}HF_kurtosis'] = kurtosis(hf_spectrum.flatten()) if len(hf_spectrum) > 0 else 0

        # Low-frequency features.
        lf_mask = dist_from_center <= radius / 4
        lf_spectrum = magnitude_spectrum[lf_mask]

        features[f'{prefix}LL_energy'] = np.sum(lf_spectrum ** 2)
        lf_hist, _ = np.histogram(lf_spectrum, bins=32)
        lf_hist = lf_hist / (np.sum(lf_hist) + 1e-10)
        features[f'{prefix}LL_entropy'] = -np.sum(lf_hist * np.log(lf_hist + 1e-10))
        features[f'{prefix}LL_var'] = np.var(lf_spectrum)

        # Legacy subband-named energy features computed from FFT quadrants.
        mid_row, mid_col = rows // 2, cols // 2
        LH = magnitude_spectrum[:mid_row, mid_col:]
        HL = magnitude_spectrum[mid_row:, :mid_col]
        HH = magnitude_spectrum[mid_row:, mid_col:]

        features[f'{prefix}LH_energy'] = np.sum(LH ** 2)
        features[f'{prefix}HL_energy'] = np.sum(HL ** 2)
        features[f'{prefix}HH_energy'] = np.sum(HH ** 2)

        # Radial spectrum.
        radial_bins = 20
        radial_profile = np.zeros(radial_bins)
        for i in range(radial_bins):
            r_min = i * radius / radial_bins
            r_max = (i + 1) * radius / radial_bins
            ring_mask = (dist_from_center >= r_min) & (dist_from_center < r_max)
            radial_profile[i] = np.mean(magnitude_spectrum[ring_mask]) if np.any(ring_mask) else 0

        features[f'{prefix}radial_energy'] = np.sum(radial_profile ** 2)

        if np.max(radial_profile) > 0:
            try:
                x_fit = np.arange(len(radial_profile))
                fit = np.polyfit(x_fit, np.log(radial_profile + 1e-10), 1)
                features[f'{prefix}radial_decay'] = -fit[0]
            except:
                features[f'{prefix}radial_decay'] = 0
        else:
            features[f'{prefix}radial_decay'] = 0

        radial_hist = radial_profile / (np.sum(radial_profile) + 1e-10)
        features[f'{prefix}radial_entropy'] = -np.sum(radial_hist * np.log(radial_hist + 1e-10))

        log_spec = np.log(magnitude_spectrum.flatten() + 1e-10)
        features[f'{prefix}slope_logspec'] = np.polyfit(np.arange(len(log_spec)), log_spec, 1)[0]

        # Angular spectrum.
        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        angular_profile = np.zeros(len(angles))

        for i, angle in enumerate(angles):
            x_mask = x - center_col
            y_mask = y - center_row
            angle_mask = np.abs(np.arctan2(y_mask, x_mask) - angle) < np.pi / 36
            angular_profile[i] = np.mean(magnitude_spectrum[angle_mask]) if np.any(angle_mask) else 0

        features[f'{prefix}FFT_E_theta'] = np.sum(angular_profile)
        features[f'{prefix}FFT_HF_theta'] = np.sum(angular_profile)
        features[f'{prefix}FFT_LF_theta'] = np.sum(angular_profile) * 0.5

        angular_hist = angular_profile / (np.sum(angular_profile) + 1e-10)
        features[f'{prefix}FFT_entropy_theta'] = -np.sum(angular_hist * np.log(angular_hist + 1e-10))

        if len(angular_profile) >= 18:
            features[f'{prefix}FFT_delta_dir'] = np.mean(np.abs(angular_profile - np.roll(angular_profile, 18)))
        else:
            features[f'{prefix}FFT_delta_dir'] = 0

        features[f'{prefix}FFT_HF_LF_ratio_theta'] = features[f'{prefix}FFT_HF_theta'] / (
                features[f'{prefix}FFT_LF_theta'] + 1e-10)
        features[f'{prefix}Fourier_Gabor_Contrast'] = features[f'{prefix}HF_energy'] / (
                features[f'{prefix}LL_energy'] + 1e-10)
        features[f'{prefix}Spectral_Texture_Index'] = features[f'{prefix}HF_energy'] * features.get(
            f'{prefix}GLCM_contrast', 1)

        mf_mask = (dist_from_center > radius / 4) & (dist_from_center <= radius / 2)
        mf_spectrum = magnitude_spectrum[mf_mask]
        mf_energy = np.sum(mf_spectrum ** 2)

        features[f'{prefix}Band_energy_ratio_MF_LF'] = mf_energy / (features[f'{prefix}LL_energy'] + 1e-10)
        features[f'{prefix}Band_energy_ratio_HF_MF'] = features[f'{prefix}HF_energy'] / (mf_energy + 1e-10)

        # ============ 4. Multiscale thermal features (24 dimensions) ============
        features[f'{prefix}base_mean'] = np.mean(data)
        features[f'{prefix}base_var'] = np.var(data)

        hist, _ = np.histogram(data, bins=32)
        hist = hist / (np.sum(hist) + 1e-10)
        features[f'{prefix}base_entropy'] = -np.sum(hist * np.log(hist + 1e-10))

        grad_x = np.gradient(data, axis=0)
        grad_y = np.gradient(data, axis=1)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        features[f'{prefix}base_grad'] = np.mean(grad_mag)
        features[f'{prefix}base_contrast'] = data.max() - data.min()

        smoothed = gaussian_filter(data, sigma=2)
        detail = data - smoothed

        features[f'{prefix}detail_energy'] = np.sum(detail ** 2)
        features[f'{prefix}detail_var'] = np.var(detail)

        detail_hist, _ = np.histogram(detail, bins=32)
        detail_hist = detail_hist / (np.sum(detail_hist) + 1e-10)
        features[f'{prefix}detail_entropy'] = -np.sum(detail_hist * np.log(detail_hist + 1e-10))

        detail_grad = np.gradient(detail)
        features[f'{prefix}detail_grad'] = np.mean(np.abs(detail_grad))

        for scale_name, sigma in [('L', 4), ('M', 2), ('S', 1)]:
            scale_data = gaussian_filter(data, sigma=sigma)
            features[f'{prefix}{scale_name}_mean'] = np.mean(scale_data)
            features[f'{prefix}{scale_name}_var'] = np.var(scale_data)

            scale_hist, _ = np.histogram(scale_data, bins=32)
            scale_hist = scale_hist / (np.sum(scale_hist) + 1e-10)
            features[f'{prefix}{scale_name}_entropy'] = -np.sum(scale_hist * np.log(scale_hist + 1e-10))

            scale_grad = np.gradient(scale_data)
            features[f'{prefix}{scale_name}_grad'] = np.mean(np.abs(scale_grad))

        # Hotspot features.
        threshold = np.percentile(data, 90)
        hotspot = data > threshold

        if np.any(hotspot):
            eroded = binary_erosion(hotspot)
            edge = hotspot & ~eroded
            features[f'{prefix}hotspot_edge_strength'] = np.mean(data[edge]) if np.any(edge) else 0
            hotspot_data = data[hotspot]
            features[f'{prefix}hotspot_clarity'] = np.std(hotspot_data)
            roi_mean = np.mean(data[hotspot])
            bg_mean = np.mean(data[~hotspot])
            features[f'{prefix}hotspot_contrast'] = roi_mean - bg_mean
        else:
            features[f'{prefix}hotspot_edge_strength'] = 0
            features[f'{prefix}hotspot_clarity'] = 0
            features[f'{prefix}hotspot_contrast'] = 0

        # ============ 5. Spatial-dispersion features ============
        threshold = np.percentile(data, 80)
        center_mask = data > threshold

        if np.any(center_mask):
            dist_transform = distance_transform_edt(~center_mask)
            features[f'{prefix}spread_mean'] = np.mean(dist_transform)
            features[f'{prefix}spread_var'] = np.var(dist_transform)
            features[f'{prefix}dispersion_index'] = features[f'{prefix}spread_var'] / (
                    features[f'{prefix}spread_mean'] + 1e-10)

            max_dist = int(np.max(dist_transform))
            if max_dist > 0:
                spread_profile = np.zeros(min(max_dist, 20))
                for i in range(len(spread_profile)):
                    mask = (dist_transform >= i) & (dist_transform < i + 1)
                    spread_profile[i] = np.mean(data[mask]) if np.any(mask) else 0

                features[f'{prefix}growth_rate'] = np.mean(np.diff(spread_profile)) if len(spread_profile) > 1 else 0
                spread_hist = spread_profile / (np.sum(spread_profile) + 1e-10)
                features[f'{prefix}curve_entropy'] = -np.sum(spread_hist * np.log(spread_hist + 1e-10))
                features[f'{prefix}curve_smoothness'] = np.std(np.diff(spread_profile)) if len(
                    spread_profile) > 1 else 0
            else:
                features[f'{prefix}growth_rate'] = 0
                features[f'{prefix}curve_entropy'] = 0
                features[f'{prefix}curve_smoothness'] = 0
        else:
            for feat in ['spread_mean', 'spread_var', 'dispersion_index', 'growth_rate', 'curve_entropy',
                         'curve_smoothness']:
                features[f'{prefix}{feat}'] = 0

        # ============ 6. Directional-gradient features (22 dimensions) ============
        G0 = sobel(data, axis=1)
        features[f'{prefix}G0_mean'] = np.mean(np.abs(G0))
        features[f'{prefix}G0_var'] = np.var(G0)
        features[f'{prefix}G0_max'] = np.max(np.abs(G0))

        G0_hist, _ = np.histogram(G0, bins=32)
        G0_hist = G0_hist / (np.sum(G0_hist) + 1e-10)
        features[f'{prefix}G0_hist_entropy'] = -np.sum(G0_hist * np.log(G0_hist + 1e-10))

        G45 = (sobel(data, axis=0) + sobel(data, axis=1)) / np.sqrt(2)
        features[f'{prefix}G45_mean'] = np.mean(np.abs(G45))
        features[f'{prefix}G45_var'] = np.var(G45)
        features[f'{prefix}G45_max'] = np.max(np.abs(G45))

        G45_hist, _ = np.histogram(G45, bins=32)
        G45_hist = G45_hist / (np.sum(G45_hist) + 1e-10)
        features[f'{prefix}G45_hist_entropy'] = -np.sum(G45_hist * np.log(G45_hist + 1e-10))

        G90 = sobel(data, axis=0)
        features[f'{prefix}G90_mean'] = np.mean(np.abs(G90))
        features[f'{prefix}G90_var'] = np.var(G90)
        features[f'{prefix}G90_max'] = np.max(np.abs(G90))

        G90_hist, _ = np.histogram(G90, bins=32)
        G90_hist = G90_hist / (np.sum(G90_hist) + 1e-10)
        features[f'{prefix}G90_hist_entropy'] = -np.sum(G90_hist * np.log(G90_hist + 1e-10))

        G135 = (sobel(data, axis=0) - sobel(data, axis=1)) / np.sqrt(2)
        features[f'{prefix}G135_mean'] = np.mean(np.abs(G135))
        features[f'{prefix}G135_var'] = np.var(G135)
        features[f'{prefix}G135_max'] = np.max(np.abs(G135))

        G135_hist, _ = np.histogram(G135, bins=32)
        G135_hist = G135_hist / (np.sum(G135_hist) + 1e-10)
        features[f'{prefix}G135_hist_entropy'] = -np.sum(G135_hist * np.log(G135_hist + 1e-10))

        gx = sobel(data, axis=1)
        gy = sobel(data, axis=0)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        grad_orient = np.arctan2(gy, gx)

        if np.sum(grad_mag) > 0:
            weighted_orient = grad_orient * grad_mag
            mean_orient = np.sum(weighted_orient) / np.sum(grad_mag)
            orient_diff = np.abs(grad_orient - mean_orient)
            orient_diff = np.minimum(orient_diff, 2 * np.pi - orient_diff)
            features[f'{prefix}GOC'] = 1.0 - np.mean(orient_diff) / np.pi
        else:
            features[f'{prefix}GOC'] = 0

        orient_hist, _ = np.histogram(grad_orient, bins=36, range=(-np.pi, np.pi))
        orient_hist = orient_hist / (np.sum(orient_hist) + 1e-10)
        features[f'{prefix}Orient_entropy'] = -np.sum(orient_hist * np.log(orient_hist + 1e-10))

        Ixx = gaussian_filter(gx ** 2, sigma=1.5)
        Iyy = gaussian_filter(gy ** 2, sigma=1.5)
        Ixy = gaussian_filter(gx * gy, sigma=1.5)

        lambda1 = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy ** 2))
        lambda2 = 0.5 * (Ixx + Iyy - np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy ** 2))

        features[f'{prefix}lambda1'] = np.mean(lambda1)
        features[f'{prefix}lambda2'] = np.mean(lambda2)
        features[f'{prefix}lambda_sum'] = features[f'{prefix}lambda1'] + features[f'{prefix}lambda2']
        features[f'{prefix}lambda_diff'] = features[f'{prefix}lambda1'] - features[f'{prefix}lambda2']

        # ============ 7. Periodicity features (12 dimensions) ============
        max_r = int(min(center_row, center_col))
        radial_profile_periodic = np.zeros(max_r)
        for i in range(max_r):
            mask = (dist_from_center >= i) & (dist_from_center < i + 1)
            radial_profile_periodic[i] = np.mean(magnitude_spectrum[mask]) if np.any(mask) else 0

        if len(radial_profile_periodic) > 2:
            peaks, _ = find_peaks(radial_profile_periodic, height=np.mean(radial_profile_periodic))
            if len(peaks) > 0:
                features[f'{prefix}periodic_freq'] = peaks[0] / max_r
                features[f'{prefix}periodic_spacing'] = max_r / (peaks[0] + 1e-10)
                features[f'{prefix}periodic_magnitude'] = radial_profile_periodic[peaks[0]]
            else:
                features[f'{prefix}periodic_freq'] = 0
                features[f'{prefix}periodic_spacing'] = 0
                features[f'{prefix}periodic_magnitude'] = 0
        else:
            features[f'{prefix}periodic_freq'] = 0
            features[f'{prefix}periodic_spacing'] = 0
            features[f'{prefix}periodic_magnitude'] = 0

        row_profile = np.mean(data, axis=0)
        autocorr = np.correlate(row_profile, row_profile, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]

        if len(autocorr) > 2:
            peaks, _ = find_peaks(autocorr, height=np.mean(autocorr))
            if len(peaks) > 1:
                features[f'{prefix}ACF_period'] = peaks[1] - peaks[0]
                features[f'{prefix}ACF_strength'] = autocorr[peaks[0]] / (autocorr[0] + 1e-10)
            else:
                features[f'{prefix}ACF_period'] = 0
                features[f'{prefix}ACF_strength'] = 0
        else:
            features[f'{prefix}ACF_period'] = 0
            features[f'{prefix}ACF_strength'] = 0

        features[f'{prefix}wavelet_peak'] = np.max(magnitude_spectrum)
        features[f'{prefix}wavelet_period'] = 1.0 / (features[f'{prefix}periodic_freq'] + 1e-10) if features[
                                                                                                        f'{prefix}periodic_freq'] > 0 else 0

        angles_list = [0, 45, 90, 135]
        freq_by_angle = []
        for angle in angles_list:
            if angle == 0:
                profile = np.mean(data, axis=0)
            elif angle == 90:
                profile = np.mean(data, axis=1)
            else:
                profile = np.mean(data, axis=0)

            fft_1d = np.fft.fft(profile)
            mag_1d = np.abs(fft_1d)
            peak_freq = np.argmax(mag_1d[1:len(mag_1d) // 2]) + 1
            freq_by_angle.append(peak_freq)

        features[f'{prefix}delta_freq_dir'] = np.std(freq_by_angle)
        features[f'{prefix}HF_LF_dir_contrast'] = np.ptp(freq_by_angle) / (np.mean(freq_by_angle) + 1e-10)

        stripe_hist, _ = np.histogram(freq_by_angle, bins=8)
        stripe_hist = stripe_hist / (np.sum(stripe_hist) + 1e-10)
        features[f'{prefix}stripe_shape_entropy'] = -np.sum(stripe_hist * np.log(stripe_hist + 1e-10))

        features[f'{prefix}dominant_periodic_dir'] = angles_list[np.argmax(freq_by_angle)]
        features[f'{prefix}periodic_coherence'] = 1.0 - features[f'{prefix}delta_freq_dir'] / (
                np.mean(freq_by_angle) + 1e-10)

        # ============ 8. Symmetry features (9 dimensions) ============
        left_half = data[:, :data.shape[1] // 2]
        right_half = np.fliplr(data[:, data.shape[1] // 2:])

        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]

        left_fft = np.abs(fft2(left_half))
        right_fft = np.abs(fft2(right_half))
        features[f'{prefix}HF_diff'] = np.mean(np.abs(left_fft - right_fft))
        features[f'{prefix}HF_ratio_lr'] = (np.sum(left_fft ** 2) + 1e-10) / (np.sum(right_fft ** 2) + 1e-10)
        features[f'{prefix}FFT_L2_diff'] = np.sqrt(np.sum((left_fft - right_fft) ** 2))

        mu_l = np.mean(left_half)
        mu_r = np.mean(right_half)
        sigma_l = np.std(left_half)
        sigma_r = np.std(right_half)
        sigma_lr = np.mean((left_half - mu_l) * (right_half - mu_r))

        c1, c2 = 0.01, 0.03
        ssim = ((2 * mu_l * mu_r + c1) * (2 * sigma_lr + c2)) / (
                (mu_l ** 2 + mu_r ** 2 + c1) * (sigma_l ** 2 + sigma_r ** 2 + c2))
        features[f'{prefix}SSIM_lr'] = ssim

        features[f'{prefix}contrast×spec_mean'] = features.get(f'{prefix}base_contrast', 1) * np.mean(
            left_fft + right_fft)
        features[f'{prefix}entropy×spec_std'] = features.get(f'{prefix}base_entropy', 1) * np.std(left_fft + right_fft)
        features[f'{prefix}homogeneity×spec_ratio'] = features.get(f'{prefix}GLCM_homogeneity', 1) * features[
            f'{prefix}HF_ratio_lr']
        features[f'{prefix}texture_energy×temp_mean'] = features.get(f'{prefix}GLCM_energy', 1) * features.get(
            f'{prefix}base_mean', 1)
        features[f'{prefix}texture_contrast×spec_entropy'] = features.get(f'{prefix}GLCM_contrast', 1) * features.get(
            f'{prefix}HF_entropy', 1)

        # ============ 9. Integrated descriptors (15 dimensions) ============
        center_col_idx = data.shape[1] // 2
        vertical_profile = data[:, center_col_idx]

        features[f'{prefix}LSV_mean'] = np.mean(vertical_profile)
        features[f'{prefix}LSV_var'] = np.var(vertical_profile)

        lsv_hist, _ = np.histogram(vertical_profile, bins=16)
        lsv_hist = lsv_hist / (np.sum(lsv_hist) + 1e-10)
        features[f'{prefix}LSV_entropy'] = -np.sum(lsv_hist * np.log(lsv_hist + 1e-10))

        if len(vertical_profile) > 2:
            x = np.arange(len(vertical_profile))
            features[f'{prefix}prof_slope'] = np.polyfit(x, vertical_profile, 1)[0]
            features[f'{prefix}prof_rough'] = np.std(np.diff(vertical_profile))

            if len(vertical_profile) > 3:
                second_diff = np.diff(vertical_profile, 2)
                features[f'{prefix}prof_curv'] = np.mean(np.abs(second_diff))
            else:
                features[f'{prefix}prof_curv'] = 0

            prof_fft = np.fft.fft(vertical_profile)
            prof_mag = np.abs(prof_fft)
            features[f'{prefix}prof_freq'] = np.argmax(prof_mag[1:len(prof_mag) // 2]) + 1
            features[f'{prefix}prof_ratio'] = np.max(vertical_profile) / (np.min(vertical_profile) + 1e-10)
        else:
            for feat in ['prof_slope', 'prof_rough', 'prof_curv', 'prof_freq']:
                features[f'{prefix}{feat}'] = 0
            features[f'{prefix}prof_ratio'] = 1

        log_mag_full = np.log(magnitude_spectrum.flatten() + 1e-10)
        features[f'{prefix}spectral_slope_idx'] = np.polyfit(np.arange(len(log_mag_full)), log_mag_full, 1)[0]
        features[f'{prefix}spectral_curv_idx'] = np.std(np.diff(log_mag_full))

        hot_threshold = np.percentile(data, 90)
        cold_threshold = np.percentile(data, 10)
        hot_mean = np.mean(data[data > hot_threshold])
        cold_mean = np.mean(data[data < cold_threshold])
        features[f'{prefix}thermal_ratio_index'] = hot_mean / (cold_mean + 1e-10)

        hotspot_mask = data > hot_threshold
        if np.any(hotspot_mask):
            hotspot_data_temp = data.copy()
            hotspot_data_temp[~hotspot_mask] = 0
            hotspot_fft = np.abs(fft2(hotspot_data_temp))
            features[f'{prefix}HSI'] = np.sum(hotspot_fft) / (np.sum(magnitude_spectrum) + 1e-10)
        else:
            features[f'{prefix}HSI'] = 0

        h, w = data.shape
        regions_list = [
            data[:h // 2, :w // 2],
            data[:h // 2, w // 2:],
            data[h // 2:, :w // 2],
            data[h // 2:, w // 2:]
        ]

        region_spectra = []
        for region in regions_list:
            if region.size > 0:
                region_fft = np.abs(fft2(region))
                region_spectra.append(np.sum(region_fft))

        features[f'{prefix}MRSI'] = np.std(region_spectra) if len(region_spectra) > 0 else 0
        features[f'{prefix}BRF'] = features.get(f'{prefix}base_grad', 0) * features.get(f'{prefix}thermal_ratio_index',
                                                                                        1)
        features[f'{prefix}SCD'] = features.get(f'{prefix}spectral_curv_idx', 0) * features.get(f'{prefix}radial_decay',
                                                                                                0)

        return features

    def extract_70d_features(self, data, prefix=""):
        """
        Extract the legacy basic/statistical descriptor set.

        Parameters:
            data: NumPy array containing the ROI temperature matrix.
            prefix: String prefix used for feature names.

        Returns:
            Dictionary containing the descriptors returned by this legacy function (33 in the study version).
        """
        features = {}

        # Basic statistical descriptors.
        features[f'{prefix}mean_temp'] = np.mean(data)
        features[f'{prefix}std_temp'] = np.std(data)
        features[f'{prefix}max_temp'] = np.max(data)

        nonzero_data = data[data != 0]
        if len(nonzero_data) > 0:
            features[f'{prefix}nonzero_median'] = np.median(nonzero_data)
            features[f'{prefix}skewness_nonzero'] = skew(nonzero_data)
            features[f'{prefix}kurtosis_nonzero'] = kurtosis(nonzero_data)
            features[f'{prefix}nonzero_mean'] = np.mean(nonzero_data)
            features[f'{prefix}nonzero_std'] = np.std(nonzero_data)
            features[f'{prefix}cv_nonzero'] = features[f'{prefix}nonzero_std'] / (
                    features[f'{prefix}nonzero_mean'] + 1e-10)
        else:
            for feat in ['nonzero_median', 'skewness_nonzero', 'kurtosis_nonzero', 'nonzero_mean', 'nonzero_std',
                         'cv_nonzero']:
                features[f'{prefix}{feat}'] = 0

        features[f'{prefix}nonzero_count'] = len(nonzero_data)
        features[f'{prefix}nonzero_fraction'] = len(nonzero_data) / data.size

        if len(nonzero_data) > 0:
            nonzero_positions = np.where(data != 0)
            if len(nonzero_positions[0]) > 0:
                min_row, max_row = np.min(nonzero_positions[0]), np.max(nonzero_positions[0])
                min_col, max_col = np.min(nonzero_positions[1]), np.max(nonzero_positions[1])
                features[f'{prefix}bbox_width'] = (max_col - min_col + 1) / data.shape[1]
                features[f'{prefix}bbox_height'] = (max_row - min_row + 1) / data.shape[0]
                features[f'{prefix}centroid_x'] = np.mean(nonzero_positions[1]) / data.shape[1]
                features[f'{prefix}centroid_y'] = np.mean(nonzero_positions[0]) / data.shape[0]
            else:
                for feat in ['bbox_width', 'bbox_height']:
                    features[f'{prefix}{feat}'] = 0
                features[f'{prefix}centroid_x'] = 0.5
                features[f'{prefix}centroid_y'] = 0.5
        else:
            for feat in ['bbox_width', 'bbox_height']:
                features[f'{prefix}{feat}'] = 0
            features[f'{prefix}centroid_x'] = 0.5
            features[f'{prefix}centroid_y'] = 0.5

        features[f'{prefix}temp_25percentile'] = np.percentile(data, 25)
        features[f'{prefix}temp_75percentile'] = np.percentile(data, 75)

        hist, _ = np.histogram(data, bins=32)
        hist = hist / (np.sum(hist) + 1e-10)
        features[f'{prefix}temp_entropy'] = -np.sum(hist * np.log(hist + 1e-10))

        max_val = features[f'{prefix}max_temp']
        if max_val > 0:
            features[f'{prefix}above_50pct_max'] = np.sum(data >= 0.5 * max_val) / data.size
            features[f'{prefix}above_80pct_max'] = np.sum(data >= 0.8 * max_val) / data.size
        else:
            features[f'{prefix}above_50pct_max'] = 0
            features[f'{prefix}above_80pct_max'] = 0

        features[f'{prefix}energy'] = np.sum(data ** 2)

        # GLCM texture descriptors.
        data_normalized = ((data - data.min()) / (data.max() - data.min() + 1e-10) * 255).astype(np.uint8)
        try:
            glcm_temp = graycomatrix(data_normalized, distances=[1], angles=[0], levels=256, symmetric=True,
                                     normed=True)
            features[f'{prefix}temp_glcm_contrast'] = graycoprops(glcm_temp, 'contrast')[0, 0]
            features[f'{prefix}temp_glcm_energy'] = graycoprops(glcm_temp, 'energy')[0, 0]
            features[f'{prefix}temp_glcm_homogeneity'] = graycoprops(glcm_temp, 'homogeneity')[0, 0]
            features[f'{prefix}temp_glcm_correlation'] = graycoprops(glcm_temp, 'correlation')[0, 0]
        except:
            for feat in ['temp_glcm_contrast', 'temp_glcm_energy', 'temp_glcm_homogeneity', 'temp_glcm_correlation']:
                features[f'{prefix}{feat}'] = 0

        # Hot-region descriptors.
        hot_threshold = np.percentile(data, 90)
        hot_regions = data > hot_threshold
        labeled_regions = sk_label(hot_regions)
        regions_props = regionprops(labeled_regions)

        if len(regions_props) > 0:
            features[f'{prefix}hot_region_count'] = len(regions_props)
            features[f'{prefix}hot_region_max_area'] = max([r.area for r in regions_props])
            features[f'{prefix}hot_region_total_area'] = sum([r.area for r in regions_props])
            features[f'{prefix}hot_region_max_perimeter'] = max([r.perimeter for r in regions_props])
        else:
            for feat in ['hot_region_count', 'hot_region_max_area', 'hot_region_total_area',
                         'hot_region_max_perimeter']:
                features[f'{prefix}{feat}'] = 0

        features[f'{prefix}T_area'] = np.mean(data)
        center_line = data[:, data.shape[1] // 2]
        features[f'{prefix}T_line_mean'] = np.mean(center_line)
        features[f'{prefix}T_line_auc'] = np.trapz(center_line)
        features[f'{prefix}PDCs_score'] = features[f'{prefix}temp_glcm_contrast'] * features[f'{prefix}temp_entropy']

        return features


def infer_roi_mask_from_zero_background(data: np.ndarray) -> np.ndarray:
    """
    Infer the main ROI mask from a matrix in which pixels outside the ROI are fixed at 0.
    """
    nonzero = np.abs(data) > CMVT_EPS

    if int(np.sum(nonzero)) < 20:
        return nonzero.astype(bool)

    labels = sk_label(nonzero, connectivity=1)
    label_ids, counts = np.unique(labels[labels > 0], return_counts=True)

    if len(label_ids) == 0:
        return nonzero.astype(bool)

    largest_label = int(label_ids[np.argmax(counts)])
    mask = labels == largest_label
    mask = binary_fill_holes(mask)

    return mask.astype(bool)


def _valid_values(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = np.asarray(data[mask], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return vals


def _robust_mad(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _masked_gaussian_background(
    data: np.ndarray,
    mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Mask-aware Gaussian background estimation:
      Gaussian(data * mask) / Gaussian(mask)

    This prevents artificial zero-valued background pixels outside the ROI from contributing as low-temperature values during convolution.
    """
    m = mask.astype(np.float64)
    numerator = gaussian_filter(
        np.where(mask, data, 0.0),
        sigma=float(sigma),
        mode="nearest",
    )
    denominator = gaussian_filter(
        m,
        sigma=float(sigma),
        mode="nearest",
    )

    background = np.zeros_like(data, dtype=np.float64)
    valid = denominator > 1e-8
    background[valid] = numerator[valid] / denominator[valid]

    return background


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels = sk_label(mask.astype(bool), connectivity=1)
    ids, counts = np.unique(labels[labels > 0], return_counts=True)

    if len(ids) == 0:
        return np.zeros_like(mask, dtype=bool)

    largest = int(ids[np.argmax(counts)])
    return labels == largest


def _moran_geary_4neighbor(
    data: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    """
    Compute Global Moran's I and Geary's C using 4-neighbor adjacency.
    Only adjacent pixels within the ROI are included; zero-valued pixels outside the ROI do not participate.
    """
    vals = _valid_values(data, mask)

    if len(vals) < 3:
        return 0.0, 0.0

    mean_val = float(np.mean(vals))
    z = np.zeros_like(data, dtype=np.float64)
    z[mask] = data[mask] - mean_val

    denom = float(np.sum(z[mask] ** 2))
    if denom <= CMVT_EPS:
        return 0.0, 0.0

    hmask = mask[:, :-1] & mask[:, 1:]
    vmask = mask[:-1, :] & mask[1:, :]

    n_edges = int(np.sum(hmask) + np.sum(vmask))
    if n_edges <= 0:
        return 0.0, 0.0

    prod_sum = float(
        np.sum(z[:, :-1][hmask] * z[:, 1:][hmask])
        + np.sum(z[:-1, :][vmask] * z[1:, :][vmask])
    )

    diff2_sum = float(
        np.sum((z[:, :-1][hmask] - z[:, 1:][hmask]) ** 2)
        + np.sum((z[:-1, :][vmask] - z[1:, :][vmask]) ** 2)
    )

    n = int(np.sum(mask))

    moran_i = (n / n_edges) * (prod_sum / denom)
    geary_c = ((n - 1) / (2.0 * n_edges)) * (diff2_sum / denom)

    return float(moran_i), float(geary_c)


def _hotspot_persistence_auc(
    data: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Multi-threshold hotspot persistence:
    At the P80, P85, P90, and P95 thresholds within the ROI, compute the area fraction of the largest connected hotspot.
    Then compute the normalized AUC of the threshold-versus-largest-hotspot-area-fraction curve.
    This descriptor characterizes hotspot persistence across increasingly stringent thresholds.
    """
    vals = _valid_values(data, mask)
    if len(vals) < 20:
        return 0.0

    qs = np.array([0.80, 0.85, 0.90, 0.95], dtype=np.float64)
    area_fracs = []

    n_valid = max(int(np.sum(mask)), 1)

    for q in qs:
        thr = float(np.quantile(vals, q))
        hot = mask & (data >= thr)
        largest = _largest_component(hot)
        area_fracs.append(float(np.sum(largest)) / n_valid)

    area_fracs = np.asarray(area_fracs, dtype=np.float64)

    auc = float(
        np.trapz(area_fracs, qs)
        / (qs[-1] - qs[0] + CMVT_EPS)
    )

    return auc


def extract_cmvt_upper_roi_features(
    data: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    """
    Add 23 CMVT-oriented candidate features for each upper/calf ROI.

    7  robust within-ROI distribution features
    6  multiscale local thermal-contrast features
    4  hotspot-ring CNR
    5  spatial-autocorrelation / thermal-topography features
    1  distal-to-proximal longitudinal thermal-gradient feature
    """
    features: Dict[str, float] = {}

    mask = infer_roi_mask_from_zero_background(data)
    vals = _valid_values(data, mask)

    # -------------------------------------------------------------------------
    # A. Robust within-ROI temperature-tail descriptors: 7 dimensions.
    # These do not duplicate the legacy nonzero_mean/std/median/max/skew/kurtosis/cv descriptors.
    # -------------------------------------------------------------------------
    if len(vals) >= 10:
        p05, p10, p25, p50, p75, p90, p95 = np.percentile(
            vals,
            [5, 10, 25, 50, 75, 90, 95],
        )

        n_top = max(1, int(np.ceil(0.10 * len(vals))))
        top10_mean = float(np.mean(np.partition(vals, -n_top)[-n_top:]))

        features[f"{prefix}valid_p05"] = float(p05)
        features[f"{prefix}valid_p10"] = float(p10)
        features[f"{prefix}valid_p90"] = float(p90)
        features[f"{prefix}valid_p95"] = float(p95)
        features[f"{prefix}valid_iqr"] = float(p75 - p25)
        features[f"{prefix}valid_mad"] = _robust_mad(vals)
        features[f"{prefix}valid_top10_mean"] = top10_mean
    else:
        for name in [
            "valid_p05",
            "valid_p10",
            "valid_p90",
            "valid_p95",
            "valid_iqr",
            "valid_mad",
            "valid_top10_mean",
        ]:
            features[f"{prefix}{name}"] = 0.0

    # -------------------------------------------------------------------------
    # B. Mask-aware multiscale local thermal contrast: 6 dimensions.
    # -------------------------------------------------------------------------
    for scale_name, sigma in [
        ("s3", 3.0),
        ("s7", 7.0),
        ("s15", 15.0),
    ]:
        if int(np.sum(mask)) >= 20:
            bg = _masked_gaussian_background(data, mask, sigma)
            residual = data[mask] - bg[mask]
            residual = residual[np.isfinite(residual)]

            if len(residual) > 0:
                n_top = max(1, int(np.ceil(0.01 * len(residual))))
                top1 = float(
                    np.mean(
                        np.partition(residual, -n_top)[-n_top:]
                    )
                )
                p95_res = float(np.percentile(residual, 95))
            else:
                top1 = 0.0
                p95_res = 0.0
        else:
            top1 = 0.0
            p95_res = 0.0

        features[f"{prefix}localcontrast_{scale_name}_p95"] = p95_res
        features[f"{prefix}localcontrast_{scale_name}_top1mean"] = top1

    # -------------------------------------------------------------------------
    # C. Hotspot-to-ring contrast/CNR: 4 dimensions.
    # -------------------------------------------------------------------------
    if len(vals) >= 20:
        hot_thr = float(np.percentile(vals, 90))
        hot_mask = mask & (data >= hot_thr)
        largest_hot = _largest_component(hot_mask)

        if np.any(largest_hot):
            ring = (
                binary_dilation(
                    largest_hot,
                    iterations=8,
                )
                & mask
                & (~largest_hot)
            )

            if int(np.sum(ring)) < 10:
                ring = mask & (~largest_hot)

            hot_vals = _valid_values(data, largest_hot)
            ring_vals = _valid_values(data, ring)

            if len(hot_vals) > 0 and len(ring_vals) > 1:
                delta_t = float(
                    np.mean(hot_vals) - np.mean(ring_vals)
                )
                cnr = delta_t / (
                    float(np.std(ring_vals)) + CMVT_EPS
                )

                ring_mad_sigma = (
                    1.4826 * _robust_mad(ring_vals)
                )
                robust_cnr = (
                    float(np.median(hot_vals) - np.median(ring_vals))
                    / (ring_mad_sigma + CMVT_EPS)
                )

                area_fraction = float(
                    np.sum(largest_hot)
                    / max(np.sum(mask), 1)
                )
            else:
                delta_t = 0.0
                cnr = 0.0
                robust_cnr = 0.0
                area_fraction = 0.0
        else:
            delta_t = 0.0
            cnr = 0.0
            robust_cnr = 0.0
            area_fraction = 0.0
    else:
        delta_t = 0.0
        cnr = 0.0
        robust_cnr = 0.0
        area_fraction = 0.0

    features[f"{prefix}hotring_deltaT"] = delta_t
    features[f"{prefix}hotring_cnr"] = float(cnr)
    features[f"{prefix}hotring_robust_cnr"] = float(robust_cnr)
    features[f"{prefix}hotring_area_fraction"] = area_fraction

    # -------------------------------------------------------------------------
    # D. Spatial autocorrelation and thermal-topography descriptors: 5 dimensions.
    # -------------------------------------------------------------------------
    moran_i, geary_c = _moran_geary_4neighbor(data, mask)
    features[f"{prefix}moran_I_4n"] = moran_i
    features[f"{prefix}geary_C_4n"] = geary_c

    if int(np.sum(mask)) >= 20:
        smooth = _masked_gaussian_background(
            data,
            mask,
            sigma=1.5,
        )

        # Use an eroded interior region when possible to reduce mask-edge artifacts in gradient/curvature measures.
        interior = binary_erosion(mask)
        if int(np.sum(interior)) < 20:
            interior = mask

        gy, gx = np.gradient(smooth)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        dxx = np.gradient(gx, axis=1)
        dyy = np.gradient(gy, axis=0)
        lap = dxx + dyy

        slope_p95 = float(
            np.percentile(grad_mag[interior], 95)
        )
        lap_abs_p95 = float(
            np.percentile(np.abs(lap[interior]), 95)
        )
    else:
        slope_p95 = 0.0
        lap_abs_p95 = 0.0

    features[f"{prefix}terrain_slope_p95"] = slope_p95
    features[f"{prefix}terrain_laplacian_abs_p95"] = lap_abs_p95
    features[f"{prefix}hotspot_persistence_auc"] = (
        _hotspot_persistence_auc(data, mask)
    )

    # -------------------------------------------------------------------------
    # E. Within-calf distal-to-proximal longitudinal gradient: 1 dimension.
    # In image coordinates, smaller y corresponds to the proximal direction and larger y to the distal direction.
    # Compute the difference between the median temperatures of the uppermost 20% and lowermost 20% of the upper/calf ROI.
    # The upper portion is closer to the knee (proximal), whereas the lower portion is closer to the ankle (distal).
    # -------------------------------------------------------------------------
    if int(np.sum(mask)) >= 20:
        ys, xs = np.where(mask)
        y20 = float(np.percentile(ys, 20))
        y80 = float(np.percentile(ys, 80))

        proximal = mask.copy()
        distal = mask.copy()

        row_index = np.arange(data.shape[0])[:, None]
        proximal &= row_index <= y20
        distal &= row_index >= y80

        prox_vals = _valid_values(data, proximal)
        dist_vals = _valid_values(data, distal)

        if len(prox_vals) > 0 and len(dist_vals) > 0:
            distal_proximal = float(
                np.median(dist_vals) - np.median(prox_vals)
            )
        else:
            distal_proximal = 0.0
    else:
        distal_proximal = 0.0

    features[f"{prefix}distal_minus_proximal_median"] = (
        distal_proximal
    )

    if len(features) != EXPECTED_CMVTLITE_PER_UPPER_ROI:
        raise AssertionError(
            f"{prefix} CMVT upper/calf feature count={len(features)}, "
            f"expected={EXPECTED_CMVTLITE_PER_UPPER_ROI}"
        )

    return features


def _roi_summary_for_cross(
    data: np.ndarray,
) -> Dict[str, object]:
    mask = infer_roi_mask_from_zero_background(data)
    vals = _valid_values(data, mask)

    if len(vals) == 0:
        return {
            "values": np.array([0.0], dtype=np.float64),
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }

    return {
        "values": vals,
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
    }


def extract_cmvt_cross_roi_features(
    roi_data: Dict[tuple, np.ndarray],
) -> Dict[str, float]:
    """
    Add 24 patient-level cross-ROI features using the anatomically corrected ROI convention.

    10 dimensions: five left-right upper/calf distribution differences for each of the Front and Back views.
    8 dimensions: four upper-lower within-participant reference pairs using the median and P90.
    4 dimensions: front-back relationships for left and right upper/calf ROIs using the median and P90.
    2 dimensions: bilateral upper/calf P95 thermal-load summaries for the Front and Back views.

    Notes:
    - upper = calf, the primary anatomical region of interest for the CMVT-oriented descriptors.
    - lower = ankle-foot, used only as an auxiliary within-participant reference region.
    - No class labels are used and no across-participant fitting is required.
    """
    features: Dict[str, float] = {}

    summaries = {
        key: _roi_summary_for_cross(data)
        for key, data in roi_data.items()
    }

    # -------------------------------------------------------------------------
    # 1) Left-right upper/calf distribution differences: 2 views × 5 = 10 dimensions.
    # -------------------------------------------------------------------------
    for view, view_name in [(1, "front"), (2, "back")]:
        left = summaries[(view, "left_upper")]
        right = summaries[(view, "right_upper")]

        features[f"cmvt_cross_{view_name}_upper_lr_median_signed"] = (
            float(left["median"]) - float(right["median"])
        )
        features[f"cmvt_cross_{view_name}_upper_lr_p90_signed"] = (
            float(left["p90"]) - float(right["p90"])
        )
        features[f"cmvt_cross_{view_name}_upper_lr_p95_signed"] = (
            float(left["p95"]) - float(right["p95"])
        )

        lv = np.asarray(left["values"], dtype=np.float64)
        rv = np.asarray(right["values"], dtype=np.float64)

        try:
            wd = float(wasserstein_distance(lv, rv))
        except Exception:
            wd = 0.0

        try:
            ks = float(
                ks_2samp(
                    lv,
                    rv,
                    alternative="two-sided",
                    method="asymp",
                ).statistic
            )
        except Exception:
            ks = 0.0

        features[f"cmvt_cross_{view_name}_upper_lr_wasserstein"] = wd
        features[f"cmvt_cross_{view_name}_upper_lr_ks_stat"] = ks

    # -------------------------------------------------------------------------
    # 2) Upper/calf versus lower/ankle-foot within-participant reference:
    #    2 views × 2 sides × 2 statistics = 8 dimensions.
    # Preserve the signed direction: upper - lower.
    # -------------------------------------------------------------------------
    for view, view_name in [(1, "front"), (2, "back")]:
        for side in ["left", "right"]:
            upper = summaries[(view, f"{side}_upper")]
            lower = summaries[(view, f"{side}_lower")]

            features[
                f"cmvt_cross_{view_name}_{side}_upper_minus_lower_median"
            ] = float(upper["median"]) - float(lower["median"])

            features[
                f"cmvt_cross_{view_name}_{side}_upper_minus_lower_p90"
            ] = float(upper["p90"]) - float(lower["p90"])

    # -------------------------------------------------------------------------
    # 3) Front-Back upper/calf relationships: 2 sides × 2 statistics = 4 dimensions.
    # -------------------------------------------------------------------------
    for side in ["left", "right"]:
        front = summaries[(1, f"{side}_upper")]
        back = summaries[(2, f"{side}_upper")]

        features[
            f"cmvt_cross_{side}_upper_front_minus_back_median"
        ] = float(front["median"]) - float(back["median"])

        features[
            f"cmvt_cross_{side}_upper_front_minus_back_p90"
        ] = float(front["p90"]) - float(back["p90"])

    # -------------------------------------------------------------------------
    # 4) Bilateral upper/calf global thermal load: 2 dimensions.
    # -------------------------------------------------------------------------
    features["cmvt_cross_front_bilateral_upper_p95_mean"] = float(
        (
            summaries[(1, "left_upper")]["p95"]
            + summaries[(1, "right_upper")]["p95"]
        )
        / 2.0
    )

    features["cmvt_cross_back_bilateral_upper_p95_mean"] = float(
        (
            summaries[(2, "left_upper")]["p95"]
            + summaries[(2, "right_upper")]["p95"]
        )
        / 2.0
    )

    if len(features) != EXPECTED_CMVTLITE_CROSS:
        raise AssertionError(
            f"CMVT cross-ROI feature count={len(features)}, "
            f"expected={EXPECTED_CMVTLITE_CROSS}"
        )

    return features


def _relv2_require(
    features: Dict[str, float],
    key: str,
) -> float:
    """
    Missing fields must not be silently replaced by 0 because doing so could conceal feature-naming or structural errors.
    """
    if key not in features:
        raise KeyError(
            f"Required base feature for Relational v2 is missing: {key}"
        )

    value = float(features[key])

    if not np.isfinite(value):
        return 0.0

    return value


def _relv2_norm_abs_diff(
    a: float,
    b: float,
) -> float:
    """
    Symmetric normalized difference:
      |a-b| / (|a|+|b|+eps)
    """
    return float(
        abs(a - b)
        / (
            abs(a)
            + abs(b)
            + RELV2_EPS
        )
    )


def extract_relational_v2_features(
    base1468_features: Dict[str, float],
) -> Dict[str, float]:
    """
    Add 144 patient-level relational candidate features.

    A. General relationships: 10 descriptors × 12 = 120
       1) Front upper/calf left-right signed difference
       2) Front upper/calf left-right absolute difference
       3) Back upper/calf left-right signed difference
       4) Back upper/calf left-right absolute difference
       5) Participant-level upper/calf left-right signed difference (Front/Back first averaged within each side)
       6) Participant-level upper/calf normalized absolute left-right asymmetry
       7) Left upper/calf - lower/ankle-foot signed difference (after Front/Back averaging)
       8) left  upper/calf - lower/ankle-foot absolute
       9) right upper/calf - lower/ankle-foot signed
      10) right upper/calf - lower/ankle-foot absolute
      11) left  upper/calf front-back absolute inconsistency
      12) right upper/calf front-back absolute inconsistency

    B. Participant-level bilateral/unilateral relationships: 6 CMVT upper/calf descriptors × 4 = 24
       1) bilateral_mean = (L+R)/2
       2) bilateral_min  = min(L,R)
       3) unilateral_abs = |L-R|
       4) lr_norm_asym   = |L-R|/(|L|+|R|+eps)

    - Variables such as bilateral_min and bilateral_mean are engineered candidate descriptors.
    """
    out: Dict[str, float] = {}

    # -------------------------------------------------------------------------
    # A. General relationships: 10 × 12 = 120 dimensions (upper/calf-centered).
    # -------------------------------------------------------------------------
    for desc in RELV2_GENERAL_DESCRIPTORS:
        # primary anatomical region: upper = calf
        ful = _relv2_require(
            base1468_features,
            f"front_left_upper_{desc}",
        )
        fur = _relv2_require(
            base1468_features,
            f"front_right_upper_{desc}",
        )
        bul = _relv2_require(
            base1468_features,
            f"back_left_upper_{desc}",
        )
        bur = _relv2_require(
            base1468_features,
            f"back_right_upper_{desc}",
        )

        # distal reference region: lower = ankle-foot
        fll = _relv2_require(
            base1468_features,
            f"front_left_lower_{desc}",
        )
        flr = _relv2_require(
            base1468_features,
            f"front_right_lower_{desc}",
        )
        bll = _relv2_require(
            base1468_features,
            f"back_left_lower_{desc}",
        )
        blr = _relv2_require(
            base1468_features,
            f"back_right_lower_{desc}",
        )

        left_calf_patient = 0.5 * (ful + bul)
        right_calf_patient = 0.5 * (fur + bur)

        left_distal_patient = 0.5 * (fll + bll)
        right_distal_patient = 0.5 * (flr + blr)

        # A1. Left-right upper/calf relationships: Front/Back plus participant level, 6 dimensions per descriptor.
        out[
            f"relv2_{desc}_front_upper_lr_signed"
        ] = float(ful - fur)

        out[
            f"relv2_{desc}_front_upper_lr_abs"
        ] = float(abs(ful - fur))

        out[
            f"relv2_{desc}_back_upper_lr_signed"
        ] = float(bul - bur)

        out[
            f"relv2_{desc}_back_upper_lr_abs"
        ] = float(abs(bul - bur))

        out[
            f"relv2_{desc}_patient_upper_lr_signed"
        ] = float(
            left_calf_patient
            - right_calf_patient
        )

        out[
            f"relv2_{desc}_patient_upper_lr_norm_abs"
        ] = _relv2_norm_abs_diff(
            left_calf_patient,
            right_calf_patient,
        )

        # A2. Same-leg upper/calf versus lower/ankle-foot relationships: 4 dimensions per descriptor.
        # Signed direction is consistently defined as primary (calf) - distal (ankle-foot).
        out[
            f"relv2_{desc}_left_upper_minus_lower_signed"
        ] = float(
            left_calf_patient
            - left_distal_patient
        )

        out[
            f"relv2_{desc}_left_upper_lower_abs"
        ] = float(
            abs(
                left_calf_patient
                - left_distal_patient
            )
        )

        out[
            f"relv2_{desc}_right_upper_minus_lower_signed"
        ] = float(
            right_calf_patient
            - right_distal_patient
        )

        out[
            f"relv2_{desc}_right_upper_lower_abs"
        ] = float(
            abs(
                right_calf_patient
                - right_distal_patient
            )
        )

        # A3. Same-leg Front versus Back upper/calf consistency: 2 dimensions per descriptor.
        # Front and Back views represent different skin surfaces; therefore only descriptor values are compared, not pixelwise overlap.
        out[
            f"relv2_{desc}_left_upper_front_back_abs"
        ] = float(abs(ful - bul))

        out[
            f"relv2_{desc}_right_upper_front_back_abs"
        ] = float(abs(fur - bur))

    if len(out) != EXPECTED_RELV2_GENERAL:
        raise AssertionError(
            f"Relational v2 general-relationship feature count={len(out)}, "
            f"expected={EXPECTED_RELV2_GENERAL}"
        )

    # -------------------------------------------------------------------------
    # B. 6 × 4 = 24 dimensions: participant-level bilateral common state / unilateral dominance.
    # Use only the anatomically corrected CMVT-oriented descriptors from the four upper/calf ROIs.
    # -------------------------------------------------------------------------
    before_bilateral = len(out)

    for desc in RELV2_BILATERAL_CMVTDESCRIPTORS:
        ful = _relv2_require(
            base1468_features,
            f"cmvt_front_left_upper_{desc}",
        )
        fur = _relv2_require(
            base1468_features,
            f"cmvt_front_right_upper_{desc}",
        )
        bul = _relv2_require(
            base1468_features,
            f"cmvt_back_left_upper_{desc}",
        )
        bur = _relv2_require(
            base1468_features,
            f"cmvt_back_right_upper_{desc}",
        )

        # First combine Front and Back values within the same side to form a leg-level calf state.
        # This averaging is performed only within the current participant and uses no statistics from other participants.
        left_leg = 0.5 * (ful + bul)
        right_leg = 0.5 * (fur + bur)

        out[
            f"relv2_cmvt_{desc}_bilateral_mean"
        ] = float(
            0.5 * (
                left_leg
                + right_leg
            )
        )

        out[
            f"relv2_cmvt_{desc}_bilateral_min"
        ] = float(
            min(
                left_leg,
                right_leg,
            )
        )

        out[
            f"relv2_cmvt_{desc}_unilateral_abs"
        ] = float(
            abs(
                left_leg
                - right_leg
            )
        )

        out[
            f"relv2_cmvt_{desc}_lr_norm_asym"
        ] = _relv2_norm_abs_diff(
            left_leg,
            right_leg,
        )

    bilateral_added = (
        len(out)
        - before_bilateral
    )

    if bilateral_added != EXPECTED_RELV2_BILATERAL:
        raise AssertionError(
            f"Relational v2 bilateral/unilateral feature count={bilateral_added}, "
            f"expected={EXPECTED_RELV2_BILATERAL}"
        )

    if len(out) != EXPECTED_RELV2_TOTAL:
        raise AssertionError(
            f"Relational v2 total feature count={len(out)}, "
            f"expected={EXPECTED_RELV2_TOTAL}"
        )

    for key, value in out.items():
        if not np.isfinite(float(value)):
            raise ValueError(
                f"Relational v2 produced NaN/Inf: {key}={value}"
            )

    return out

# Canonical public input order.
ROI_KEYS = (
    "front_left_lower",
    "front_left_upper",
    "front_right_lower",
    "front_right_upper",
    "back_left_lower",
    "back_left_upper",
    "back_right_lower",
    "back_right_upper",
)


def extract_features_1612(roi_matrices: Dict[str, np.ndarray]) -> Dict[str, float]:

    missing = [key for key in ROI_KEYS if key not in roi_matrices]
    extra = [key for key in roi_matrices if key not in ROI_KEYS]
    if missing or extra:
        raise ValueError(f"Expected exactly the eight ROI keys. Missing={missing}, extra={extra}")

    arrays: Dict[str, np.ndarray] = {}
    for key in ROI_KEYS:
        arr = np.asarray(roi_matrices[key], dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"{key} must be a two-dimensional matrix; got shape={arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{key} contains NaN or Inf")
        arrays[key] = arr

    extractor = ThermalFeatureExtractor()
    all_features: Dict[str, float] = {}
    roi_data_lookup: Dict[tuple, np.ndarray] = {}

    ordered = [
        (1, "left_lower", "front_left_lower"),
        (1, "left_upper", "front_left_upper"),
        (1, "right_lower", "front_right_lower"),
        (1, "right_upper", "front_right_upper"),
        (2, "left_lower", "back_left_lower"),
        (2, "left_upper", "back_left_upper"),
        (2, "right_lower", "back_right_lower"),
        (2, "right_upper", "back_right_upper"),
    ]

    for view, roi_name, public_key in ordered:
        data = arrays[public_key]
        roi_data_lookup[(view, roi_name)] = data
        view_name = "front" if view == 1 else "back"
        prefix = f"{view_name}_{roi_name}_"

        features_136 = extractor.extract_136d_features(data, prefix)
        features_70 = extractor.extract_70d_features(data, prefix)

        if len(features_136) != EXPECTED_136_COUNT:
            raise AssertionError(
                f"{public_key}: extract_136d_features returned {len(features_136)}; "
                f"expected {EXPECTED_136_COUNT}"
            )
        if len(features_70) != EXPECTED_70_FUNCTION_ACTUAL_COUNT:
            raise AssertionError(
                f"{public_key}: extract_70d_features returned {len(features_70)}; "
                f"expected {EXPECTED_70_FUNCTION_ACTUAL_COUNT}"
            )
        if set(features_136) & set(features_70):
            raise AssertionError(f"Duplicate ROI-level feature names in {public_key}")

        all_features.update(features_136)
        all_features.update(features_70)

    if len(all_features) != EXPECTED_LEGACY_TOTAL_FEATURES:
        raise AssertionError(
            f"ROI-level feature count={len(all_features)}; expected {EXPECTED_LEGACY_TOTAL_FEATURES}"
        )

    cmvt_features: Dict[str, float] = {}
    for view, view_name in [(1, "front"), (2, "back")]:
        for side in ["left", "right"]:
            key = (view, f"{side}_{CMVT_PRIMARY_PART}")
            prefix = f"cmvt_{view_name}_{side}_{CMVT_PRIMARY_PART}_"
            cmvt_features.update(extract_cmvt_upper_roi_features(roi_data_lookup[key], prefix))

    if len(cmvt_features) != EXPECTED_CMVTLITE_UPPER_TOTAL:
        raise AssertionError(
            f"Calf-centered CMVT feature count={len(cmvt_features)}; "
            f"expected {EXPECTED_CMVTLITE_UPPER_TOTAL}"
        )

    cross_features = extract_cmvt_cross_roi_features(roi_data_lookup)
    if set(cmvt_features) & set(cross_features):
        raise AssertionError("Duplicate names between calf-centered and cross-ROI CMVT features")
    cmvt_features.update(cross_features)

    if len(cmvt_features) != EXPECTED_CMVTLITE_TOTAL:
        raise AssertionError(
            f"CMVT-oriented feature count={len(cmvt_features)}; expected {EXPECTED_CMVTLITE_TOTAL}"
        )
    if set(all_features) & set(cmvt_features):
        raise AssertionError("CMVT-oriented feature names overlap ROI-level feature names")

    all_features.update(cmvt_features)
    if len(all_features) != EXPECTED_BASE_TOTAL_FEATURES:
        raise AssertionError(
            f"Base feature count={len(all_features)}; expected {EXPECTED_BASE_TOTAL_FEATURES}"
        )

    relational = extract_relational_v2_features(all_features)
    if set(all_features) & set(relational):
        raise AssertionError("Relational feature names overlap the first 1,468 feature names")
    all_features.update(relational)

    if len(all_features) != EXPECTED_TOTAL_FEATURES:
        raise AssertionError(
            f"Final feature count={len(all_features)}; expected {EXPECTED_TOTAL_FEATURES}"
        )

    cleaned: Dict[str, float] = {}
    for key, value in all_features.items():
        try:
            numeric = float(value)
        except Exception:
            numeric = 0.0
        if not np.isfinite(numeric):
            numeric = 0.0
        cleaned[key] = numeric

    return cleaned


def extract_features_1468(roi_matrices: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Return the 1,468-dimensional representation used for CMVT presence detection.

    This is the first 1,468 features of the complete library: 1,352 ROI-level
    features plus 116 CMVT-oriented features.
    """
    full = extract_features_1612(roi_matrices)
    return dict(list(full.items())[:EXPECTED_BASE_TOTAL_FEATURES])

