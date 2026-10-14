# Mathematical Definitions of the 1,612 Handcrafted Thermal Features

This document defines the feature calculations used in the study. It should be read together with [`feature_dictionary_1612.csv`](../feature_dictionary_1612.csv), which maps every feature to a formula group through the `formula_id` field, and [`src/feature_extraction_1612.py`](../src/feature_extraction_1612.py), which contains the corresponding implementation.
The equations and parameter settings below follow the implementation used for the reported analysis.

## 1. Feature architecture and notation

Each participant is represented by eight normalized ROI temperature matrices:
- front left calf (`front_left_upper`)
- front right calf (`front_right_upper`)
- front left ankle-foot (`front_left_lower`)
- front right ankle-foot (`front_right_lower`)
- back left calf (`back_left_upper`)
- back right calf (`back_right_upper`)
- back left ankle-foot (`back_left_lower`)
- back right ankle-foot (`back_right_lower`)

The complete feature library contains:
- 1,352 ROI-level features: 169 descriptors × 8 ROIs
- 116 CMVT-oriented features: 92 calf-centered descriptors + 24 cross-ROI descriptors
- 144 patient-level relational features

Thus,

```math

1612 = 1352 + 116 + 144.

```

For CMVT presence detection, the feature representation comprised 1,352 ROI-level features plus 116 CMVT-oriented features (1,468 dimensions). For the three-class and four-class tasks, the additional 144 patient-level relational features were included, giving the full 1,612-dimensional representation.

### 1.1 General notation

Let $T$ denote one stored normalized ROI matrix. Pixels outside the delineated ROI are stored as 0. Let $\varepsilon=10^{-10}$ unless otherwise specified.
For the 136-feature legacy extractor, matrices larger than $128\times128$ are resized to $128\times128$ with range preservation and anti-aliasing before feature calculation. Denote this working matrix by $I$. The additional 33 legacy descriptors and the CMVT-oriented descriptors are calculated from the input ROI matrix without this 128×128 resizing step.
The source function `extract_70d_features` returns 33 descriptors in the version used for this study. The function name is retained to preserve correspondence with the analysis code; each ROI therefore contributes $136+33=169$ ROI-level features.
Unless a definition explicitly uses valid/nonzero pixels or an inferred mask, the legacy ROI-level calculation follows the stored matrix representation, including zero-valued background pixels. The later CMVT-oriented descriptors are mask-aware and exclude the inferred background.
Entropy features use the natural logarithm. Histogram probabilities are normalized by the total bin count plus $\varepsilon$.

## 2. ROI-level features (1,352 dimensions)

The same 169 ROI-level descriptors are calculated for each of the eight ROI matrices.

### ROI-EDGE — Edge and gradient features

The implementation first computes two Sobel responses

```math

S_0=\mathrm{Sobel}(I,\mathrm{axis}=0),\qquad
S_1=\mathrm{Sobel}(I,\mathrm{axis}=1),

```

and gradient magnitude

```math

G=\sqrt{S_0^2+S_1^2}.

```

Let $Q_{75}(G)$ be the 75th percentile of $G$ and

```math

E=\{(x,y):G(x,y)>Q_{75}(G)\}.

```

The features are:
- sobel_count: $|E|$
- sobel_mean: $\mathrm{mean}(G)$
- sobel_var: $\mathrm{var}(G)$
- edge_density: $|E|/N$
- gradient_mean: $\mathrm{mean}(G)$
- gradient_var: $\mathrm{var}(G)$

Thus `gradient_mean` and `sobel_mean`, and likewise `gradient_var` and `sobel_var`, are identical by implementation.
Canny edges are calculated with `skimage.filters.canny(I, sigma=1.0)`:
- canny_count: number of Canny edge pixels
- canny_density: Canny edge pixels divided by total matrix pixels
- canny_mean: mean value of $I$ at Canny edge pixels; 0 if no edge pixel is present

For edge orientation,

```math

\theta=\mathrm{atan2}(S_1,S_0).

```

A 16-bin histogram over $-\pi,\pi$ is calculated using only pixels in $E$. If $p_k$ is the normalized histogram,

```math

H_{edge}=-\sum_k p_k\log(p_k+\varepsilon).

```

This gives edge_entropy.
A separate 32-bin histogram is calculated from all values of $G$. Its normalized probabilities $q_k$ are used for:

```math

\mathrm{grad\_hist\_mean}=\mathrm{mean}(q_k),

```

```math

\mathrm{grad\_hist\_entropy}=-\sum_k q_k\log(q_k+\varepsilon).

```

### ROI-TEXTURE — GLCM, LBP, and Gabor features

Before GLCM and LBP calculation, the working matrix is mapped to 8-bit integers:

```math

I_{8}=\mathrm{uint8}\left(255\frac{I-\min(I)}{\max(I)-\min(I)+\varepsilon}\right).

```

#### GLCM

A normalized symmetric gray-level co-occurrence matrix $P(i,j)$ is calculated with distance 1, angle 0, and 256 gray levels using graycomatrix(..., distances=[1], angles=[0], levels=256, symmetric=True, normed=True).
The implementation records:

```math

\text{GLCM contrast}=\sum_{i,j}(i-j)^2P(i,j),

```

```math

\text{GLCM homogeneity}=\sum_{i,j}\frac{P(i,j)}{1+(i-j)^2},

```

GLCM_energy as returned by graycoprops(..., 'energy'), and

```math

\text{GLCM entropy}=-\sum_{i,j}P(i,j)\log(P(i,j)+\varepsilon).

```

#### LBP

Uniform local binary patterns are calculated with P=8, R=1, and method='uniform'. A 59-bin histogram over $[0,59)$ is normalized to $p_k$.

```math

\mathrm{LBP\_entropy}=-\sum_k p_k\log(p_k+\varepsilon),

```

```math

\mathrm{LBP\_uniformity}=\sum_k p_k^2,

```

and LBP_variance is the variance of the pixelwise LBP output values.

#### Gabor

A Gabor filter is applied with skimage.filters.gabor(I, frequency=0.1); all other function arguments use the library defaults. If $R_G$ is the returned real component:

```math

\mathrm{gabor\_mean}=\mathrm{mean}(|R_G|),

```

```math

\mathrm{gabor\_var}=\mathrm{var}(|R_G|),

```

```math

\mathrm{gabor\_dir\_strength}=\mathrm{std}(R_G).

```

### ROI-SPECTRAL — Frequency-domain features

Let

```math

F_s=\mathrm{fftshift}(\mathrm{FFT2}(I)),\qquad A=|F_s|.

```

For a spectrum of size $H\times W$, the center is $(H/2,W/2)$ in integer index coordinates and

```math

R=\min(\lfloor H/2\rfloor,\lfloor W/2\rfloor).

```

Let $r(u,v)$ denote Euclidean distance from the spectral center.

#### High-, mid-, and low-frequency regions

The implementation defines:

```math

\Omega_{HF}=\{r>R/2\},

```

```math

\Omega_{MF}=\{R/4<r\le R/2\},

```

```math

\Omega_{LF}=\{r\le R/4\}.

```

High-frequency energy and ratio are

```math

E_{HF}=\sum_{(u,v)\in\Omega_{HF}}A(u,v)^2,

```

```math

R_{HF}=\frac{E_{HF}}{\sum_{u,v}A(u,v)^2+\varepsilon}.

```

HF_entropy is the entropy of a normalized 32-bin histogram of high-frequency magnitudes. HF_centroid is a legacy name; in the implementation it is the arithmetic mean of the high-frequency magnitudes, not a frequency-coordinate centroid. HF_skew and HF_kurtosis are calculated with `scipy.stats.skew` and `scipy.stats.kurtosis` using their default settings.
For the low-frequency region:

```math

E_{LF}=\sum_{\Omega_{LF}}A^2,

```

with LL_entropy from a 32-bin histogram and LL_var equal to the variance of the low-frequency magnitudes.
The legacy names LH_energy, HL_energy, and HH_energy are calculated from three quadrants of the FFT magnitude spectrum:
- LH_energy: upper-right quadrant energy
- HL_energy: lower-left quadrant energy
- HH_energy: lower-right quadrant energy
The identifiers are retained to match the feature names used in the analysis; these three quantities are calculated from quadrants of the FFT magnitude spectrum rather than from a discrete wavelet transform.

#### Radial spectrum

The interval $[0,R)$ is divided into 20 equal radial bins. For radial bin $b$, let $a_b$ be the mean magnitude in that annulus. Then

```math

\mathrm{radial\_energy}=\sum_{b=1}^{20}a_b^2.

```

A linear fit is applied to $\log(a_b+\varepsilon)$ versus bin index $b$. If the fitted slope is $\beta_1$,

```math

\mathrm{radial\_decay}=-\beta_1.

```

With $p_b=a_b/(\sum_ba_b+\varepsilon)$,

```math

\mathrm{radial\_entropy}=-\sum_b p_b\log(p_b+\varepsilon).

```

slope_logspec is the slope of a first-order polynomial fit to the flattened vector $\log(A+\varepsilon)$ versus its flattened array index.

#### Angular spectrum

Thirty-six reference angles are equally spaced over $[0,2\pi)$. For each angle $\theta_k$, the implementation averages $A$ over pixels satisfying

```math

|\mathrm{atan2}(y-y_c,x-x_c)-\theta_k|<\pi/36.

```

Let the resulting angular profile be $a_k$. The implementation defines:

```math

\mathrm{FFT\_E\_theta}=\sum_k a_k,

```

```math

\mathrm{FFT\_HF\_theta}=\sum_k a_k,

```

```math

\mathrm{FFT\_LF\_theta}=0.5\sum_k a_k,

```

```math

\mathrm{FFT\_HF\_LF\_ratio\_theta}=
\frac{\mathrm{FFT\_HF\_theta}}{\mathrm{FFT\_LF\_theta}+\varepsilon}.

```

With $p_k=a_k/(\sum_ka_k+\varepsilon)$,

```math

\mathrm{FFT\_entropy\_theta}=-\sum_kp_k\log(p_k+\varepsilon).

```

Opposite directions are separated by 18 bins, giving

```math

\mathrm{FFT\_delta\_dir}
=\mathrm{mean}|a_k-a_{k+18}|,

```

where the profile is circularly shifted by 18 positions.
The remaining composite spectral descriptors are

```math

\mathrm{Fourier\_Gabor\_Contrast}=\frac{E_{HF}}{E_{LF}+\varepsilon},

```

```math

\mathrm{Spectral\_Texture\_Index}=E_{HF}\times\mathrm{GLCM\_contrast},

```

```math

\mathrm{Band\_energy\_ratio\_MF\_LF}=\frac{E_{MF}}{E_{LF}+\varepsilon},

```

```math

\mathrm{Band\_energy\_ratio\_HF\_MF}=\frac{E_{HF}}{E_{MF}+\varepsilon}.

```

### ROI-MULTISCALE — Multiscale thermal and hotspot descriptors

The base descriptors are

```math

\mathrm{base\_mean}=\mathrm{mean}(I),\qquad
\mathrm{base\_var}=\mathrm{var}(I),

```

and base_entropy is the entropy of a normalized 32-bin histogram of $I$.
With numerical gradients $g_0=\nabla_0I$ and $g_1=\nabla_1I$,

```math

\mathrm{base\_grad}=\mathrm{mean}\sqrt{g_0^2+g_1^2},

```

```math

\mathrm{base\_contrast}=\max(I)-\min(I).

```

#### Detail image

A Gaussian-smoothed image is calculated with $\sigma=2$:

```math

I_{\sigma=2}=G_2*I,

```

and

```math

D=I-I_{\sigma=2}.

```

The features are

```math

\mathrm{detail\_energy}=\sum D^2,

```

```math

\mathrm{detail\_var}=\mathrm{var}(D),

```

detail_entropy from a normalized 32-bin histogram of $D$, and detail_grad equal to the mean absolute value across the gradient arrays returned by numpy.gradient(D).

#### Gaussian scales

Three Gaussian scales are used:
- L: $\sigma=4$
- M: $\sigma=2$
- S: $\sigma=1$

For each smoothed image $I_s$, four quantities are stored: mean, variance, 32-bin histogram entropy, and mean absolute value across the gradient arrays returned by numpy.gradient(I_s).

#### Legacy hotspot descriptors

The hotspot threshold is the 90th percentile of all values in $I$:

```math

t_{90}=Q_{90}(I),\qquad H=\{I>t_{90}\}.

```

After one binary erosion of $H$, the hotspot boundary is

```math

\partial H=H\setminus\mathrm{erode}(H).

```

The features are

```math

\mathrm{hotspot\_edge\_strength}=\mathrm{mean}(I_{\partial H}),

```

```math

\mathrm{hotspot\_clarity}=\mathrm{std}(I_H),

```

```math

\mathrm{hotspot\_contrast}=\mathrm{mean}(I_H)-\mathrm{mean}(I_{\neg H}).

```

If the required hotspot set is empty, the corresponding values are set to 0.

### ROI-DIFFUSION — Distance-based spread descriptors

The high-value seed region is defined by

```math

t_{80}=Q_{80}(I),\qquad C=\{I>t_{80}\}.

```

The Euclidean distance transform is calculated on the complement:

```math

D_{dist}=\mathrm{distance\_transform\_edt}(\neg C).

```

The implementation records

```math

\mathrm{spread\_mean}=\mathrm{mean}(D_{dist}),

```

```math

\mathrm{spread\_var}=\mathrm{var}(D_{dist}),

```

```math

\mathrm{dispersion\_index}=\frac{\mathrm{spread\_var}}{\mathrm{spread\_mean}+\varepsilon}.

```

Let $d_{max}=\lfloor\max(D_{dist})\rfloor$. For $j=0,\ldots,\min(d_{max},20)-1$, a spread profile is formed as

```math

s_j=\mathrm{mean}\{I:D_{dist}\in[j,j+1)\}.

```

Then

```math

\mathrm{growth\_rate}=\mathrm{mean}(\Delta s),

```

```math

\mathrm{curve\_entropy}=-\sum_jp_j\log(p_j+\varepsilon),\qquad
p_j=\frac{s_j}{\sum_js_j+\varepsilon},

```

```math

\mathrm{curve\_smoothness}=\mathrm{std}(\Delta s).

```

This group contains six descriptors.

### ROI-DIRECTION — Directional gradients and structure tensor

Four directional responses are defined by

```math

G_0=\mathrm{Sobel}(I,\mathrm{axis}=1),

```

```math

G_{45}=\frac{\mathrm{Sobel}(I,0)+\mathrm{Sobel}(I,1)}{\sqrt2},

```

```math

G_{90}=\mathrm{Sobel}(I,\mathrm{axis}=0),

```

```math

G_{135}=\frac{\mathrm{Sobel}(I,0)-\mathrm{Sobel}(I,1)}{\sqrt2}.

```

For each $G_\alpha$, four descriptors are stored:
- mean of $|G_\alpha|$
- variance of the signed $G_\alpha$
- maximum of $|G_\alpha|$
- entropy of a normalized 32-bin histogram of signed $G_\alpha$

For orientation coherence, let

```math

g_x=\mathrm{Sobel}(I,\mathrm{axis}=1),\quad
g_y=\mathrm{Sobel}(I,\mathrm{axis}=0),

```

```math

m=\sqrt{g_x^2+g_y^2},\qquad \theta=\mathrm{atan2}(g_y,g_x).

```

The magnitude-weighted mean orientation is

```math

\bar\theta=\frac{\sum\theta m}{\sum m}.

```

The circular difference is

```math

d=\min(|\theta-\bar\theta|,2\pi-|\theta-\bar\theta|),

```

and

```math

\mathrm{GOC}=1-\frac{\mathrm{mean}(d)}{\pi}.

```

Orient_entropy is the entropy of a 36-bin histogram of $\theta$ over $-\pi,\pi$; the histogram is not magnitude-weighted.
For the structure tensor,

```math

J_{xx}=G_{1.5}*(g_x^2),\quad
J_{yy}=G_{1.5}*(g_y^2),\quad
J_{xy}=G_{1.5}*(g_xg_y),

```

where $G_{1.5}$ denotes Gaussian smoothing with $\sigma=1.5$.
Pixelwise eigenvalues are

```math

\lambda_{1,2}=\frac12\left[J_{xx}+J_{yy}\pm
\sqrt{(J_{xx}-J_{yy})^2+4J_{xy}^2}\right].

```

The stored features are the mean $\lambda_1$, mean $\lambda_2$, their sum, and their difference.

### ROI-PERIODIC — Periodicity descriptors

A one-pixel radial spectral profile is generated from the FFT magnitude spectrum. For each integer radius $r=0,\ldots,R-1$,

```math

a_r=\mathrm{mean}\{A(u,v):r\le r(u,v)<r+1\}.

```

Peaks are detected with `scipy.signal.find_peaks` using height=mean(a_r). If the first detected peak is at index $p$,

```math

\mathrm{periodic\_freq}=\frac{p}{R},

```

```math

\mathrm{periodic\_spacing}=\frac{R}{p+\varepsilon},

```

```math

\mathrm{periodic\_magnitude}=a_p.

```

A one-dimensional profile is also calculated as

```math

v=\mathrm{mean}(I,\mathrm{axis}=0).

```

After full autocorrelation, only non-negative lags are retained. Peaks are detected above the mean autocorrelation. If at least two peaks are present at $p_1,p_2$,

```math

\mathrm{ACF\_period}=p_2-p_1,

```

```math

\mathrm{ACF\_strength}=\frac{ACF(p_1)}{ACF(0)+\varepsilon}.

```

The retained feature identifiers wavelet_peak and wavelet_period are computed from the FFT-based periodicity analysis used in the study:

```math

\mathrm{wavelet\_peak}=\max(A),

```

```math

\mathrm{wavelet\_period}=\frac{1}{\mathrm{periodic\_freq}+\varepsilon}

```

when periodic_freq > 0, otherwise 0. These two features do not use a wavelet transform in the current implementation.
For directional periodicity, four labels are used: $0^\circ,45^\circ,90^\circ,135^\circ$. The $0^\circ$, $45^\circ$, and $135^\circ$ cases use mean(I, axis=0), whereas the $90^\circ$ case uses mean(I, axis=1). For each profile, the non-DC FFT index with the maximum magnitude in the first half-spectrum is recorded as $f_k$.
The implementation then defines

```math

\mathrm{delta\_freq\_dir}=\mathrm{std}(f_k),

```

```math

\mathrm{HF\_LF\_dir\_contrast}=\frac{\max(f_k)-\min(f_k)}{\mathrm{mean}(f_k)+\varepsilon},

```

stripe_shape_entropy as the entropy of an 8-bin histogram of the four $f_k$ values,

```math

\mathrm{dominant\_periodic\_dir}=\mathrm{argmax}_{\alpha\in\{0,45,90,135\}}f_\alpha,

```

and

```math

\mathrm{periodic\_coherence}=1-\frac{\mathrm{std}(f_k)}{\mathrm{mean}(f_k)+\varepsilon}.

```

### ROI-SYMMETRY — Within-ROI left-right symmetry and composite descriptors

The working ROI matrix is split vertically. The right half is flipped left-right and both halves are cropped to the same width. Denote them by $L$ and $R_f$.
Let

```math

F_L=|\mathrm{FFT2}(L)|,\qquad
F_R=|\mathrm{FFT2}(R_f)|.

```

The frequency-symmetry descriptors are

```math

\mathrm{HF\_diff}=\mathrm{mean}|F_L-F_R|,

```

```math

\mathrm{HF\_ratio\_lr}=\frac{\sum F_L^2+\varepsilon}{\sum F_R^2+\varepsilon},

```

```math

\mathrm{FFT\_L2\_diff}=\sqrt{\sum(F_L-F_R)^2}.

```

The feature named SSIM_lr is a scalar SSIM-like expression implemented directly in the code, not a call to skimage.metrics.structural_similarity. Let $\mu_L,\mu_R$ and $\sigma_L,\sigma_R$ be the means and standard deviations of the two halves and

```math

\sigma_{LR}=\mathrm{mean}[(L-\mu_L)(R_f-\mu_R)].

```

With $c_1=0.01$ and $c_2=0.03$,

```math

\mathrm{SSIM\_lr}=
\frac{(2\mu_L\mu_R+c_1)(2\sigma_{LR}+c_2)}
{(\mu_L^2+\mu_R^2+c_1)(\sigma_L^2+\sigma_R^2+c_2)}.

```

Five composite descriptors are

```math

\mathrm{contrast×spec\_mean}=\mathrm{base\_contrast}\times\mathrm{mean}(F_L+F_R),

```

```math

\mathrm{entropy×spec\_std}=\mathrm{base\_entropy}\times\mathrm{std}(F_L+F_R),

```

```math

\mathrm{homogeneity×spec\_ratio}=\mathrm{GLCM\_homogeneity}\times\mathrm{HF\_ratio\_lr},

```

```math

\mathrm{texture\_energy×temp\_mean}=\mathrm{GLCM\_energy}\times\mathrm{base\_mean},

```

```math

\mathrm{texture\_contrast×spec\_entropy}=\mathrm{GLCM\_contrast}\times\mathrm{HF\_entropy}.

```

### ROI-INTEGRATED — Central-profile and integrated descriptors

Let the central-column profile be

```math

v(y)=I\left(y,\left\lfloor W/2\right\rfloor\right).

```

The implementation records

```math

\mathrm{LSV\_mean}=\mathrm{mean}(v),\qquad
\mathrm{LSV\_var}=\mathrm{var}(v),

```

and LSV_entropy from a normalized 16-bin histogram of $v$.
A first-order polynomial is fitted to $v(y)$ versus row index $y$:

```math

\mathrm{prof\_slope}=\text{slope of }\mathrm{polyfit}(y,v,1).

```

The remaining profile features are

```math

\mathrm{prof\_rough}=\mathrm{std}(\Delta v),

```

```math

\mathrm{prof\_curv}=\mathrm{mean}|\Delta^2v|,

```

prof_freq = index of the largest non-DC magnitude in the first half of the 1D FFT of $v$, and

```math

\mathrm{prof\_ratio}=\frac{\max(v)}{\min(v)+\varepsilon}.

```

For the flattened log FFT magnitude $\ell=\log(A+\varepsilon)$,

```math

\mathrm{spectral\_slope\_idx}=\text{slope of }\mathrm{polyfit}(0,1,\ldots,\ell-1;\ \ell,1),

```

```math

\mathrm{spectral\_curv\_idx}=\mathrm{std}(\Delta\ell).

```

Let $t_{90}=Q_{90}(I)$ and $t_{10}=Q_{10}(I)$. Define the upper- and lower-tail value sets as

```math
H_{90}=\{I_i\mid I_i>t_{90}\},\qquad
L_{10}=\{I_i\mid I_i<t_{10}\}.
```

The thermal ratio index is

```math
\mathrm{thermal\_ratio\_index}
=
\frac{\mathrm{mean}(H_{90})}
{\mathrm{mean}(L_{10})+\varepsilon}.
```

For HSI, values outside the $I>t_{90}$ hotspot mask are set to zero to form $I_H$. Then

```math

\mathrm{HSI}=\frac{\sum|\mathrm{FFT2}(I_H)|}
{\sum|\mathrm{FFT2}(I)|+\varepsilon}.

```

For MRSI, $I$ is divided into four image quadrants. If $s_q$ is the sum of FFT magnitudes in quadrant $q$,

```math

\mathrm{MRSI}=\mathrm{std}(s_1,s_2,s_3,s_4).

```

Finally,

```math

\mathrm{BRF}=\mathrm{base\_grad}\times\mathrm{thermal\_ratio\_index},

```

```math

\mathrm{SCD}=\mathrm{spectral\_curv\_idx}\times\mathrm{radial\_decay}.

```

### ROI-STAT — Basic statistical and nonzero-pixel descriptors

These descriptors are calculated from the full input ROI matrix $T$ and, where explicitly stated, from its nonzero subset

```math

V=\{T_i:T_i\ne0\}.

```

The full-matrix statistics are

```math

\mathrm{mean\_temp}=\mathrm{mean}(T),\quad
\mathrm{std\_temp}=\mathrm{std}(T),\quad
\mathrm{max\_temp}=\max(T).

```

For the nonzero subset $V$:
- nonzero_median: median($V$)
- skewness_nonzero: `scipy.stats.skew`(V) with default settings
- kurtosis_nonzero: `scipy.stats.kurtosis`(V) with default settings
- nonzero_mean: mean($V$)
- nonzero_std: std($V$)
- cv_nonzero: std($V$)/(mean($V$)+$\varepsilon$)
- nonzero_count: $|V|$
- nonzero_fraction: $|V|/N$

If $V$ is empty, the corresponding nonzero summary statistics are set to 0.

### ROI-GEOMETRY — Nonzero foreground geometry

Let $\Omega_f={(x,y):T(x,y)\ne0}$ and let $H,W$ be the matrix height and width. If $x_{min},x_{max},y_{min},y_{max}$ denote the nonzero bounding coordinates,

```math

\mathrm{bbox\_width}=\frac{x_{max}-x_{min}+1}{W},

```

```math

\mathrm{bbox\_height}=\frac{y_{max}-y_{min}+1}{H},

```

```math

\mathrm{centroid\_x}=\frac{\mathrm{mean}(x_i)}{W},\qquad
\mathrm{centroid\_y}=\frac{\mathrm{mean}(y_i)}{H}.

```

If no nonzero pixel exists, bounding-box dimensions are set to 0 and both centroid coordinates are set to 0.5.

### ROI-DISTRIBUTION — Distribution and intensity descriptors

The implementation records the 25th and 75th percentiles of the full input matrix:

```math

\mathrm{temp\_25percentile}=Q_{25}(T),\qquad
\mathrm{temp\_75percentile}=Q_{75}(T).

```

temp_entropy is calculated from a normalized 32-bin histogram of $T$.
Let $T_{max}=\max(T)$. If $T_{max}>0$,

```math

\mathrm{above\_50pct\_max}=\frac{|\{T_i\ge0.5T_{max}\}|}{N},

```

```math

\mathrm{above\_80pct\_max}=\frac{|\{T_i\ge0.8T_{max}\}|}{N}.

```

Otherwise both are set to 0.
The energy feature is

```math

\mathrm{energy}=\sum_iT_i^2.

```

### ROI-TEMP-GLCM — Temperature GLCM descriptors

The full input matrix $T$ is independently mapped to 8-bit integers using its own minimum and maximum, then a symmetric normalized GLCM is calculated with distance 1, angle 0, and 256 gray levels. The stored features are temp_glcm_contrast, temp_glcm_energy, temp_glcm_homogeneity, and temp_glcm_correlation as returned by skimage.feature.graycoprops.

### ROI-HOTREGION — Hot-region morphology

The threshold is the 90th percentile of the full input matrix:

```math

t_{90}=Q_{90}(T),\qquad H=\{T>t_{90}\}.

```

Connected components are labeled with `skimage.measure.label` using its default connectivity, and region measurements are obtained with `regionprops`.
The four features are:
- hot_region_count: number of connected hot components
- hot_region_max_area: largest component area in pixels
- hot_region_total_area: total area of all hot components in pixels
- hot_region_max_perimeter: largest component perimeter returned by `regionprops`

If no component is present, all four values are set to 0.

### ROI-THERMAL-PROFILE — Area and center-line thermal descriptors

The four descriptors are

```math

\mathrm{T\_area}=\mathrm{mean}(T),

```

```math

\mathrm{T\_line\_mean}=\mathrm{mean}\left(T[:,\lfloor W/2\rfloor]\right),

```

```math

\mathrm{T\_line\_auc}=\mathrm{trapz}\left(T[:,\lfloor W/2\rfloor]\right),

```

```math

\mathrm{PDCs\_score}=\mathrm{temp\_glcm\_contrast}\times\mathrm{temp\_entropy}.

```

## 3. CMVT-oriented features (116 dimensions)

The CMVT-oriented block contains 92 calf-centered features and 24 cross-ROI features. These descriptors are calculated without class labels.
For these features, upper denotes the calf and lower denotes the ankle-foot reference region.

### 3.1 Mask used by CMVT-oriented descriptors

An ROI mask is inferred from the zero-background representation:
1. initial foreground: $|T|>10^{-10}$;
2. if fewer than 20 foreground pixels exist, this binary mask is used directly;
3. otherwise, 4-connected components are labeled (connectivity=1), only the largest component is retained, and internal holes are filled.

Let this mask be $M$. Valid values are

```math

V=\{T(x,y):(x,y)\in M\text{ and }T(x,y)\text{ is finite}\}.

```

A mask-aware Gaussian background at scale $\sigma$ is

```math

B_\sigma=
\frac{G_\sigma*(T\,M)}{G_\sigma*M},

```

where division is performed only where the denominator exceeds $10^{-8}$; other positions remain 0.

### CMVT-ROBUST — Robust valid-pixel distribution descriptors

For each of the four calf ROIs, the implementation calculates:
- valid_p05: $Q_5(V)$
- valid_p10: $Q_{10}(V)$
- valid_p90: $Q_{90}(V)$
- valid_p95: $Q_{95}(V)$
- valid_iqr: $Q_{75}(V)-Q_{25}(V)$

The median absolute deviation is

```math

\mathrm{valid\_mad}=\mathrm{median}|V-\mathrm{median}(V)|.

```

If $n=|V|$, let $n_{10}=\max(1,\lceil0.10n\rceil)$. Then

```math

\mathrm{valid\_top10\_mean}=\text{mean of the }n_{10}\text{ largest values in }V.

```

If fewer than 10 valid pixels are available, all seven values are set to 0.

### CMVT-LOCAL — Mask-aware multiscale local contrast

Three Gaussian scales are used:

```math

\sigma\in\{3,7,15\}.

```

For each scale,

```math

R_\sigma=T-B_\sigma

```

is evaluated only within $M$. The two stored features are

```math

\mathrm{localcontrast\_s}\sigma\mathrm{\_p95}=Q_{95}(R_\sigma[M]),

```

and the mean of the largest 1% of residual values:

```math

\mathrm{localcontrast\_s}\sigma\mathrm{\_top1mean}
=\mathrm{mean}(\text{top }\lceil0.01|M|\rceil\text{ residuals}).

```

This gives 6 descriptors per calf ROI.

### CMVT-HOTRING — Hotspot-to-ring contrast descriptors

The hotspot threshold is the 90th percentile of valid ROI values:

```math

t_{hot}=Q_{90}(V).

```

A binary hotspot mask is formed by $M\cap{T\ge t_{hot}}$, and only its largest 4-connected component is retained. This largest hotspot is dilated by 8 iterations. The ring is the dilated region restricted to $M$ and excluding the hotspot itself. If the ring contains fewer than 10 pixels, the implementation falls back to $M$ excluding the hotspot.
Let $H$ and $R$ denote the valid hotspot and ring values. Then

```math

\mathrm{hotring\_deltaT}=\mathrm{mean}(H)-\mathrm{mean}(R),

```

```math

\mathrm{hotring\_cnr}=\frac{\mathrm{mean}(H)-\mathrm{mean}(R)}{\mathrm{std}(R)+\varepsilon},

```

```math

\mathrm{hotring\_robust\_cnr}=
\frac{\mathrm{median}(H)-\mathrm{median}(R)}
{1.4826\,\mathrm{MAD}(R)+\varepsilon},

```

```math

\mathrm{hotring\_area\_fraction}=\frac{|H_{mask}|}{|M|}.

```

If the required sets are insufficient, these values are set to 0.

### CMVT-SPATIAL — Spatial autocorrelation and thermal-topography descriptors

#### Global Moran's I

Let $z_i=T_i-\bar T_M$ for valid pixels, and let $E_4$ be the set of horizontal and vertical neighboring pixel pairs within $M$, with each pair counted once. Let $n=|M|$ and $m=|E_4|$. The implementation calculates

```math

I_M=\frac{n}{m}\frac{\sum_{(i,j)\in E_4}z_iz_j}{\sum_{i\in M}z_i^2}.

```

#### Geary's C

```math

C_G=\frac{n-1}{2m}
\frac{\sum_{(i,j)\in E_4}(z_i-z_j)^2}{\sum_{i\in M}z_i^2}.

```

If fewer than three valid pixels exist, the variance denominator is effectively zero, or no valid neighbor pair exists, both values are set to 0.

#### Thermal slope and Laplacian

A mask-aware Gaussian background with $\sigma=1.5$ is calculated. The mask is eroded once; if fewer than 20 pixels remain, the original mask is used. Let

```math

g_x=\partial_xB_{1.5},\qquad g_y=\partial_yB_{1.5},

```

```math

G=\sqrt{g_x^2+g_y^2}.

```

The implementation stores

```math

\mathrm{terrain\_slope\_p95}=Q_{95}(G\text{ in the interior mask}).

```

Second directional derivatives are computed numerically as

```math

L=\partial_xg_x+\partial_yg_y,

```

and

```math

\mathrm{terrain\_laplacian\_abs\_p95}=Q_{95}(|L|\text{ in the interior mask}).

```

#### Hotspot persistence

For

```math

q\in\{0.80,0.85,0.90,0.95\},

```

a threshold $Q_q(V)$ is applied. At each threshold, the largest 4-connected hotspot component is retained and its area fraction $a(q)$ relative to $|M|$ is calculated. The persistence descriptor is

```math

\mathrm{hotspot\_persistence\_auc}
=\frac{\mathrm{trapz}(a(q),q)}{0.95-0.80+\varepsilon}.

```

### CMVT-LONGITUDINAL — Distal-to-proximal calf gradient

Let the row coordinates of the calf mask be $y$. Define $y_{20}=Q_{20}(y)$ and $y_{80}=Q_{80}(y)$. The proximal and distal mask subsets are

```math

M_{prox}=M\cap\{y\le y_{20}\},

```

```math

M_{dist}=M\cap\{y\ge y_{80}\}.

```

The feature is

```math

\mathrm{distal\_minus\_proximal\_median}
=\mathrm{median}(T[M_{dist}])-
\mathrm{median}(T[M_{prox}]).

```

### CMVT-CROSS-LR — Left-right calf distribution differences

For each front/back view, the valid-pixel distributions of the left and right calf ROIs are summarized by median, 90th percentile, 95th percentile, and the complete valid-value vectors.
For statistic $s\in{\mathrm{median},P90,P95}$,

```math

D_{LR}^{(s)}=s_L-s_R.

```

The implementation also records the first Wasserstein distance

```math

W_1(V_L,V_R)

```

using `scipy.stats.wasserstein_distance`, and the two-sided Kolmogorov-Smirnov statistic

```math

D_{KS}=\sup_x|F_L(x)-F_R(x)|

```

using `scipy.stats.ks_2samp`(..., alternative='two-sided', method='asymp').
This gives 5 descriptors for the front view and 5 for the back view.

### CMVT-CROSS-UL — Calf versus ankle-foot reference differences

For each view and each side, the signed difference is calculated as calf minus ankle-foot:

```math

D_{UL}^{(s)}=s_{calf}-s_{ankle-foot},

```

for $s\in{\mathrm{median},P90}$. With two views and two sides, this gives 8 features.

### CMVT-CROSS-FB — Front-back calf differences

For each anatomical side,

```math

D_{FB}^{(s)}=s_{front,calf}-s_{back,calf},

```

for $s\in{\mathrm{median},P90}$, giving 4 features.

### CMVT-CROSS-BILATERAL — Bilateral calf P95 means

For each view,

```math

B_{P95}=\frac{P95_{left,calf}+P95_{right,calf}}{2}.

```

One value is calculated for the front view and one for the back view.

## 4. Patient-level relational features (144 dimensions)

The relational layer is a deterministic within-participant transformation of the already calculated 1,468 features. It does not use class labels, thresholds learned from labels, or statistics from other participants.

### 4.1 General descriptor set

The 120 general relational features are constructed from these ten ROI-level descriptors:
1. `base_mean`
2. `base_var`
3. `base_grad`
4. `base_contrast`
5. `hotspot_contrast`
6. `GLCM_contrast`
7. `GLCM_homogeneity`
8. `HF_ratio`
9. `Orient_entropy`
10. `dispersion_index`
For a generic descriptor $d$, define
    - $F_{UL}$: front-left calf value
    - $F_{UR}$: front-right calf value
    - $B_{UL}$: back-left calf value
    - $B_{UR}$: back-right calf value
    - $F_{LL}$: front-left ankle-foot value
    - $F_{LR}$: front-right ankle-foot value
    - $B_{LL}$: back-left ankle-foot value
    - $B_{LR}$: back-right ankle-foot value
      and patient-level averages

```math

L_c=\frac{F_{UL}+B_{UL}}{2},\qquad
R_c=\frac{F_{UR}+B_{UR}}{2},

```

```math

L_d=\frac{F_{LL}+B_{LL}}{2},\qquad
R_d=\frac{F_{LR}+B_{LR}}{2}.

```

#### REL-GEN-LR-SIGNED — View-specific left-right signed differences

For each descriptor $d$:

```math

D_{front,LR}^{(d)}=F_{UL}-F_{UR},

```

```math

D_{back,LR}^{(d)}=B_{UL}-B_{UR}.

```

Across 10 descriptors, this gives 20 features.

#### REL-GEN-LR-ABS — View-specific absolute left-right differences

```math

A_{front,LR}^{(d)}=|F_{UL}-F_{UR}|,

```

```math

A_{back,LR}^{(d)}=|B_{UL}-B_{UR}|.

```

Across 10 descriptors, this gives 20 features.

#### REL-GEN-PATIENT-LR-SIGNED — Patient-level left-right signed difference

```math

D_{patient,LR}^{(d)}=L_c-R_c.

```

This gives 10 features.

#### REL-GEN-PATIENT-LR-NORM — Patient-level normalized left-right asymmetry

```math

A_{patient,LR}^{(d)}=
\frac{|L_c-R_c|}{|L_c|+|R_c|+\varepsilon}.

```

This gives 10 features.

#### REL-GEN-UL-SIGNED — Calf versus ankle-foot signed differences

For the left and right sides:

```math

D_{left,UL}^{(d)}=L_c-L_d,

```

```math

D_{right,UL}^{(d)}=R_c-R_d.

```

Across 10 descriptors, this gives 20 features.

#### REL-GEN-UL-ABS — Calf versus ankle-foot absolute differences

```math

A_{left,UL}^{(d)}=|L_c-L_d|,

```

```math

A_{right,UL}^{(d)}=|R_c-R_d|.

```

Across 10 descriptors, this gives 20 features.

#### REL-GEN-FB-ABS — Front-back calf inconsistency

Because the anterior and posterior views represent different skin surfaces, the implementation compares descriptor values rather than pixelwise overlap:

```math

A_{left,FB}^{(d)}=|F_{UL}-B_{UL}|,

```

```math

A_{right,FB}^{(d)}=|F_{UR}-B_{UR}|.

```

Across 10 descriptors, this gives 20 features.

### 4.2 CMVT-derived bilateral/unilateral descriptor set

The remaining 24 relational features are constructed from six mask-aware calf descriptors:
1. `valid_top10_mean`
2. `localcontrast_s7_p95`
3. `localcontrast_s15_top1mean`
4. `hotring_deltaT`
5. `hotring_area_fraction`
6. `hotspot_persistence_auc`
For any one of these descriptors, let

```math

L=\frac{F_L+B_L}{2},\qquad
R=\frac{F_R+B_R}{2},

```

where front and back values are first averaged within each anatomical leg.

#### REL-CMVT-MEAN — Bilateral mean

```math

\mathrm{bilateral\_mean}=\frac{L+R}{2}.

```

#### REL-CMVT-MIN — Bilateral minimum

```math

\mathrm{bilateral\_min}=\min(L,R).

```

#### REL-CMVT-ABS — Absolute unilateral difference

```math

\mathrm{unilateral\_abs}=|L-R|.

```

#### REL-CMVT-NORM — Normalized left-right asymmetry

```math

\mathrm{lr\_norm\_asym}=
\frac{|L-R|}{|L|+|R|+\varepsilon}.

```

Applying these four relations to the six CMVT-derived descriptors gives $6\times4=24$ features.

## 5. Dimensional accounting

| Component | Calculation | Dimensions |
|---|---|---:|

| Legacy ROI-level features | 8 ROIs × (136 + 33) | 1,352 |
| CMVT calf-centered descriptors | 4 calf ROIs × 23 | 92 |
| CMVT cross-ROI descriptors | patient-level | 24 |
| CMVT-oriented subtotal | 92 + 24 | 116 |
| Binary-task representation | 1,352 + 116 | 1,468 |
| General relational features | 10 descriptors × 12 relations | 120 |
| CMVT-derived bilateral/unilateral relations | 6 descriptors × 4 relations | 24 |
| Relational subtotal | 120 + 24 | 144 |
| Complete feature library | 1,468 + 144 | 1,612 |

The complete feature names and their mapping to the formula groups in this document are provided in [`feature_dictionary_1612.csv`](../feature_dictionary_1612.csv).
