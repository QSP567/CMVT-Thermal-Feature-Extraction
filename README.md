# CMVT Thermal Feature Extraction

This repository contains the feature-extraction implementation,
feature dictionary, and mathematical definitions of the handcrafted
infrared thermography features used in our CMVT study.

## Feature architecture

The complete handcrafted feature library contains 1,612 features:

- 1,352 ROI-level features
- 116 CMVT-oriented features
- 144 patient-level relational features

For CMVT presence detection, a 1,468-dimensional representation was used,
comprising the 1,352 ROI-level features and 116 CMVT-oriented features.

For the three-class and four-class classification tasks, an additional
144 patient-level relational features were included, resulting in the
full 1,612-dimensional representation.

Each participant was represented by eight ROI temperature matrices:

- Front left calf
- Front right calf
- Front left ankle-foot
- Front right ankle-foot
- Back left calf
- Back right calf
- Back left ankle-foot
- Back right ankle-foot

The complete feature library was constructed as:

**1,612 = 1,352 + 116 + 144**

## 1. ROI-level features

The 1,352 ROI-level features were obtained from eight ROIs, with
169 descriptors calculated for each ROI.

These descriptors include statistical, gradient and edge, texture,
frequency-domain, multiscale thermal, spatial-distribution, periodicity,
structure-tensor, hotspot, and intra-ROI symmetry features.

## 2. CMVT-oriented features

A total of 116 CMVT-oriented features were added:

- 92 calf-centered features: 23 descriptors × 4 calf ROIs
- 24 patient-level cross-ROI features

The calf-centered descriptors characterize robust thermal distributions,
multiscale local thermal contrast, hotspot-to-ring contrast, spatial
autocorrelation and thermal topography, hotspot persistence, and
distal-to-proximal thermal variation.

The cross-ROI descriptors characterize left-right calf differences,
calf-to-ankle-foot relationships, front-back calf relationships, and
bilateral calf thermal patterns.

## 3. Patient-level relational features

A further 144 patient-level relational features were constructed from
predefined within-participant anatomical relationships:

- 120 features: 10 core descriptors × 12 anatomical relationships
- 24 bilateral/unilateral relational features:
  6 CMVT descriptors × 4 relationship measures

All relational features were calculated from measurements belonging to
the same participant and did not use class labels.

## Repository contents

- [`feature_dictionary_1612.csv`](feature_dictionary_1612.csv)  
  Complete list of the 1,612 feature names and their corresponding
  feature families, anatomical regions, descriptors, and definitions.

- [`docs/FEATURE_DEFINITIONS.md`](docs/FEATURE_DEFINITIONS.md)  
  Mathematical definitions, parameter settings, and calculation rules
  for the feature families used in this study.

- [`src/feature_extraction_1612.py`](src/feature_extraction_1612.py)  
  Python implementation of the complete handcrafted feature-extraction
  pipeline.

## Anatomical terminology

In this repository:

- `upper` refers to the calf region
- `lower` refers to the ankle-foot region
- `front` and `back` refer to the two thermographic acquisition views

The original variable names are retained in the source code for
reproducibility.

## Reproducibility

The feature-extraction procedure is deterministic for a given set of
input ROI temperature matrices.

Feature calculation itself does not use binary, three-class, or
four-class outcome labels.
