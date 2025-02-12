# AI-driven Preclinical Disease Risk Assessment Using Imaging in UK Biobank
## Info
Code to the paper "AI-driven Preclinical Disease Risk Assessment Using Imaging in UK Biobank"

This is a copy of the code from the cluster. The paths and configuration files are removed for privacy reasons. The code is provided as is.

The structure of the code is as follows:
1. **_PrepareDataset_** implements the data preparation and dataset constructions using UK Biobank ICD-code sources 
2. **_TwoDImage_** implements the experiments with ResNet 3D images.
3. **_analysisNumericFeatures_** implements the experiments with the non-image and image-derived (radiomics) features.

Please create issue if you have any questions.

## Install
```bash
conda env create -f env.yml
conda activate ukb
```

