
This is my part of 4th Place Solution for the Child Mind Institute - Detect Sleep States (Kaggle competition).

detail document: https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459637


# How to Reproduce (for Competition Organizers)
## Hardware
- Cloud Service Used: https://cloud.jarvislabs.ai/ (RTX6000 Ada x 1)
- Cloud Service Framework: PyTorch-2.1
- CPU: AMD EPYC 7713 64-Core Processor
- GPU: NVIDIA RTX 6000 Ada (48 GB)
- RAM: 128 GB

## OS/platform
- Ubuntu 20.04.5 LTS

## 3rd-party software


## Training
1. Upload competition dataset in `/kaggle/input/child-mind-institute-detect-sleep-states`
    - i.e. `/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet`, etc...
2. Run following notebook to prepare input dataset:
    - `nbs/data_preprocessing.ipynb`
3. Run follwing notebooks to train models:
    - `nbs/fm_v13.ipynb`
    - `nbs/fm_v15.ipynb`
    - `nbs/fm_v20.ipynb`
    - `nbs/fm_v21.ipynb`

NOTE:
- To avoid OOM, free memory when each notebook is finished executing by restarting the notebook.

## Supplemental Information for Competition Organizers
- Dockerfile is used instead of `B4.requirements.txt`.
- `src/config.yaml` is used instead of `B6. SETTINGS.json`.
- `B7. Serialized copy of the trained models` are in the 4 datasets
    - https://www.kaggle.com/datasets/nikhilmishradev/sleep-model-fm-v13-final
    - https://www.kaggle.com/datasets/nikhilmishradev/sleep-model-fm-v15-final
    - https://www.kaggle.com/datasets/nikhilmishradev/sleep-model-fm-v20-final
    - https://www.kaggle.com/datasets/nikhilmishradev/sleep-model-fm-v21-final
- `B8. entry_points.md` is not included because my all codes are `.ipynb` format.