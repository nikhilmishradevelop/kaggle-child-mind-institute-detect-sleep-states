
This is my part of 4th Place Solution for the Child Mind Institute - Detect Sleep States (Kaggle competition).

**Detailed Document**: https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459637


# How to Reproduce (for Competition Organizers)
## Hardware
- Cloud Service Used: https://cloud.jarvislabs.ai/ (RTX6000 Ada x 1)
- Cloud Service Framework: Tensorflow-2.12
- CPU: AMD EPYC 7713 64-Core Processor
- GPU: NVIDIA RTX 6000 Ada (48 GB)
- RAM: 128 GB

## OS/platform
- Ubuntu 20.04.5 LTS

## 3rd-party software
- Python: 3.8.10
- Cuda: 12.2

## Requirements
    - Only the additional packages to be installed have been mentioned in requriements.txt (pip install -r requirements.txt)
    - Rest of the packages are already present in the cloud service provider VM

## Training
1. Upload competition dataset in `/kaggle/input/child-mind-institute-detect-sleep-states`
    - i.e. `/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet`, etc...
    in the **data** folder as specified in settings.json
    - settings.json has the path specified relative to the notebooks

2. Run following notebook to prepare input dataset:
    - `nbs/data_preprocessing.ipynb`
3. Run follwing notebooks to train models:
    - `nbs/fm_v13.ipynb`
    - `nbs/fm_v15.ipynb`
    - `nbs/fm_v20.ipynb`
    - `nbs/fm_v21.ipynb`

4. `ensembling_experiments.ipynb` is notebook is not required to generate the final submission but you can use to see a basic ensembling example. A mix of manual and automatic hyperparmeter tuning was used to tune the weights for WBF.

NOTE:
- To avoid OOM, free memory when each notebook is finished executing by restarting the notebook or killing the kernel.

## Inference(Predictions)

Please use this inference code for the final submitted solution
https://www.kaggle.com/code/nikhilmishradev/fork-of-cmi-dss-final-sub-blend-new-postprocess

You can find the original competition inference code here 
https://www.kaggle.com/nikhilmishradev/cmi-dss-final-sub-blend-new-postprocess

- Both are the same (except the new one has retrained models to check reproducibility)
- Retrained models are marked as "-final" at the end


## Supplemental Information for Competition Organizers

- Please use the cloud service provider VM environment for maximum reproducibility without any issues (refer #How to Reproduce)
- My models use tensorflow hence the results will differ slightly at each run, but the final scores on the test data will be almost the same.
- `B7. Serialized copy of the trained models` are in the 4 kaggle datasets (made public)

    **Retrained Models**
    - https://www.kaggle.com/datasets/nikhilmishradev/sleep-model-fm-v13-final
    - https://www.kaggle.com/datasets/nikhilmishradev/sleep-model-fm-v15-final
    - https://www.kaggle.com/datasets/nikhilmishradev/sleep-model-fm-v20-final
    - https://www.kaggle.com/datasets/nikhilmishradev/sleep-model-fm-v21-final

    **Original Models**
    - https://www.kaggle.com/datasets/nikhilmishradev/sleep-model-fm-v13
    - https://www.kaggle.com/datasets/nikhilmishradev/sleep-model-fm-v15
    - https://www.kaggle.com/datasets/nikhilmishradev/sleep-model-fm-v20
    - https://www.kaggle.com/datasets/nikhilmishradev/sleep-model-fm-v21
- `B8. entry_points.md` is not included because my all codes are `.ipynb` format.