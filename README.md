Simple repository to train deep learning segmentation models on the [Next Day Wildfire Spread dataset](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread).


# Conda environment creation
We use a conda environment to manage the dependencies of this project. To create the environment, run the following command:
```
conda env create --name wildfire_env --file wildfire_env.yml
```
Note that this YAML file assumes that you have a GPU-enabled, linux-based machine with the required drivers for CUDA 11.8 installed. If your hardware configuration differs, you may need to modify the .yml file accordingly, specifically the PyTorch-related packages.

# Data download and conversion to numpy format
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread) and extract the archive to a folder of your choosing.
Then, open [convert_tfrecord_to_numpy.py](convert_tfrecord_to_numpy.py#L74) and point `data_path` variable to the path of the extracted dataset.
Once you have done that, run the script to convert the dataset to numpy format:
```
python convert_tfrecord_to_numpy.py
```
This script will write 3 npz files (train.npz, val.npz, test.npz) to `/your/specified/data/path_npz` (i.e., it adds a `_npz` suffix to the same `data_path` you specified).

# Running the training script
Open [train.py](train.py#L16) and update `WANDB_DIR` and `NPZ_DIR`. `WANDB_DIR` is the directory where Weights & Biases will store the logs of the training process. `NPZ_DIR` is the directory where the npz files are stored after running the conversion script.

Once you have updated these variables, you can run the training script:
```
python train.py
```