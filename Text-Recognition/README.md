This is a Pytorch implementation of Text recognition part of the paper.

### Contents
1. [Installations](#installations)
2. [Downloads](#downloads)
3. [Quick Demo](#demo)
4. [Training](#training)
5. [Evaluation](#evaluate)

### Installations
1. The following libraries/packages are required for successful execution of this project.
    - PyTorch
    - Docopt
    - Numpy
    - Opencv
    - Pillow
    - Scipy
    - Six
    - Tqdm

2. For convience, please execute `requirements.txt` in cmd prompt.

### Downloads
The following files must be downloaded for successful setup.
1. **Dataset**
    Please download the Traffsign text recognition dataset from our google drive link.
    After completion, put the downloaded folders in `data/mnt/ramdisk/max/90kDICT32px/` directory.
2. **pretrained Model**: 
	  Please download the pretrained model from our google drive link and put into `checkpoints/`.

### Quick Demo

After completing the aforementioned procedure, let's predict the demo images.
Please adjust hyper-parameters in `./src/config.py`. Then, execute the following command.

```command
$ python src/predict.py demo/Sample_20.jpg
```

## Training

Before starting training, adjust hyper-parameters in `./src/config.py`.
And train crnn models,

```command
$ python src/train.py
```

### Evaluation

```command
$ python src/evaluate.py
```
