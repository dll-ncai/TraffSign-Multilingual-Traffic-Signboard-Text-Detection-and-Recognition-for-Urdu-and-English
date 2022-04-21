This is a Pytorch implementation of Text detection part of the paper.

### Contents
1. [Installations](#installations)
2. [Downloads](#downloads)
3. [Training](#training)
4. [Testing](#testing)
5. [Demo](#demo)
6. [Results](#results)

### Installations
1. The following libraries/packages are required for successful execution of this project.
  - PyTorch
  - Shapely
  - Opencv
  - TensorboardX

2. For convience, please execute `requirements.txt` in cmd prompt.

### Downloads
The following files must be downloaded for successful setup.
1. **Dataset**
    Please download the Traffsign dataset from our google drive link.
    After completion, put the training and test folders in `dataset` directory.
2. **Backbone Network**:
	  Please download the backbone model from our google drive link and put into `.\tmp\backbone_net`.
3. **pretrained Model**: 
	  Please download the backbone model from our google drive link and put into `.\tmp`.
	  
### Training
After successful completion of setup, please verify the directory paths and other parameters in `config.py`.
Then execute the following cmd.
```
python train.py
```

### Testing
To test the model, please verify the paths of test directory and also specify the pretrained model in `checkpoint` in the `config.py` file.
Then execute the following cmd.
```
python eval.py
```


### Demo
To test some demos, please download the pre-trained model, or train the model and specify its path in `checkpoint` of `config.py` file.
Then put the desired images in `.\demo\test_img`, and specify the path of `test_img_path` and `res_img_path`, you will find result in `.\demo\result_img`
2. You should also specify the pretrained model in `checkpoint`.
3. Then execute the following command.
```
python eval.py
```


### Results
Here are some results of proposed text detector on our Traffsign dataset.

![image_1](https://github.com/aatiibutt/TraffSign-Multilingual-Traffic-Signboard-Text-Detection-and-Recognition-for-Urdu-and-English/blob/patch-1/Text-Detection/Results/20200907_101850%200041.jpg)
![image_2](https://github.com/aatiibutt/TraffSign-Multilingual-Traffic-Signboard-Text-Detection-and-Recognition-for-Urdu-and-English/blob/patch-1/Text-Detection/Results/20201108_152309%204776.jpg)

In case of any query, please drop an email at matifbutt@outlook.com.
