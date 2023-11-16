# Deepfakes-classifier
Recommendation of a method to classify real and fake videos.
 ## Introduction
Today's technology allows us to do the incredible things such as creation of fake videos or images of real people, [deepfakes](https://en.wikipedia.org/wiki/Deepfake). Deepfakes [are going viral](https://www.creativebloq.com/features/deepfake-examples) and creating a lot of credibility and security concerns. That is why deepfake detection is a fast growing area of research (I put some of the papers related to deepfakes in the end of the notebook).

In this analysis I will try to look close at the videos from the sample dataset on Kaggle and find traits which can help us distinguish the fakes from the real videos.

- #### <font color='skyblue'>Install MTCNN :</font> to use it instead of Haarcascade

```shell
pip install mtcnn
```
```shell
Collecting mtcnn
ownloading https://files.pythonhosted.org/packages/09/d1/2a4269e387edb97484157b872fa8a1953b53dcafbe4842a1967f549ac5ea/mtcnn-0.1.1-py3-none-any.whl (2.3MB)
     |████████████████████████████████| 2.3MB 4.8MB/s eta 0:00:01
Requirement already satisfied: opencv-python>=4.1.0 in /opt/conda/lib/python3.6/site-packages (from mtcnn) (4.1.2.30)
Requirement already satisfied: keras>=2.0.0 in /opt/conda/lib/python3.6/site-packages (from mtcnn) (2.3.1)
Requirement already satisfied: numpy>=1.11.3 in /opt/conda/lib/python3.6/site-packages (from opencv-python>=4.1.0->mtcnn) (1.17.4)
Requirement already satisfied: keras-applications>=1.0.6 in /opt/conda/lib/python3.6/site-packages (from keras>=2.0.0->mtcnn) (1.0.8)
Requirement already satisfied: keras-preprocessing>=1.0.5 in /opt/conda/lib/python3.6/site-packages (from keras>=2.0.0->mtcnn) (1.1.0)
Requirement already satisfied: six>=1.9.0 in /opt/conda/lib/python3.6/site-packages (from keras>=2.0.0->mtcnn) (1.13.0)
Requirement already satisfied: pyyaml in /opt/conda/lib/python3.6/site-packages (from keras>=2.0.0->mtcnn) (5.1.2)
Requirement already satisfied: scipy>=0.14 in /opt/conda/lib/python3.6/site-packages (from keras>=2.0.0->mtcnn) (1.3.3)
Requirement already satisfied: h5py in /opt/conda/lib/python3.6/site-packages (from keras>=2.0.0->mtcnn) (2.9.0)
Installing collected packages: mtcnn
Successfully installed mtcnn-0.1.1
Note: you may need to restart the kernel to use updated packages.
```

