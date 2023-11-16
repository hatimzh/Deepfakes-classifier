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
- #### <font color='skyblue'>Import librairies that we need :</font>

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import file utilities
import os
import glob

# import charting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
%matplotlib inline

from IPython.display import HTML

# import computer vision
import cv2
from skimage.measure import compare_ssim
```

- #### <font color='skyblue'>Using The GPU :</font> to accelerate our code

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set GPU device index (e.g., 0, 1, 2, etc.)
```
## Load Data

First of all, we need to declare the paths to train and test samples and metadata file:

```python
TEST_PATH = '../input/deepfake-detection-challenge/test_videos/'
TRAIN_PATH = '../input/deepfake-detection-challenge/train_sample_videos/'

metadata = '../input/deepfake-detection-challenge/train_sample_videos/metadata.json'
```
Look at the number of samples in test and train sets:

```python
# load the filenames for train videos
train_fns = sorted(glob.glob(TRAIN_PATH + '*.mp4'))

# load the filenames for test videos
test_fns = sorted(glob.glob(TEST_PATH + '*.mp4'))

print('There are {} samples in the train set.'.format(len(train_fns)))
print('There are {} samples in the test set.'.format(len(test_fns)))
```
```shell
There are 400 samples in the train set.
There are 400 samples in the test set.
```
And load the metadata:

```python
meta = pd.read_json(metadata).transpose()
meta.head()
```
|   | label |	split |	original |
|---|---|---|---|
| aagfhgtpmv.mp4 |	FAKE |	train |	vudstovrck.mp4 |
| aapnvogymq.mp4 |	FAKE	| train	| jdubbvfswz.mp4 |
| abarnvbtwb.mp4 |	REAL	| train	| None |
| abofeumbvv.mp4 |	FAKE	| train	| atvmxvwyns.mp4 |
| abqwwspghj.mp4 |	FAKE	| train	| qzimuostzz.mp4 |

In the metadata we have a reference to the original video, but those videos can't be found among the samples on Kaggle.

You can find the original videos if you download the whole dataset.

Analyze the number or fake and real samples:

```python
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'FAKE', 'REAL'
sizes = [meta[meta.label == 'FAKE'].label.count(), meta[meta.label == 'REAL'].label.count()]

fig1, ax1 = plt.subplots(figsize=(10,7))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=['#f4d53f', '#02a1d8'])
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Labels', fontsize=16)

plt.show()
```
![ae5e6997-a1a4-4342-96ee-aa7c2f1cecf4](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/06f555ca-b577-4099-8ddd-990bf49ddcf6)

__Only 19% of samples are real videos.__ I don't know if this is the same for the whole dataset.

## Preview Videos and Zoom into Faces

Let's start with looking at some frames of the videos and trying to look closer at the faces.
we used [MTCNN](https://github.com/ipazc/mtcnn) to detect the areas containing the faces on the image.

```python
from mtcnn import MTCNN


def get_frame(filename):
    '''
    Helper function to return the 1st frame of the video by filename
    INPUT:
        filename - the filename of the video
    OUTPUT:
        image - 1st frame of the video (RGB)
    '''
    # Playing video from file
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()

    # Our operations on the frame come here
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return image

def get_label(filename, meta):
    '''
    Helper function to get a label from the filepath.
    INPUT:
        filename - filename of the video
        meta - dataframe containing metadata.json
    OUTPUT:
        label - label of the video 'FAKE' or 'REAL'
    '''
    video_id = filename.split('/')[-1]
    return meta.loc[video_id].label

def get_original_filename(filename, meta):
    '''
    Helper function to get the filename of the original image
    INPUT:
        filename - filename of the video
        meta - dataframe containing metadata.json
    OUTPUT:
        original_filename - name of the original video
    '''
    video_id = filename.split('/')[-1]
    original_id = meta.loc[video_id].original
    original_filename = os.path.splitext(original_id)[0]
    return original_filename



def visualize_frame(filename, meta, train=True):
    '''
    Helper function to visualize the 1st frame of the video by filename and metadata
    INPUT:
        filename - video filename
        meta - dataframe containing metadata.json
        train - indicates that the video is among train samples and the label can be retrieved from metadata
    '''
    # Get the 1st frame of the video
    image = get_frame(filename)

    # Display the 1st frame of the video
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].set_title('Original frame')

    # Load the MTCNN face detector
    detector = MTCNN()

    # Detect faces in the image
    faces = detector.detect_faces(image)

    # Make a copy of the original image to plot detections on
    image_with_detections = image.copy()

    # Loop over the detected faces and mark the image where each face is found
    for face in faces:
        (x, y, w, h) = face['box']

        # Draw a rectangle around each detected face
        cv2.rectangle(image_with_detections, (x, y), (x+w, y+h), (255, 0, 0), 3)

    axs[1].imshow(image_with_detections)
    axs[1].axis('off')
    axs[1].set_title('Highlight faces')

    # Crop out the first detected face
    if len(faces) > 0:
        (x, y, w, h) = faces[0]['box']
        crop_img = image[y:y+h, x:x+w]
    else:
        crop_img = image.copy()

    # Plot the first detected face
    axs[2].imshow(crop_img)
    axs[2].axis('off')
    axs[2].set_title('Zoom-in face')

    if train:
        plt.suptitle('Image {image} label: {label}'.format(image=filename.split('/')[-1], label=get_label(filename, meta)))
    else:
        plt.suptitle('Image {image}'.format(image=filename.split('/')[-1]))
    plt.show()

```





