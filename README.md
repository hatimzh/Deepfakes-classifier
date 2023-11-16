# Deepfakes-classifier
Recommendation of a method to classify real and fake videos.
>**Note:** It's impossible to upload the notebook due to large file size.
>this is why I write its contents here.
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
- #### <font color="skyblue">function use Haarcascade :</font>

```python
def visualize_frame_casc(filename, meta, train = True):
    '''
    Helper function to visualize the 1st frame of the video by filename and metadata
    INPUT:
        filename - video filename
        meta - dataframe containing metadata.json
        train - indicates that the video is among train samples and the label can be retrived from metadata
    '''
    # get the 1st frame of the video
    image = get_frame(filename)

    # Display the 1st frame of the video
    fig, axs = plt.subplots(1,3, figsize=(20,7))
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].set_title('Original frame')

    # Extract the face with haar cascades
    face_cascade = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_default.xml')

    # run the detector
    # the output here is an array of detections; the corners of each detection box
    # if necessary, modify these parameters until you successfully identify every face in a given image
    faces = face_cascade.detectMultiScale(image, 1.2, 3)

    # make a copy of the original image to plot detections on
    image_with_detections = image.copy()

    # loop over the detected faces, mark the image where each face is found
    for (x,y,w,h) in faces:
        # draw a rectangle around each detected face
        # you may also need to change the width of the rectangle drawn depending on image resolution
        cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)

    axs[1].imshow(image_with_detections)
    axs[1].axis('off')
    axs[1].set_title('Highlight faces')

    # crop out the 1st face
    crop_img = image.copy()
    for (x,y,w,h) in faces:
        crop_img = image[y:y+h, x:x+w]
        break;

    # plot the 1st face
    axs[2].imshow(crop_img)
    axs[2].axis('off')
    axs[2].set_title('Zoom-in face')

    if train:
        plt.suptitle('Image {image} label: {label}'.format(image = filename.split('/')[-1], label=get_label(filename, meta)))
    else:
        plt.suptitle('Image {image}'.format(image = filename.split('/')[-1]))
    plt.show()
```
```python
visualize_frame(train_fns[0], meta)
```
![b2c3a991-1c6a-4eae-8590-b2d42d0a459f](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/ba3cfe8f-80c7-47db-a46f-e85b0f77078b)

On this video the nose of the person is strange.

```python
visualize_frame(train_fns[4], meta)
```
![0bb52b54-faaa-464a-a6cd-778a7feb23ac](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/ae631565-3330-4a1d-88a0-31e42a9dba93)

- #### <font color="skyblue">Haarcascad results :</font>

```python
visualize_frame_casc(train_fns[4], meta)
```
![79e5fdaf-a501-44a3-8987-d8185d64f189](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/8f24a25f-b8d0-4079-93b8-36b9adfd45de)

The glasses of this don't look very realistic. There is also a strange rounded shape around the right eye of the lady. Strange white spot to the right of the mouth.

```python
visualize_frame(train_fns[8], meta)
```
![a23aaff2-9480-4c30-a55f-cbe57871d7e9](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/0c3096fb-bd62-4fb2-aa09-fee85e411397)

- #### <font color="skyblue">Haarcascad results :</font>

```python
visualize_frame_casc(train_fns[8], meta)
```
![dddc4599-24b7-4953-bf49-2e7fbb2c719f](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/96cb0334-8a8c-41dd-9a73-5b2fbbe3a21b)

The face of this person is so blurry.

Let's also look at a couple of real images:

```python
visualize_frame('../input/deepfake-detection-challenge/train_sample_videos/afoovlsmtx.mp4', meta)
```
![547ce392-6dba-4eb2-9209-2f9e396f4569](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/8c314575-238e-4298-9645-24534723f8e9)

- #### <font color="skyblue">Haarcascad results :</font>
```python
visualize_frame_casc('../input/deepfake-detection-challenge/train_sample_videos/afoovlsmtx.mp4', meta)
```
![3fb1f5f5-acbb-485a-a01b-da0f7208fafa](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/7fe66ed4-3e73-4fa0-9bfb-351ecf9b5f10)

We can see that real faces have such details as:
* actual teeth (not just one white blob);
* glasses with reflections.

These fakes are really nice! Only small details tell that those are not real.

## Preview Multiple Frames

Let's look at multiple frames:

```python
import math
def get_frames(filename):
    '''
    Get all frames from the video
    INPUT:
        filename - video filename
    OUTPUT:
        frames - the array of video frames
    '''
    frames = []
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break;

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(image)

    cap.release()
    return frames

def create_animation(filename):
    '''
    Function to plot the animation with matplotlib
    INPUT:
        filename - filename of the video
    '''
    fig = plt.figure(figsize=(10,7))
    frames = get_frames(filename)

    ims = []
    for frame in frames:
        im = plt.imshow(frame, animated=True)
        ims.append([im])

    animation = ArtistAnimation(fig, ims, interval=30, repeat_delay=1000)
    plt.show()
    return animation

def visualize_several_frames(frames, step=100, cols = 3, title=''):
    '''
    Function to visualize the frames from the video
    INPUT:
        filename - filename of the video
        step - the step between the video frames to visualize
        cols - number of columns of frame grid
    '''
    n_frames = len(range(0, len(frames), step))
    rows = n_frames // cols
    if n_frames % cols > 0:
        rows = rows + 1

    fig, axs = plt.subplots(rows, cols, figsize=(20,20))
    for i in range(0, n_frames):
        frame = frames[i]

        r = i // cols
        c = i % cols

        axs[r,c].imshow(frame)
        axs[r,c].axis('off')
        axs[r,c].set_title(str(i))

    plt.suptitle(title)
    plt.show()
```
```python
frames = get_frames(train_fns[0])
visualize_several_frames(frames, step=50, cols = 2, title=train_fns[0].split('/')[-1])
```
![ab1fd075-dc5d-4f1b-86f3-7abf38a31e24](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/3ecd5184-7b4e-4321-99dd-fc05e49af308)

Static images don't look so bad, but if we look at the video (use the code for animation: `create_animation` function above) we see a lot of artifacts, which tell us that the video is fake.

Now let's look closer at the person's face in motion:

> Some technics to free RAM memory

```python
import gc
gc.collect()
```
```shell
25762
```
```python
del train_fns
del test_fns
```
```python
import matplotlib.animation as animation
def get_frames_zoomed(filename):
    '''
    Get all frames from the video zoomed into the face
    INPUT:
        filename - video filename
    OUTPUT:
        frames - the array of video frames
    '''
    frames = []
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print(f"Error opening video file {filename}")
        return None

    face_cascade = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()

        if not ret:
            break;

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(image, 1.2, 3)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            crop_img = image[y:y+h, x:x+w]
            frames.append(crop_img)

    cap.release()
    return frames

def create_animation_zoomed(filename):
    '''
    Function to create the animated cropped faces out of the video
    INPUT:
        filename - filename of the video
    '''
    fig, ax = plt.subplots(1,1, figsize=(10,7))
    frames = get_frames_zoomed(filename)

    def update(frame_number):
        plt.axis('off')
        plt.imshow(frames[frame_number])

    animation_obj = animation.FuncAnimation(fig, update, frames=len(frames), interval=30, repeat=True)
    return animation_obj
```
```python
animation = create_animation_zoomed(train_fns[0])
HTML(animation.to_jshtml())
```
![download](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/19d5eacb-16bd-4781-92cc-8a504bdf5eb2)

![7a4bb456-2fd7-441f-9ed1-3b6640736ca4](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/121f94a0-d3db-4015-82b3-bb5bb32c4d60)

We clearly see that the video is fake looking closer at the face! Some frames are really creepy. And there is flickering.

```python
# visualize the zoomed in frames
frames_face = get_frames_zoomed(train_fns[0])
visualize_several_frames(frames_face, step=55, cols = 2, title=train_fns[0].split('/')[-1])
```
![c8c7e3fb-4cf9-442e-8341-55b8f5ab2648](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/730ec9df-759a-4c64-97df-3392ffa7bcb6)

Individual frames don't look too bad. This means that we have to build models using maximum frames, we can't just sample some frames. But we can use only frames containing faces to train the model.

## Explore the Similarity between Frames

```python

import tensorflow as tf

def get_similarity_scores(frames):
    '''
    Get the list of similarity scores between the frames.
    '''
    scores = []
    for i in range(1, len(frames)):
        frame = frames[i]
        prev_frame = frames[i-1]

        # Convert the frames to tensors
        frame = tf.convert_to_tensor(frame)
        prev_frame = tf.convert_to_tensor(prev_frame)

        # Calculate the SSIM score between the frames
        score = tf.image.ssim(frame, prev_frame, max_val=255)
        scores.append(score)
    return scores


def plot_scores(scores):
    '''
    Plot the similarity scores
    '''
    plt.figure(figsize=(12,7))
    plt.plot(scores)
    plt.title('Similarity Scores')
    plt.xlabel('Frame Number')
    plt.ylabel('Similarity Score')
    plt.show()
```
We are using TensorFlow to make our program run faster with the GPU
- #### <font color='skyblue'>Using Tensorflow:</font>
```python
#tf.config.experimental.set_visible_devices([], 'GPU')
scores = get_similarity_scores(frames)
plot_scores(scores)
```
![f1720369-d268-41f0-b334-f140b3fb3e77](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/35fbbf4e-9cf3-43a5-a103-855f3812f9ae)

We can see that there are some similarity drops, let's try to look at the frames in this area:

```python
max_dist = np.argmax(scores[1:50])
max_dist
plt.imshow(frames_face[max_dist])
```
```shell
<matplotlib.image.AxesImage at 0x7b3d04319d30>
```
![52e93fb9-56da-4c85-a3d0-bb76b1982baf](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/5d5c7b28-df72-4b74-9b71-0570a2477160)

```python
plt.imshow(frames_face[max_dist+5])
```
```shell
<matplotlib.image.AxesImage at 0x7b3d042f81d0>
```
Let's compare similarity score with the original video (it is not among samples, I uploaded it in separate dataset):
Open video and look at the first frame:
```python
visualize_frame('../input/deepfake-detection-challenge/test_videos/bkuzquigyt.mp4', meta, train = False)
```
![0d447826-3ce6-4637-8b6c-ccbf628c32cc](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/6ccb5a09-27d1-4e3d-be55-66eebb724146)

The difference between real and fake is quite clear. Just look at the nose.
Let's get the frames and plot the similarity scores:
```python
# get frames from the original video
orig_frames = get_frames('../input/deepfake-detection-challenge/test_videos/bkuzquigyt.mp4')
# plot similarity scores
orig_scores = get_similarity_scores(orig_frames)
plot_scores(orig_scores)
```
![fb0994ff-aae4-4698-81f7-49c391f85227](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/0c009d53-918e-465f-8c75-d7473ca3e37b)

Plot similarity scores together:

```python
plt.figure(figsize=(12,7))
plt.plot(scores, label = 'fake image', color='r')
plt.plot(orig_scores, label = 'real image', color='g')
plt.title('Similarity Scores (Real and Fake)')
plt.show()
```
![8c48a105-b204-4c3c-aced-5924a3f0e3ff](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/3c0fd559-4030-4064-9657-06b9d9f82093)

The similarity scores of the original image are almost identical __if we take the whole frames__. Image similarity could still work if taking only frames containing faces. But I should fix the face detection first.
- #### <font color='skyblue'>Classification of the SSIM :</font>

```python
def classify(scores_list):
    '''
    INPUT :
        takes liste of scores of the video
    OUTPUT :
        Type of the video
    '''
    median=(max(scores_list)-min(scores_list))/2
    if  median<0.1:
        if (min(scores_list))>0.92:
            print("It's a real video !")
        else:
            print("It's a fake video !!")
    else :
        print("Maybe it's fake !")
```
- ### <font color='skyblue'>TEST:</font>
---
> **<font color="#00CCAA">videos in the test dataset :</font>**
```python
visualize_frame('../input/deepfake-detection-challenge/test_videos/ryxaqpfubf.mp4', meta, train = False)
```
![be8b087d-5534-4f77-85e4-61590e559fe6](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/148aa4fc-d61a-43e0-9d45-854a1b5f7bb5)

```python
# get frames from the real test video in our dataset
real_frames = get_frames('../input/deepfake-detection-challenge/test_videos/ryxaqpfubf.mp4')
# plot similarity scores
real_scores = get_similarity_scores(real_frames)
plot_scores(real_scores)
```
![cc4ea8a7-86f7-4596-a194-2a35ebe40e2a](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/5b2fc6eb-a79a-41fa-b7c2-094a65890b41)

```python
# check if it fake or real
classify(real_scores)
```
```shell
It's a real video !
```
```python
visualize_frame("../input/deepfake-detection-challenge/test_videos/ahjnxtiamx.mp4",meta,train=False)
```
![4028b07f-705c-45cf-83f6-536eb2a5743f](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/134eb3eb-c84f-4609-9407-077f511a9981)

```python
# get frames from the fake test video in our dataset
fake_frames = get_frames("../input/deepfake-detection-challenge/test_videos/ahjnxtiamx.mp4")
# plot similarity scores
fake_scores = get_similarity_scores(fake_frames)
plot_scores(fake_scores)
```
![2e0d18f0-8fc7-4738-8e10-bb4ec0b56720](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/9145c8a1-544d-4565-a07e-0bf30bd06ec3)

```python
# check if it fake or real
classify(fake_scores)
```
```shell
It's a fake video !!
```
> **<font color="#00CCAA">Problem of the new deepfake methods :</font>** (released a weeks ago)
```python
visualize_frame("../input/fake-nizar/swapped-video.mp4",meta,train=False)
```
![f1c1678c-0d63-4230-b9f6-be6e699043c5](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/c87ac36a-3a13-4683-9435-d1014605bc64)

```python
# get frames from the fake test video in our dataset
test_frames = get_frames('../input/fake-nizar/swapped-video.mp4')
# plot similarity scores
test_scores = get_similarity_scores(test_frames)
plot_scores(test_scores)
```
![be6c2c58-7c63-4142-9b86-7be56e62156b](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/62f76eea-7dcd-48ab-ac33-c83bbb6d9f07)

```python
#Check if it's fake or real
classify(test_scores)
```
```shell
Maybe it's fake !
```
```python
visualize_frame("../input/fake-nizar/video.mp4",meta,train=False)
```
![51283103-abc4-4d10-81b6-7dd2e948fe7e](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/115c452c-033b-41fc-ad14-b1932b67e788)

```python
# get frames from the real test video in our dataset
test_frames2 = get_frames('../input/fake-nizar/video.mp4')
# plot similarity scores
test_scores2 = get_similarity_scores(test_frames2)
plot_scores(test_scores2)
```
![6152ae47-b535-49ce-a1ed-fbdcad984db6](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/a2953ba7-8b46-4ff5-9845-a721ac74bcda)

```python
plt.figure(figsize=(12,7))
plt.plot(test_scores, label = 'swaped video', color='blue')
plt.plot(test_scores2, label = 'original video', color='red')
plt.title('Difference between original and swaped video')
plt.show()
```
![9356ce17-6720-4249-bd34-e4de456a4e6d](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/2039cc39-83cf-4e99-8023-34b22d642cf8)

```python
visualize_frame("../input/fake-nizar/fake.mp4",meta,train=False)
```
![cd196732-541a-4057-bb86-c96cbefbfb1c](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/309d5b64-4200-4c91-9521-3be85f655057)

```python
# get frames from the fake test video in our dataset
test_frames3 = get_frames('../input/fake-nizar/fake.mp4')
# plot similarity scores
test_scores3 = get_similarity_scores(test_frames3)
plot_scores(test_scores3)
```
![bb63bb3d-8287-4313-89c6-6688050bb8b1](https://github.com/hatimzh/Deepfakes-classifier/assets/96501113/c51d27cf-79b2-42ea-92de-0f9d7f91be52)

```python
#Check if it's fake or real
classify(test_scores3)
```
```shell
Maybe it's fake !
```
## Deepfake Research Papers

1. [Unmasking DeepFakes with simple Features](https://arxiv.org/pdf/1911.00686v2.pdf): The method is based on a classical frequency domain analysis
followed by a basic classifier. Compared to previous systems, which need to be fed with large amounts of labeled data, this
approach showed very good results using only a few annotated training samples and even achieved good accuracies in fully
unsupervised scenarios. [Github repo](https://github.com/cc-hpc-itwm/DeepFakeDetection)

2. [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/pdf/1901.08971v3.pdf): This paper
examines the realism of state-of- the-art image manipulations, and how difficult it is to detect them, either automatically
or by humans. [Github repo](https://github.com/ondyari/FaceForensics)

3. [In Ictu Oculi: Exposing AI Generated Fake Face Videos by Detecting Eye Blinking](https://arxiv.org/pdf/1806.02877v2.pdf): Method is based on detection of eye blinking in the videos,
which is a physiological signal that is not well presented in the synthesized fake videos. Method is tested over
benchmarks of eye-blinking detection datasets and also show promising performance on detecting videos generated with DeepFake.
[Github repo](https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi)

4. [USE OF A CAPSULE NETWORK TO DETECT FAKE IMAGES AND VIDEOS](https://arxiv.org/pdf/1910.12467v2.pdf): "Capsule-Forensics"
method to detect fake images and videos. [Github repo](https://github.com/nii-yamagishilab/Capsule-Forensics-v2)

5. [Exposing DeepFake Videos By Detecting Face Warping Artifacts](https://arxiv.org/pdf/1811.00656v3.pdf): Deep learning based
method that can effectively distinguish AI-generated fake videos (referred to as DeepFake videos hereafter) from real videos.
Method is based on the observations that current DeepFake algorithm can only generate images of limited resolutions, which
need to be further warped to match the original faces in the source video. Such transforms leave distinctive artifacts in
the resulting DeepFake videos, and we show that they can be effectively captured by convolutional neural networks (CNNs).
[Github repo](https://github.com/danmohaha/CVPRW2019_Face_Artifacts)

6. [Limits of Deepfake Detection: A Robust Estimation Viewpoint](https://arxiv.org/pdf/1905.03493v1.pdf): This work gives a
generalizable statistical framework with guarantees on its reliability. In particular, we build on the information-theoretic
study of authentication to cast deepfake detection as a hypothesis testing problem specifically for outputs of GANs,
themselves viewed through a generalized robust statistics framework.

## Conclusion

In this notebook:
* I loaded saparate frames of the videos and sequences of video frames.
* I created some animation of fake videos.
* I used Haar cascades for face detection and zoomed into real and fake faces.
* I looked at the similarity between frames.

Fake videos can be detected by:
* Small missing details (nose, glasses, teeth),
* Blurry contours of the face,
* Flickering.






