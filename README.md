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
