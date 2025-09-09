# AAGCN-Sign-Language-Recognition-999-classes

This is an updated version of the old AAGCN model for Sign Language Recognition. In this version, I trained the model on 999 classes. Yeah, there was a minor issue when capturing the videos so the dataset is lack of 1 label. Additionally, the dataset has some missing videos, which caused issues. To address this, I cleaned up and reorganized the code, splitting it into sub-files to make it more professional and easier to manage.

I also added a solution for labels with too few videos: the code now duplicates existing videos from that label so that each class has enough data for training.

Below, you’ll find more detailed information about this project.

# Set up
Download all the files and run the jupyter file "main.ipynb". To run the main file, please change the path of the input dataset. The code will read all videos in the folder. The dataset folder follows this structure:
<pre>
  dataset/
├── video1.mp4
├── video2.mp4
├── video3.mp4
└── video4.mp4
</pre>

In case you want to have a clearer look on the training progress, you can look at the Total_(VSL_999).ipynb file. This file combines all the code together, running step by step. To run this file, all you need is just adding the dataset folder path, or maybe you can change the model values such as the number of epochs, batch size, etc in the model coding part (at the end of the file).

# Introduction
Recent advances in computer vision and deep learning have made state-of-the-art automated gesture recognition methods possible. However, these methods either require intensive computational complexity or face difficulties extracting critical features. Additionally, there has not been any large-scale dataset in Vietnamese sign language (VSL) yet. To address these problems, we proposed a new dataset with a modified Adaptive Attention Graph Convolutional Networks (AAGCN) model, which models the body as a graph of nodes, designed particularly for this dataset. The model achieves competitive results in certain studies. We also included a new preprocessing method for reconstructing the missing information. This model achieved an accuracy of 94.3% on the VSL dataset with 100 classes on the best fold.

# Dataset
## Vietnames Sign Language Dataset 
Although previous studies have applied their models to the VSL dataset, these datasets were self-collected and relatively small, so they cannot fully demonstrate their models' capabilities and effectiveness in handling the complexities of the VSL. So, in this paper, we use a new, larger-scale VSL dataset, collected from a school for hearing-impaired children in Hanoi, Vietnam. This dataset comprises video samples recorded from 30 actors, covering 999 distinct word classes that represent the most frequently used spoken Vietnamese words.

## Ankara University Turkish Sign Language Dataset (AUTSL)
Another dataset used in our study is the Ankara University Turkish Sign Language Dataset (AUTSL) [14], a large-scale dataset designed for sign language recognition tasks. AUTSL consists of 38,336 video samples performed by 43 different signers, covering 226 sign classes. The dataset provides a diverse range of signing variations, captured from multiple viewpoints, making it a suitable choice for pre-training deep learning models. By leveraging AUTSL for pre-training, we aim to enhance feature extraction capabilities and improve model generalization. After pre-training on AUTSL, that achieved the accuracy of 85%, we fine-tune and evaluate our model on the newly introduced VSL dataset to assess its performance.

# Methodology
I started by splitting the videos into frames and extracting keypoints from those frames by using Mediapipe Holistic. While Mediapipe is a strong framework, it cannot detect some keypoints in some occasion. This due to some special movement of the hands when performing the action. I used Bilinear interpolation to re-construct the missing keypoints. K-fold is used to split the dataset into train-valid set. With K = 10, 9 of them were used for training while the left one is for testing. We train the model 10 times and choose the fold with the highest accuracy. 

About the model, based on the original Graph Convolution Networks (GCN) and the inspiration with Transformer, I combined GCN model with Attention Mechanism. This includes spatial (Identifies and highlights key joints in the skeleton graph), temporal (Highlights important time frames in the sequence), and channel attentions (Identifies important feature channels in terms of output from convolutional layers). 

# Results
We have just had the result on 100 classes, which is 94.3%. The performance of this model on 999 classes will be updated soon. 

# Conclusion
For a better understand of my work, please read the article.
Thanks for reading.
