# Mars-Rover-20-21
ROVER - IMAGE PROCESSING

Object Detection:
Object detection plays a very important role in autonomously driven robots. The main aim was to be able to detect a tennis-ball. The ball can be moving and the system should be able to detect it in different kinds of situations for example, various lighting conditions. There are several methods to accomplish this object detection task using neural networks. Further, we will be describing the procedure we went about in order to successfully accomplish the task.


![image](https://user-images.githubusercontent.com/60462821/121772004-6bd85180-cb90-11eb-88f3-d3e57aac7b83.png)


Installation:
In order to begin with the task, we first installed python3, OpenCV, Keras and Tensorflow.
OpenCV:
OpenCV (Open Source Computer Vision Library)  is the huge open-source library for computer vision, machine learning, and image processing and now it plays a major role in real-time operation which is very important in today’s systems. By using it, one can process images and videos to identify objects, faces, or even handwriting of a human.

OpenCV Course:
https://drive.google.com/folderview?id=1aoMNZr66dvexoF9pBmBc5HGaiBNr87sl
Regression:
To get started, we went through the basics of regression and the two types:

1) Linear Regression. 
 
Linear Regression is a supervised machine learning algorithm and the most popular one. It is a statistical method that is used for predictive analysis. Linear regression makes predictions for continuous/real or numeric variables such as sales, salary, age, product price, etc.

Linear regression algorithms show a linear relationship between a dependent (y) and one or more independent (y) variables, hence called linear regression. Since it shows the linear relationship, which means it finds how the value of the dependent variable is changing according to the value of the independent variable. 

Link: https://medium.com/@shuklapratik22/linear-regression-from-scratch-a3d21eff4e7c

2) Logistic Regression

Logistic regression is also a supervised machine learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables.
Logistic regression predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1

Link: https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac

 
Convolutional Neural Networks:
A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.
A convolutional neural network is a feed-forward neural network that is generally used to analyze visual images by processing data with grid-like topology. It’s also known as ConvNet. A convolutional neural network is used to detect and classify objects in an image.

 

To learn more, check out the below playlist:
https://www.youtube.com/playlist?list=PLpFsSf5Dm-pd5d3rjNtIXUHT-v7bdaEIe
Keras and Tensorflow:
TensorFlow is a software library or framework. It is used for implementing machine learning and deep learning applications. Keras is compact, easy to learn, high-level Python library run on top of TensorFlow framework. It is made with the focus of understanding deep learning techniques, such as creating layers for neural networks maintaining the concepts of shapes and mathematical details.

Course Link:
https://www.coursera.org/professional-certificates/tensorflow-in-practice?fbclid=IwAR04eaVXSC4gmFzDXo4cZ57AdKTIkyPzoDq9acnPTWVnQjuSfSHGns2jTfo#about
Dataset:
To build the dataset for our model, we gathered 2145 tennis ball images and 2145 random images which did not contain a tennis ball in them. Further, we divided the dataset of each category into two, 80% for training and 20% for testing. 

Drive Link for dataset:
https://drive.google.com/drive/folders/1gTKipmUp9z4Hj-ZsSMK4MF2KA-7BcFp0?usp=sharing 

Object detection using Convolutional Neural Networks(CNN):

For binary image classification, there are several pre-trained models that can be used to obtain a good accuracy, namely:
1.	Inceptionv3
2.	ResNet50
3.	VGG-19
4.	VGG-16

We were getting the best accuracy using the VGG-16 model. The summary of the model is:



Here are the results of some test images:


![image](https://user-images.githubusercontent.com/60462821/121772028-96c2a580-cb90-11eb-942a-32282050df3f.png)
![image](https://user-images.githubusercontent.com/60462821/121772038-9de9b380-cb90-11eb-87ac-ffa46cc8e4c8.png)


























