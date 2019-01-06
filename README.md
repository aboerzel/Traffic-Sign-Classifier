# **Traffic Sign Recognition** 

## Udacity Self Driving Car Engineer Nanodegree - Project 3

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[class_labels]: ./examples/class_labels.png "class_labels"
[class_distribution_1]: ./examples/class_distribution_1.png "class_distribution_1"
[class_distribution_2]: ./examples/class_distribution_2.png "class_distribution_2"
[train_dataset]: ./examples/train_dataset.png "train_dataset"
[train_dataset_grayscaled]: ./examples/train_dataset_grayscaled.png "train_dataset_grayscaled"
[train_dataset_equalized]: ./examples/train_dataset_equalized.png "train_dataset_equalized"
[train_dataset_augmented]: ./examples/train_dataset_augmented.png "train_dataset_augmented"
[test_images]: ./examples/test_images.png "test_images"
[test_images_predictions]: ./examples/test_images_predictions.png "test_images_predictions"
[top-5-predictions]: ./examples/top-5-predictions.png "top-5-predictions"
[activation_map_visualization]: ./examples/activation_map_visualization.png "activation_map_visualization"
[vggnet_training_history]: ./examples/training-loss-and-accuracy_vggnet_rmsprop.png "vggnet_training_history"
[confusion_matrix]: ./examples/confusion_matrix.png "confusion_matrix"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [Traffic_Sign_Classifier.ipynb](Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

![alt text][class_labels]

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][class_distribution_1]

![alt text][class_distribution_2]


![alt text][train_dataset]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][train_dataset_grayscaled]

![alt text][train_dataset_equalized]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][train_dataset_augmented]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        		                			| 
|:---------------------:|:-------------------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   						                	| 
| Convolution 3x3     	| 32 3x3 filters, 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					| outputs 32x32x32							                    |
| Batch normalization 	| outputs 32x32x32              	                			|
| Convolution 3x3     	| 32 3x3 filters, 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					| outputs 32x32x32							                    |
| Batch normalization 	| outputs 32x32x32                 	                			|
| Max pooling	      	| 2x2 stride, outputs 16x16x32				                    |
| Dropout				| 0.25, outputs 16x16x32	        		                    |
| Convolution 3x3     	| 64 3x3 filters, 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					| outputs 16x16x64							                    |
| Batch normalization 	| outputs 16x16x64                 	                			|
| Convolution 3x3     	| 64 3x3 filters, 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					| outputs 16x16x64   						                    |
| Batch normalization 	| outputs 16x16x64                 	                			|
| Max pooling	      	| 2x2 stride, outputs 8x8x64				                    |
| Dropout				| 0.25, outputs 8x8x64						                    |
| Fully connected		| outputs 512               									|
| RELU					| outputs 512  			       				                    |
| Batch normalization 	| outputs 512               	                    			|
| Dropout				| 0.5, outputs 512									            |
| Fully connected		| outputs 43                									|
| Softmax				| outputs 43                									|

 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.66%
* validation set accuracy of 97.30%
* test set accuracy of 94.66%

![alt text][vggnet_training_history]


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


![alt text][confusion_matrix]


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][test_images]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
![alt text][test_images_predictions]

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

![alt text][top-5-predictions]

For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][activation_map_visualization]