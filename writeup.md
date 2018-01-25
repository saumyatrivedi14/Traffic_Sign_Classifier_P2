# **Traffic Sign Recognition** 

---
**Introduction**

The project uses the DAVE-2 (https://devblogs.nvidia.com/deep-learning-self-driving-cars/) Deep Learning Model architecture in training a Convolutional Neural Network to classify 43 different German Traffic Signs using TensorFlow. In total this CNN uses 9 layers including one layer of data pre-processing method (normalization), followed by 5 convolutional layers and 3 fully-connected layers augumented by regularization technique (Dropout). In addition to the given validation data and test data, the trained network is also tested on 8 randomly chosen images from the Web.

**Goals of the Project**

* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Report_Pics/Dataset1.JPG "Visualization"
[image2]: ./Report_Pics/Dataset2.JPG
[image3]: ./Report_Pics/Normalization.JPG "Normalization"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"



---

### Data Set Summary & Exploration

I used the numpy and matplotlib libraries to calculate summary statistics and visualize the traffic signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### Visualization of the dataset.

Here is an exploratory visualization of the data set. The first image shows a random traffic sign pic selected from the dataset with the class index (class index and their descriptions of the traffic sign are provided in signnames.csv file in the repository). There are three bar charts shown below which gives the data distribution across all 43 class for each Training, Validation and Test set.

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

### 1. Pre-Processing of Dataset

As a first step, I decided to normalize the images in the dataset and fit teach pixel value in the range of 0-1 (because plt.imshow takes RGB values in the range of 0-1) to reduce the spread of data and the overall optimization duration (as the data is well conditioned). I tried grayscaling the images but that didn't improve the accuracy of the Network and it worked pretty well without it.

Here is an example of a traffic sign image before and after Normalization.

![alt text][image3]


#### 2. Model Architecture - Similar to DAVE-2 (Developed by NVIDIA)

My final model consisted of the following layers:

| Layer         		      |     Description	        					                 | 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x3 RGB image   							                   | 
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 28x28x6 	  |
| RELU					             |												                                   |
| Convolution 5x5	      | 1x1 stride, valid padding, outputs 24x24x16			|
| RELU					             |												                                   |
| Convolution 5x5	      | 2x2 stride, valid padding, outputs 10x10x36			|
| RELU					             |												                                   |
| Convolution 3x3	      | 2x2 stride, valid padding, outputs 4x4x48  			|
| RELU					             |												                                   |
| Convolution 3x3	      | 2x2 stride, valid padding, outputs 1x1x64		  	|
| RELU					             |												                                   |
| Flatten					          |	outputs 64	                                   |
| Fully connected		     | outputs 120                          									|
| RELU   				           |                                      									|
|	Dropout               |	Keep Probability of 75%            											|
| Fully connected		     | outputs 84                          									 |
| RELU   				           |                                      									|
|	Dropout               |	Keep Probability of 75%            											|
| Fully connected		     | outputs 43                          									 |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

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
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


