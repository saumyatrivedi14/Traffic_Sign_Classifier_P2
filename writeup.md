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
[image4]: ./New_Traffic_Signs/1.png "Traffic Sign 1"
[image5]: ./New_Traffic_Signs/2.png "Traffic Sign 2"
[image6]: ./New_Traffic_Signs/3.png "Traffic Sign 3"
[image7]: ./New_Traffic_Signs/4.png "Traffic Sign 4"
[image8]: ./New_Traffic_Signs/5.png "Traffic Sign 5"
[image9]: ./New_Traffic_Signs/6.png "Traffic Sign 6"
[image10]: ./New_Traffic_Signs/7.png "Traffic Sign 7"
[image11]: ./New_Traffic_Signs/8.png "Traffic Sign 8"



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


#### 3. Model Training.

To train the model, initially I used the LeNet-5 architecture which has Adam Optimizer but initial both training and validation accuracy came out pretty low which means the model was underfitting (approx. 78%), I tried increasing the epoch size from 10 and reduce the batch size from 128, also changed learn rate and keep probability during dropout. I also tried adding dropout, max_pooling, avg_pooling layers in between the convolution layer and fully connected layer, but the maximum vaidation accuracy I could achieve was 93.2% with training accuracy reaching 99.0%.

So I thought of changing the arachitecture and making the netwrok more deeper by adding more layers, DAVE-2 CNN has been successfully implemented on vehicle, trained using three cameras (left, center, right), to drive using only single center camera. (https://devblogs.nvidia.com/deep-learning-self-driving-cars/). After reading the paper, I thought of using the network architecture with some modifications to the filter depth, with this latest change I achieved 94.6% validation accuracy.

Final Model Results are as follows:
* Training Accuracy - 98.9%
* Validation Accuracy - 94.6%
* Test Accuracy - 91.9%


### Test a Model on New Images

Here are eight German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11]

The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. This compares favorably to the accuracy on the test set of 91.9%

Here are the results of the prediction:

| Image			            |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority Road (12) 		| Priority Road (12)								| 
| Speed Limit (30 km/h) (1)    			| Speed Limit (30 km/h) (1)								|
| Keep Right (38)					| Keep Right (38)							|
| Right-of-way at the next intersection (11)	      		| Right-of-way at the next intersection (11)		 				|
| General Caution (18)			| General Caution (18) 							|
| Road Work (25)			| Wild Animals Crossing	(31)					|
| Turn Left Ahead (34)			|Turn Left Ahead (34)  							|
| Stop (14)			| Stop (14)	   							|

'Road Work' sign was classified incorrectly as 'Wild Animals Crossing', rest all were classified correctly. I think it happened because the sign is kind of at an inclination and also reflection of sun might have also increased the pixel values which might have confused the model. 

The code for making predictions on my final model is located in the 23rd cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a 'Priority Road' sign (probability of 1.0). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Priority Road (12)   									| 
| 0.00     				| Right-of-way at the next intersection (11)										|
| 0.00					| Traffic signals	(26)							|
| 0.00	      			| Keep right (38)				 				|
| 0.00				    | End of no passing (41)  							|

For the Second image, the model is kind of unsure that this is a 'Speed limit (30km/h)' sign (probability of 0.36). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.36         			| Speed limit (30km/h) (1)   									| 
| 0.24     				| Road work (25)										|
| 0.07					| Speed limit (60km/h) (3)							|
| 0.06	      			| Slippery road (23)				 				|
| 0.06				    | No passing for vehicles over 3.5 metric tons (10)  							|

For the Third image, the model is relatively sure that this is a 'Keep right' sign (probability of 1.0). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Keep right (38)   									| 
| 0.00     				| Speed limit (20km/h) (0)										|
| 0.00					| Speed limit (30km/h) (1)							|
| 0.00	      			| Speed limit (50km/h) (2)				 				|
| 0.00				    | Speed limit (60km/h) (3)  							|

For the Fourth image, the model is relatively sure that this is a 'Right-of-way at the next intersection' sign (probability of 1.0). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Right-of-way at the next intersection (11)   									| 
| 0.00     				| Beware of ice/snow (30)										|
| 0.00					| Traffic signals (26)							|
| 0.00	      			| Priority road	(12)				 				|
| 0.00				    | Pedestrians (27)  							|

For the Fifth image, the model is relatively sure that this is a 'General caution' sign (probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| General caution (18)   									| 
| 0.00     				| Pedestrians (27)										|
| 0.00					| Traffic signals (26)							|
| 0.00	      			| Right-of-way at the next intersection (11)				 				|
| 0.00				    | Speed limit (30km/h)    (1)  							|

For the Sixth image, the model is wrong that this is a 'Wild animals crossing' sign (probability of 0.93) instead of 'Road Work' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.93         			| Wild animals crossing (31)   									| 
| 0.03     				| Double curve (21)										|
| 0.01					| Speed limit (80km/h) (5)							|
| 0.00	      			| Road work	(25)				 				|
| 0.00				    | Slippery road (23)  							|

For the Seventh image, the model is relatively sure that this is a 'Turn left ahead' sign (probability of 1.0). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Turn left ahead (34)   									| 
| 0.00     				| Keep right (38)										|
| 0.00					| Slippery road	(23)							|
| 0.00	      			| No entry	(17)				 				|
| 0.00				    | Roundabout mandatory (40)  							|

For the Eighth image, the model is relatively sure that this is a 'Stop' sign (probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Stop (14)   									| 
| 0.00     				| Road work (25)										|
| 0.00					| Bumpy road (22)							|
| 0.00	      			| Yield	(13)				 				|
| 0.00				    | Speed limit (80km/h) (5)  							|


