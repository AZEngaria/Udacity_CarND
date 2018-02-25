# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualise_final.png "Visualization"
[image2]: ./grayscale.PNG "Grayscaling"
[image4]: ./SampleDataWriteup/german_1.jpg "Traffic Sign 1"
[image5]: ./SampleDataWriteup/german_3.jpg "Traffic Sign 2"
[image6]: ./SampleDataWriteup/german_4.jpg "Traffic Sign 3"
[image7]: ./SampleDataWriteup/german_5.jpg "Traffic Sign 4"
[image8]: ./SampleDataWriteup/german_6.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed with respect to number of classes in the training, 

![alt text][image1]
<pre>
ClassID SignName                                                % Training  % Validation
0       Speed limit (20km/h)                                     0.52       0.68
1       Speed limit (30km/h)                                     5.69       5.44
2       Speed limit (50km/h)                                     5.78       5.44
3       Speed limit (60km/h)                                     3.62       3.40
4       Speed limit (70km/h)                                     5.09       4.76
5       Speed limit (80km/h)                                     4.74       4.76
6       End of speed limit (80km/h)                              1.03       1.36
7       Speed limit (100km/h)                                    3.71       3.40
8       Speed limit (120km/h)                                    3.62       3.40
9       No passing                                               3.79       3.40
10      No passing for vehicles over 3.5 metric tons             5.17       4.76
11      Right-of-way at the next intersection                    3.36       3.40
12      Priority road                                            5.43       4.76
13      Yield                                                    5.52       5.44
14      Stop                                                     1.98       2.04
15      No vehicles                                              1.55       2.04
16      Vehicles over 3.5 metric tons prohibited                 1.03       1.36
17      No entry                                                 2.84       2.72
18      General caution                                          3.10       2.72
19      Dangerous curve to the left                              0.52       0.68
20      Dangerous curve to the right                             0.86       1.36
21      Double curve                                             0.78       1.36
22      Bumpy road                                               0.95       1.36
23      Slippery road                                            1.29       1.36
24      Road narrows on the right                                0.69       0.68
25      Road work                                                3.88       3.40
26      Traffic signals                                          1.55       1.36
27      Pedestrians                                              0.60       0.68
28      Children crossing                                        1.38       1.36
29      Bicycles crossing                                        0.69       0.68
30      Beware of ice/snow                                       1.12       1.36
31      Wild animals crossing                                    1.98       2.04
32      End of all speed and passing limits                      0.60       0.68
33      Turn right ahead                                         1.72       2.04
34      Turn left ahead                                          1.03       1.36
35      Ahead only                                               3.10       2.72
36      Go straight or right                                     0.95       1.36
37      Go straight or left                                      0.52       0.68
38      Keep right                                               5.34       4.76
39      Keep left                                                0.78       0.68
40      Roundabout mandatory                                     0.86       1.36
41      End of no passing                                        0.60       0.68
42      End of no passing by vehicles over 3.5 metric tons       0.60       0.68
</pre>

### Design and Test a Model Architecture

#### Techniques for Preprocessing the image data.

As a first step, I decided to convert the images to grayscale because image color is not a distinguishing feature for traffic signs. There are no two traffic signs with different colors and same symbol.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data to Centering the image values - (x_train-128.0)/128.0 as these values work well with CNNs that have RELU activations.

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and can be used in this project.


#### 2. Model architecture

My final model consisted of the following layers:

| Layer             		|     Description	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Input             		| 32x32x1 GrayScale image   					| 
| Convolution1 5x5x1x6     	| 1x1 stride, Valid Padding, outputs 28x28x6 	|
| RELU    					|												|
| Max pooling       	 	| 2x2 stride,  outputs 14x14x6  				|
| Convolution2 5x5x6x16 	| 1x1 stride, Valid Padding, outputs 10x10x16 	|
| RELU    					|												|
| Max pooling       	 	| 2x2 stride,  outputs 5x5x16   				|
| Fully connected   		| 400x120. Outputs = 120. Bias 120  			|
| RELU    					|												|
| Dropout    				| Keep Probability: 50%							|
| Fully connected   		| 120x84. Output = 84. Bias 84       			|
| RELU    					|												|
| Dropout    				| Keep Probability: 50%							|
| Fully connected   		| 84x43. Output = 43. Bias 43       			|
| Softmax   				|  									            |
 


#### 3. Hyperparamerters. 

To train the model, I used Stochastic Gradient Descent optimized by AdamOptimizer at a learning rate of 0.00097. Each batch was a randomized sample of 156 training samples. 


#### 4. The Approach to reach validation set accuracy to be at least 0.93.

My final model results were:
* Train set accuracy of 0.998
* validation set accuracy of 0.958
* test set accuracy of 0.944

The approach to classify the traffic symbols was to implement a standard Lenet-5 CNN and iteratively tune it to improve performance for this specific dataset. The Lenet-5 model comprises of a stack of two convolution layers and three fully connected layers with RELU activations interleaved betweeen them. The convolutions layers outputs are also fed through MaxPooling layers after RELU. One of the changes that improved performance for this dataset is the inclusion of dropout layers connected to fully-connected layers. This was added when I noticed the model was overfitting to the training data set. Learning rate, batch size and the probablity for the dropout layers were the most important hyperparameters that I had to tune. My initial learning rate of 0.05 with the GradientDescent optimizer was failing to train, possibly getting stuck at a local optima. Reducing learning rate by an order was sufficient to get the model to train. I also switched the optimizer to AdamOptimizer as it converged significantly faster than GradientDescent.
 

### Test a Model on New Images

#### 1. Qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

One of the interesting things I noticed was the model fails to classify a "known" traffic sign if the sign is not centered or does not cover a significant part of the image. Cropping the image to mostly include just the sign gives 100% accuracy. This shows that the dataset is insufficient and makes a good case for augmenting the data set with transformed images.

Another observation is that model appears to have low precision in some cases. Testing with an unseen input (last image) - "No Parking" results in the model classifying it with more that 60% accuracy as a "End of no passing". I believe, this probablity would have been lesser if the color components were included in images used for training. The second image is not in the training dataset.

#### 2. Model's Predictions on new traffic signs.

Here are the results of the prediction:

|			  Image 								|     Prediction	        					| 
|:-------------------------------------------------:|:---------------------------------------------:| 
| Road work                                 		| Road work    									| 
| No Parking                            			| End of no passing   							|
| Speed limit (60km/h)                              | Speed limit (60km/h)							|
| No entry                            	      		| No entry  					 				|
| Right-of-way at the next intersection 			| Right-of-way at the next intersection  		|  


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 12630 samples.

#### 3.  Top 5 softmax probabilities for each prediction. 

The top five soft max probabilities for the 5 test data are below. The model classifies signs with almost 70% certainty.

For the second image (No Parking Image), the model is relatively sure that this is a End of No Passing sign (probability of 0.6), and the image does contain a No Parking. The top five soft max probabilities were


|			  Max Probability 								|     Prediction	        					| Probabilities        					           |
|:--------------------------:|:----------------------------:| :-----------------------------------:| 
|  0.687683702                                 		| Road work    								| '0.687', '0.227', '0.0848', '0.00014', '0.000121'  |
|  0.687683702                          			| End of no passing   							| '0.687', '0.227', '0.0848', '0.00014', '0.000121'  |
|  0.687683702                             | Speed limit (60km/h)							| '0.687', '0.227', '0.0848', '0.00014', '0.000121'  |
|  0.687683702                 	      		| No entry  					 				          | '0.687', '0.227', '0.0848', '0.00014', '0.000121'  |
|  0.687683702 			| Right-of-way at the next intersection  		           | '0.687', '0.227', '0.0848', '0.00014', '0.000121' |





