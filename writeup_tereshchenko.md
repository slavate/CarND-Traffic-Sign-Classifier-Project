# **Traffic Sign Recognition** 
## Viacheslav Tereshchenko

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following: 

- Load the data set (see below for links to the project data set) 
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/dataset/sl_80_17680.jpg "Speed Limit 80"
[image2]: ./examples/dataset/sl_30_3897.jpg "Speed Limit 30"
[image3]: ./examples/dataset/truck_prohibited_54809.jpg "Vehicles over 3.5 metric tons prohibited"
[image4]: ./examples/dataset/d_curve_left_65856.jpg "Dangerous curve to the left"
[image5]: ./new_images/test/no-sign-3.jpg
[image6]: ./new_images/test/sl-20.jpg 
[image7]: ./new_images/test/sl-80.jpg
[image8]: ./new_images/test/sl-80-1.jpg
[image9]: ./new_images/test/sl-el-80.jpg
[image10]: ./new_images/test/stop-sign.jpg
[image11]: ./new_images/test/yield-sign.jpg

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

The code for this step is contained in the second and third code cells of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

- Number of training examples = 34799
- Number of validation examples = 4410
- Number of testing examples = 12630
- Image data shape = (32, 32, 3)
- The number of unique classes/labels of classes = 43

Moreover I have implemented function which saves number of samples for each unique class to csvfile. This function helped me further for dataset analysis. 

#### 2. Exploratory visualization of the dataset and.

The code for this step is contained in the code cells Nr. 4 and 5 of the IPython notebook.  
In this section I visualize a random selected image with its label and amount of samples in the dataset. For further analysis image can be saved to the folder \examples\dataset. In the following 4 images one can see some examples of training data and make a conclusion that dataset includes images from different light conditions.  

![alt text][image1]![alt text][image2]![alt text][image3]![alt text][image4] 

### Design and Test a Model Architecture

#### 1. Data preprocessing

The code for this step is contained in the sixth and ninth code cell of the IPython notebook.
At first I have implemented three functions: **"normalize_dataset", "normalize" and "histequalize"**.

- **normalize_dataset**: This function is called if the whole dataset should be preprocessed. 


- **normalize**: This function normalizes brightness of the image. And afterwards it calls the function **histequalize**. This function also can grayscale image, which is done depending on the input parameter "rgb". At the end of this function the pixel values of the image are normalized to the range from -0.5 to 0.5, due to numerical stability. 


- **histequalize**: In this function I am doing histogram equalization of the image, after it the contrast of the image should be normalized.


I've tried to learn my CNN model with both grayscale and rgb images, the accuracy performance was almost the same. Additionaly rgb images contain color information, so I decided to proceed with rgb images.


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Originally three datasets sets were provided (see section 1 "Basic summary of the data set").


My final training set had 168833 number of images. My validation set and test set had 4410 and 12630 number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because original training dataset contains much more less samples for some classes, e.g. "Speed limit (20km/h)" has only 180 examples in comparison to "Speed limit (50km/h)", which has 2010. That means that during trainig process CNN may not learn features for signs with not enough samples.  
To add more data to the the data set, I used keras function "ImageDataGenerator". I decided only to shift and rotate (minor rotation) the images. Afterwards I saved augmented data to the new pickle file "train_augmented.p".

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB or 32x32x1 grayscale image  		| 
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 32x32x3 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x12 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x24   |
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x24 				    |
| Fully connected 1		| inputs 600, outputs 200           			|
| RELU					|												|
| Dropout				|												|
| Fully connected 2		| inputs 200, outputs 100           			|
| RELU					|												|
| Dropout				|												|
| Fully connected 3		| inputs 100, outputs 43	           			|
| logits                |                                      	        |



#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the ninth and tenth cell of the ipython notebook. My model architecture is based on LeNet model. 

To train the model, I used an AdamOptimizer. I also tried GradientDescentOptimizer however its performance with the same parameters was worser. Moreover Adam stays for "Adaptive Moment Estimation", i.e. this optimizer computes adaptive learning rate for each parameter. Thats why I continued to use Adamoptimizer. Following parameters worked for me the best: maximum number of epochs = 20, batch size = 128, starting learning rate = 1e-2 and lowering it.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eighth cell of the Ipython notebook.

My final model results were:

* training set accuracy of 99,2%
* validation set accuracy of 96,9% 
* test set accuracy of 94,6%

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen? 
	* I choosed LeNet architecture because it was proposed in the class. 
* What were some problems with the initial architecture?
	* I should only adjust the input depth to 3, if I were using RGB images.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
	* I tried to change almost all the parameters such as: filter size for convolutional layers 1 and 2. 3x3 filter worked out worser than 5x5. Moreover I tried to increase the output depth for convulutional layers. I tried different sizes of fully conected layers. I added additional convolution layers and so called "one by one" convolutional layers.   
* Which parameters were tuned? How were they adjusted and why?
	* learning rate, dropout probability.
* What are some of the important design choices and why were they chosen?
	* Increasing output depth of the convolutional layers improved the model accuracy. However it also increased training time and memory consumation. Dropout 
	*  For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:

* What architecture was chosen?
	* LeNet
* Why did you believe it would be relevant to the traffic sign application?
	* Because it worked in the previous lab for classifying handwritten numbers, i.e. it already worked on image classification problem. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
	* accuracy values for all 3 data sets are high enough, moreover I looked at precision and recall for each sign type (see Excel-Documents).   
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10]
![alt text][image11]

The first image does not contain any sign, since the model was trained only for classification of 43 sign types it will definetly classified wrongly. However it is interesting to see which prediciton values will be outputed. Electronic sign was also not learned, so I am not expecting that it will be correctly classified.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the eighth cell of the Ipython notebook (function "classify").

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| no sign      			| Stop   										| 
| sl-20      			| Speed limit (20km/h)   						| 
| sl-80     			| Speed limit (80km/h) 							|
| sl-80-1				| Speed limit (30km/h)							|
| sl-el-80      		| Speed limit (70km/h)   						|
| stop-sign	      		| Stop					 						|
| yield-sign			| Yield      									|


The model was able to correctly guess 4 of the 7 traffic signs. Since 2 types of the images were not trained ("no-sign" and "elctronic sign") I will not calculate them in accuracy, i.e. 4 of 5 images were correctly classified.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

For the first image, the model is sure that this is a stop sign or no entry sign and the image does not contain any sign. The top five soft max probabilities were.
label: no-sign-3.jpg, prediciton: Stop

* confidence = 0.519 for Stop
* confidence = 0.475 for No entry
* confidence = 0.003 for End of no passing
* confidence = 0.001 for End of speed limit (80km/h)
* confidence = 0.001 for Vehicles over 3.5 metric tons prohibited

The second image was correcty classified.
label: sl-20.jpg, prediciton: Speed limit (20km/h)

* confidence = 1.000 for Speed limit (20km/h)
* confidence = 0.000 for Speed limit (60km/h)
* confidence = 0.000 for Speed limit (30km/h)
* confidence = 0.000 for Speed limit (120km/h)
* confidence = 0.000 for Speed limit (80km/h)

The third image was correcty classified.
label: sl-80-1.jpg, prediciton: Speed limit (80km/h)

* confidence = 0.733 for Speed limit (80km/h)
* confidence = 0.147 for Speed limit (100km/h)
* confidence = 0.120 for Speed limit (30km/h)
* confidence = 0.001 for Speed limit (50km/h)
* confidence = 0.000 for Speed limit (60km/h)

The fourth image was sure that this is a speed limit 30 sign, however this is one speed limit 80 sign.
label: sl-80.jpg, prediciton: Speed limit (30km/h)

* confidence = 1.000 for Speed limit (30km/h)
* confidence = 0.000 for Go straight or right
* confidence = 0.000 for End of speed limit (80km/h)
* confidence = 0.000 for Speed limit (20km/h)
* confidence = 0.000 for Speed limit (50km/h)

The fifth image was sure that this is a speed limit 70 sign, however this is one electronic speed limit 80 sign.
label: sl-el-80.jpg, prediciton: Speed limit (70km/h)

* confidence = 0.813 for Speed limit (70km/h)
* confidence = 0.107 for Speed limit (120km/h)
* confidence = 0.030 for Roundabout mandatory
* confidence = 0.016 for Speed limit (30km/h)
* confidence = 0.010 for Speed limit (80km/h)

The 6th image was correcty classified.
label: stop-sign.jpg, prediciton: Stop

* confidence = 1.000 for Stop
* confidence = 0.000 for Speed limit (20km/h)
* confidence = 0.000 for Children crossing
* confidence = 0.000 for Priority road
* confidence = 0.000 for Bicycles crossing

The 7th image was correcty classified.
label: yield-sign.jpg, prediciton: Yield

* confidence = 1.000 for Yield
* confidence = 0.000 for Priority road
* confidence = 0.000 for End of no passing by vehicles over 3.5 metric tons
* confidence = 0.000 for No passing for vehicles over 3.5 metric tons
* confidence = 0.000 for General caution