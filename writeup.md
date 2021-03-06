#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/carlosbkm/CarND-Traffic-Sign-Classifier.git) (branch master)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
 
In the second cell there is the code to get some basic summary of the data set:

* The size of training set is ? 34799 images
* The size of test set is ? 12630 images
* The shape of a traffic sign image is ? 32 x 32 pixels
* The number of unique classes/labels in the data set is ? 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook. Ten images from the training set has benn picked randomly and shown in a row.

<img src="https://cloud.githubusercontent.com/assets/4292837/24590850/ce499c24-17f4-11e7-82e3-fa26130bfaeb.png"/>

Also, in the next cell there is a bar chart which shows the classes distribution along the train samples.

<img src="https://cloud.githubusercontent.com/assets/4292837/24590852/ce4a8dbe-17f4-11e7-923c-c07b16db123b.png" />

By the plot it can be seen that the data is highly unbalanced. Just below, in the fith cell, there is a simple code which prints the number of samples per class. Looking at the output we can see that while there are classes with around 2000 samples, more than half classes are below 500. 

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

Using the code in the seventh cell, I applied 3 transformations to the images:
* As a first step, I decided to convert the images to grayscale because getting rid of two channels helps a lot in getting a better processing performance with dealing with a big amount of samples and as it is mentioned in the paper from Pierre Sermanet and Yan LeCunn (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), training with color channels doesn't make any improvement.

* Also, I equalized the brightness of the images using histogram equalization. This helps by making easier to extract the traffic sign features from the images which have very low and high bright values.

* And the most important transformation was to normalize the data, to make the values stay in a range from 0 to 1. I divided them by 255, which is the highest value a b/w image pixel will have. This helps the neural network to learn faster by keeping the weights smaller.

This is how our images look like after preprocessing:

<img src="https://cloud.githubusercontent.com/assets/4292837/24335664/750ac97a-1282-11e7-9ced-8558e64b412f.png"/>

The code for this step is contained in the eight code cell of the IPython notebook. 

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used the data from 'valid.p' which comes with the images download from German Traffic Signs as the cross validation data.

The nineth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because after playing with hyperparameters and normalization, I found that in our specific case of Traffic Signs Classifications an augmentation of the data yields the best improvement in learn accuracy.

First I followed a simple approach of slightly flip images. However, this one proved to be insufficient, and so I also applied another transformations like rotation, translation, shear and brightness. By augmenting the data set this way, the neural network yielded a more stable learning curve. 

My final training set had 179364 number of images. My validation set had 4410 images and the test set 12630 images.

Although the train set samples were augmented almost by 5, the distribution remains unbalanced, so It would be a would a good improvement to increase selectively the samples to make our model more robust.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 15th cell of the ipython notebook. 

I used a LeNet nn like the one presented for the Mnist problem in the Convolutional Networks lesson of the course.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		| Input = 5x5x16. Output = 400    									|
| Fully connected		| Input = 400. Output = 120       									|
| RELU					|												|
| Fully connected		| Input = 120. Output = 84       									|
| RELU					|												|
| Fully connected		| Input = 84. Output = 43      									|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 18th and 19th cell of the ipython notebook. 

To train the model, I used an AdamOptimizer optimizer, which gets better result than Gradient Descent.

Also, I used a batch size of 128. Values from 128 to 256 are usually recommended for this kind of problem. I tried both values, and since a batch of 256 didn't provide any benefit, I stayed with 128.

The tunning of the learning rate is a key aspect in the results obtained by the nn. Is easy to come into an overfitting scenario if you don't choose the right value, so I played a lot with the value until the learning curve of the net was satisfactory.

I started with 10 epochs and increased until 40, with a learning rate of 0.0008. After epoch 13 the network learning curve is almost flat, with a tiny increase in accuracy by epoch. However, I kept it until epoch 40 without running into overfitting.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 20th cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.943 
* test set accuracy of 0.918

If a well known architecture was chosen:
* What architecture was chosen? I chose a LeNet architecture
* Why did you believe it would be relevant to the traffic sign application? The paper from Yan LeCun (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) specifically targets the problem of traffic sign classification, and it has been proven to get good results.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? The values on validation set accuracy and test set accuracy are very close one to another. If the values from the test set accuracy were too low in comparison with the validation set ones, that would mean that we probably are running into an overfitting problem. 

Also, I tried to introduce some variations on the model, like a dropout layer but since I was not really having and overfitting problem I didn't see an improvement and I took it out from the architecture.

Other activation functions were also tested, like elu, but again, I wouldn't get better results.

In conclusion, the final accuracy of the network could be much more improved by heavily augmenting the data set. In my case, I did a very small and subtle augmentation for a matter of time to process and come to the deadline of the project.
 
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I got new images from the German traffic signs site which had been used in some online contests of traffic sign classifier. Then I randomly picked 5 for the test. The images can be downloaded from here: http://benchmark.ini.rub.de/Dataset/GTSRB_Online-Test-Images.zip

<img src="https://cloud.githubusercontent.com/assets/4292837/24590848/ce4876be-17f4-11e7-8905-143def61c34d.png" />

Here are the results of the prediction:

| Image			        |     Prediction	        					|  Difficulty          |
|:---------------------:|:---------------------------------------------:|:---------------------------------------------:|
| No passing for vehicles over 3.5 ton      		| No passing for vehicles over 3.5 ton   									|  Blurred and very dark. Difficult. |
| Right of way at the next intersection     			| Right of way at the next intersection  									| Good visibility. Easy.  |
| Stop					| Stop							|  Highly pixelated. Easy  |
| No passing      		| No passing					 				| Highly pixelated. Easy.  |
| Speed limit (30 km/h)			| Speed limit (30 km/h)	      						| Low brightness . Easy  |

The classes as predicted from the nn:
<img src= https://cloud.githubusercontent.com/assets/4292837/24591043/dbaa501c-17f8-11e7-9481-e9ba7086cf3b.png  />

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set, which was of 91.8%. Although the images used are completely new to the train, validation and test set, they are very similar in format and characteristics to those, and so the classification obtains very good results. 

For this reason we really couldn't conclude whether the model was overfitted or not. It perfoms well with images from the German Traffic Signs site, but it would be necessary to check with images from other sources to check its generalization.

The code for the test on the new images is in the 21st cell of the notebook.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The softmax probabilities of my model can be found in the 23rd cell of the notebook:

<img src="https://cloud.githubusercontent.com/assets/4292837/24336235/a25103f0-128b-11e7-86d7-9dd83cd0ce8a.png" />

In the 24th cell there is the code to print the 5 top probabilities for each class. As it can be seen, there is a very high certainty on the winner class for every traffic sign of the new tested images. All of them have a probability of 99-100 percent. Thi is explained by the similarity of the tested images with the ones which were used to train the net (although different set, all of them were taken form the German Traffic signs site).

<img src= https://cloud.githubusercontent.com/assets/4292837/24590990/bb9a8af4-17f7-11e7-82a8-a9842880ceef.png  />
