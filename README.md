# Circle detector 
## Problem definition
The goal in this example is to maximize the overlaps between predicted circle parameters and target circle parameters from noised images. Evaluation is precision @ at least 70% area are overlapped, targeted benchmark is 0.90 precision@0.7.
 
## Loss function
The function to determine the area of two overlapped circles can be found (https://www.xarg.org/2016/07/calculate-the-intersection-points-of-two-circles/). The function is not continuous across the parameter dimensions, there are few conditions (points) that are not differentiable, to demonstrate, let r1 be radius of first circle, let r2 be radius of second circle, let d be distance between two circle centroid:
1. r1 + r2 \< d, no overlap between two circles (or tangent of two circles)
2. |r1 - r2| \< d, two circle intersected 

Although there are many ways to incorporate an non differentiable function in differentiable parameter trainings for example manually define the gradient of each disconnected point. But I'll leave this to future works.

Due to this, I would like to choose a smooth function for our algorithm to optimize. In this case, I choose the loss for shortening the distance between circle parameters (positive correlate with metric). Here I choose two loss functions for two different type of circle parameters:
* euclidean_distance_loss : the eculidean distance between predicted circle centre and target circle centre
* Mean squared error : to calculate the mismatch loss of predicted circle radius and target circle radius, the reason i choose MSE here is because I want to penalize the model to predict prior (average, median etc) and I also pretty sure there's no outliers in the data (I also know the data generation process).

## Network architecture
![CNN](https://user-images.githubusercontent.com/6015707/66263041-48e18180-e7a1-11e9-9ef5-78963c6be7d4.png)

Multi-filter CNN with fully connected layers, "relu" astivation for both CNN and dense and output layers.
Use batch normalization, dropout to improve model performance and training converge speed.
Regarding to CNN layer, I use multi filter CNN for encoding, I choose two filter sizes 3 and 10, the benefits of multi size filters are for example capture different granularity features. 

## Model design
I use the CNN network to map raw image to signals, in this example I defined two signals, one is circle centre, another one is radius length.
I built two models to predict two objectives separately, of course there are many ways to do this like combine multiple losses in same loss function etc, I'll leave this to future work. 
In this example each model is about 1Mb big.

## Model training
I train model with noise level = 2 generated data , data are random sampled from data generator. The sampling process can be improved to shorten the training process, I'll leave this to future work.

## Inference and evaluation
During inference I use the data generator to generate images at noise level 2 to predict circle parameters, two models predicts centre and radius independently. I then pass the circle parameter to `iou` evaluation function to calculate the overlap percentage. 

## Results
With 2 models trained with random generated data, I ran 30 independent experiments, each experiment predicts 100 images, with 30 trails, I calculate bootstrap confidence interval at confidence level 95, the result I got for average precision@0.7 is:

| average precisoin@0.7 | lower 95 ci | upper 95 ci |
| ------------- | ------------- | ------------- |
| 0.917    | 0.9066666666666667 | 0.9273333333333333 |


## Additional results
I also tried a denoising-autoencoder with Hough Transform approach to calculate circle parameters, I trained a denoising autoencoder to denoise the image from noise lvl 2 to normal image, then use Hough Transform to calculate from denoised image.
With the method I achieved 0.95 precision (only tried one configuration)

The value of this approach, I can use this method to generate labels if I dont know the data generate process, then use Hough Generated labels to train CNN classifier (Hough is slow, CNN is faster during inference). By doing this, I "bootstrapped" a machine learning application with no labelled data. 


