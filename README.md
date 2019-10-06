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

# Model architecture
![CNN](https://user-images.githubusercontent.com/6015707/66263041-48e18180-e7a1-11e9-9ef5-78963c6be7d4.png)

Multi-filter CNN with fully connected layers, "relu" astivation for both CNN and dense and output layers.
Use batch normalization, dropout to improve model performance and training converge speed.
Regarding to CNN layer, I use multi filter CNN for encoding, I choose two filter sizes 3 and 10, the benefits of multi size filters are for example capture different granularity features. 