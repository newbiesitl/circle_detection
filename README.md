# Circle detector 
## Problem definition
In this example the objective is to maximize the overlaps between predicted circle parameters and target circle parameters, the circle parameter is prediceted from noised images. Evaluation metric precision of at least 70% area of predicted circle and target circle are overlapped, targeted benchmark is 0.90 precision@0.7.
 
## Loss function
The method to determine the area of two overlapped circles can be found at (https://www.xarg.org/2016/07/calculate-the-intersection-points-of-two-circles/). The function is not continuous across the parameter dimensions, there are few conditions (points) that are not differentiable, to demonstrate, let r1 be radius of first circle, let r2 be radius of second circle, let d be distance between two circle centroid:
1. r1 + r2 \< d, no overlap between two circles (or tangent of two circles)
2. |r1 - r2| \< d, two circle intersected 

Although there are many ways to incorporate an non differentiable function in differentiable parameter trainings for example manually define the gradient of each disconnected point. But I'll leave this to future works.

Due to this, I would like to choose a smooth function for our algorithm to optimize. In this case, I choose the loss for shortening the distance between circle parameters (positive correlate with metric). Here I choose two loss functions for two different type of circle parameters:
* euclidean_distance_loss : the eculidean distance measures the distance between predicted circle centre and target circle centre
* Mean squared error : MSE is used to match predicted circle radius with target circle radius, the reason i choose MSE here is because I want to penalize the model to predict prior (average, median etc) and I also pretty sure there's no outliers in the data (I also know the data generation process).

## Network architecture
![CNN](https://user-images.githubusercontent.com/6015707/66263041-48e18180-e7a1-11e9-9ef5-78963c6be7d4.png)

Multi-filter CNN with fully connected layers, "relu" astivation for both CNN and dense and output layers. Batch normalization and dropout are used to improve model performance and training convergence speed.
Regarding to CNN layer, I use multi filter CNN for encoding, I choose two filter sizes 3 and 10, the benefits of multi size filters are multi folded, for example one benefit is multiple size filters capture different granularity features. 

## Model design
I use the CNN network to map raw image to signals, in this example I defined two signals, one is circle centre, another one is radius length.
I built two models to predict two objectives independently, of course there are many ways to do this like combining multiple losses in same loss function etc, I'll leave this to future work. 
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

So I'm 95% confidence that this predictor will achieve precision between 0.90 and 0.92 on real population / deployed in production (if data generation process doesn't change, there's method to detect that, I'll leave this for future discussion)

## Additional results
I also tried a denoising-autoencoder with Hough Transform approach to calculate circle parameters, I trained a denoising autoencoder to denoise the image from noise lvl 2 to normal image, then use Hough Transform to calculate from denoised image.
With the method I achieved 0.95 precision (only tried one configuration)

The value of this approach is I can use this method to generate labels if I dont know the data generate process, I use Hough Generated labels to train CNN classifier (people may ask why not just use Hough? There are many reasons, first there's no GPU implementation of that, hard to speed up with GPU hardwards, 2nd Hough is slow, CNN is faster during inference). By doing this, I "bootstrapped" a machine learning application with no labelled data. 

## Future works
1. Improve loss function, it's always good to optimize the metric directly, and in this example, training loss is not linear correlated with problem metric, in make the optimization process simpler, I can convert the problem metric to training metric, there are few ways to do it, one way is to use TF to define the gradient of discontinuous points (https://www.tensorflow.org/guide/create_op). 
2. Improve sampling strategy to speed up training process, I can use the trained network to produce a distribution of prediction using for example Monte Carlo Dropout [1]. I can use some statistics of prediction distribution to rank unlabelled data for labelling, one statistic i can use is the standard deviation of prediction, for example in this case I can produce the prediction in the form of:
    1. circle centre x:  25+-2.5
    2. circle radius: 17 +- 10
 and I only query the label for high uncertainty labels (high std) to train the model, in NLP this will speed up training (save labels) up to 90%.
 
 
 
## How to run
- Unzip the file
- Install Anaconda
- `cd circle_detection`
- `conda create -n circle_detect python=3.5`
- `conda activate circle_detect`
- `pip install -r requirements.txt`
- If you want to train a model from scratch: `python build_models.py`
- Run evaluation `python main.py`
 
 
 
 [1]: 
Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. 2016."
