# Learning Rate Adaptation Techniques for Stochastic Gradient Descent  

## By @Aitor4, @oykut and @andra-pumnea 
## Mentored by @d-behi  
  
Stochastic Gradient Descent (SGD) has become one of the preferred methods for optimizing machine learning algorithms, especially neural networks. It has proved to achieve high performance for large-scale problems, even though the computational complexity of the underlying optimization algorithms is non-trivial. However, the performance of this method heavily depends on the learning rate parameter and how it is tuned. The most basic approach to setting the learning rate is choosing a small constant value or decreasing it gradually throughout the learning. This method has its own downside, as having a too small learning rate can cause the algorithm to converge too slowly, while a too large one may lead to fluctuation or divergence in the loss function.  

The aim of this project is to implement the various algorithms (updaters) that optimize gradient descent, specifically the most recently developed learning rate adaptation techniques on top of Apache Spark. Their implementation is integrated in the context of Linear Support Vector Machines and Logistic Regression. For this purpose, we also implemented and evaluated additional methds not previously present in Spark, such as early stopping, that can affect the behavior of these techinques. We have implemented the following updaters: **Momentum, Nesterov Accelerated Gradient, Adagrad, Adadelta, RMSProp, Adam, AdaMax, Nadam, AMSGrad.**

## Files

You can find the results of our experiments in the [results](results) folder, the plots corresponding to those experiments in the [plots](results/plots) folder, the datasets used for those experiments in the [data](data) folder, and finally our source code in the [mllib](mllin) folder.

## Experiment conclusions

The conclusions of our experiments are the following:

* All the learning rate adaptation techniques except Adadelta have, in general, a performance superior to the simple one. RMSprop might be the weakest one during our experiments, but as we have seen, for one dataset it results in the best performance, and in no experiments it shows much worse performance than the simple one. These observations happen for varying conditions (different datasets and algorithms) so they can be considered a general property.\\

* The specific comparative results between the implemented updaters are problem-dependent. This is not surprising since many studies and literature show how different datasets corresponding to different problems have a key impact on what updater, algorithm and related machine learning conditions work better. However, we can state that the algorithms adapted for a good performance with sparse datasets (except Adadelta and sometimes RMSprop) perform better with a larger difference to the baseline when optimizing on sparse datasets than when doing it on dense ones (even though they also show better results than the simple one there).\\

* Following from the previous point, there is no technique which is superior over all the others for every problem. However, there are some which consistently have results close to the best one, do not present divergent behavior across iterations, are specifically adapted to deal with sparse data and are less sensitive to the exact initial learning rate set as a hyperparameter. We have observed in our experiments that AMSGrad, Adam and Adamax have this behaviour. Also Nadam in general has similar performance, except its divergence in section \ref{susy}. However, this might be an "accident" of one particular experiment, since Nadam's definition is quite similar to these other 3 techniques.\\

* If we were to run an algorithm on a new dataset, two optimization approaches are possible. First, run experiments with every updater (maybe on a subsample of the entire dataset) with different learning rates to find out which updater performs better. Then, use that configuration on the final optimization to obtain our model. This, in practice, means executing the same experiments ten times, one per updater (each of which probably is repeated several times due to other hyperparameters not related to the optimization process). This can be prohibitive in some cases or not worth the potential improvement in optimization in others as compared to other possible improvements coming from the machine learning algorithm itself or larger computation time that would be available without that process. The second option is to use just one updater, either trying some different learning rates or, in the extreme case, just using the recommended learning rate for that particular updater, and hoping that its results are reasonable and not far from the optimal ones. In this case, we would recommend to use one of the techniques mentioned above which show consistently good performance. As a conclusion, there is a trade-off between how many experiments to execute and how optimal we can expect our optimization process to be, as it is the case in many other machine learning hyperparameters.
