# Learning Rate Adaptation Techniques for Stochastic Gradient Descent  

## Öykü Özlem Çakal (@oykut), Aitor Palacios Cuesta (@Aitor4), Andrada Pumnea (@andra-pumnea) 
## Mentored by Behrouz Derakhshan (@d-behi)  
  
Stochastic Gradient Descent (SGD) has become one of the preferred methods for optimizing machine learning algorithms, especially neural networks. It has proved to achieve high performance for large-scale problems, even though the computational complexity of the underlying optimization algorithms is non-trivial. However, the performance of this method heavily depends on the learning rate parameter and how it is tuned. The most basic approach to setting the learning rate is choosing a small constant value or decreasing it gradually throughout the learning. This method has its own downside, as having a too small learning rate can cause the algorithm to converge too slowly, while a too large one may lead to fluctuation or divergence in the loss function.  

The aim of this project is to implement the various algorithms (updaters) that optimize gradient descent, specifically the most recently developed learning rate adaptation techniques on top of Apache Spark. Their implementation is integrated in the context of Linear Support Vector Machines and Logistic Regression. For this purpose, we also implemented and evaluated additional methds not previously present in Spark, such as early stopping, that can affect the behavior of these techinques. We have implemented the following updaters: **Momentum, Nesterov Accelerated Gradient, Adagrad, Adadelta, RMSProp, Adam, AdaMax, Nadam, AMSGrad.**

