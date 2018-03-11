# Learning Rate Adaptation Techniques for Stochastic Gradient Descent  

## Issue 19  
## Öykü Özlem Çakal (@oykut), Aitor Palacios Cuesta (@Aitor4), Andrada Pumnea (@andra-pumnea) 
## Mentored by Behrouz Derakhshan (@d-behi)  
  
Stochastic Gradient Descent (SGD) has become one of the preferred methods for optimizing machine learning algorithms, especially neural networks. It has proved to achieve high performance for large-scale problems, even though the computational complexity of the underlying optimization algorithms is non-trivial. However, the performance of this method heavily depends on the learning rate parameter and how it is tuned. The most basic approach to setting the learning rate is choosing a small constant value or decreasing it gradually throughout the learning. This method has its own downside, as having a too small learning rate can cause the algorithm to converge too slowly, while a too large one may lead to fluctuation or divergence in the loss function.  

The aim of this project is to implement the various algorithms (updaters) that optimize gradient descent, specifically the most recently developed learning rate adaptation techniques, using Apache Spark. Their implementation is evaluated in the context of Linear Support Vector Machines and Logistic Regression. Finally, the performance of the final model is compared both in the case of constant and decreasing learning rate. A thorough analysis is performed on the behavior of the different implemented techniques, which are compared and contrasted. For this purpose, we also implemented and evaluated additional methds not previously present in Spark, such as early stopping, that can affect the behavior of these techinques. We have evaluated the following updaters: **Momentum, Nesterov Accelerated Gradient, Adagrad, Adadelta, RMSProp, Adam, AdaMax, Nadam, AMSGrad.**


Links to our deliverables:  

[Mid-term presentation](Presentations/Midterm/BDAPRO-SGD-Issue19.pdf)  
[Final presentation](Presentations/Final/Final-BDAPRO-SGD-Issue19.pdf)  
[Code](Sources/BDAPRO-Learning-rate-SGD)  
[Report](FinalReports/Report-BDAPRO-SGD-Issue19.pdf)  

Note that the code folder contains the logs resulting from our experiments, plots and a script to generate those plots (results folder), the small version of the datasets used (data folder), and our actual source code (mllib/src folder).
