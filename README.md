# Learning Rate Adaptation Techniques for Stochastic Gradient Descent in Spark
(Developing)

The purpose of this project is the implementation of 8 different stochastic gradient descent algorithms that employ learning rate adaptation techniques in Spark and the study of their performance using different algorithms (logistic regression, SVM, linear regression...) and different datasets (sparse and dense).  

So far, we have implemented the different algorithms and tested them in a small, sparse dataset (https://archive.ics.uci.edu/ml/datasets/adult). The testing accuracies running on a local machine for the logistic regression algorithm have been the following:  

  
Running until convergence (precision = 0.001)  

Simple SGD: Accuracy = 0.8372950064492353, time elapsed: 21174 milliseconds.  
Momentum: Accuracy = 0.8505620047908605, time elapsed: 22175 milliseconds.  
Nesters: Accuracy = 0.8505620047908605, time elapsed: 23041 milliseconds.  
Adagrad: Accuracy = 0.8501320557705301, time elapsed: 30087 milliseconds.  
Adadelta: Accuracy = 0.8472452552054542, time elapsed: 94333 milliseconds.  
RMSProp: Accuracy = 0.8501320557705301, time elapsed: 91811 milliseconds.  
Adam: Accuracy = 0.8499477919046742, time elapsed: 32701 milliseconds.  
Adamax: Accuracy = 0.8499477919046742, time elapsed: 26042 milliseconds.  
Nadam: Accuracy = 0.8472452552054542, time elapsed: 46271 milliseconds.  
  
Running for 100 iterations:  
  
Simple SGD:  Accuracy = 0.831951354339414, time elapsed: 11854 milliseconds.  
Momentum: Accuracy = 0.8504391622136233, time elapsed: 15012 milliseconds.  
Nesterov: Accuracy = 0.850316319636386, time elapsed: 14052 milliseconds.  
Adagrad: Accuracy = 0.8500092131932928, time elapsed: 13749 milliseconds.   (LR=1)  
Adadelta: Accuracy = 0.7670290522695166, time elapsed: 14163 milliseconds.  
RMSProp: Accuracy = 0.7777777777777778, time elapsed: 16001 milliseconds.  
Adam: Accuracy = 0.8501934770591487, time elapsed: 14132 milliseconds.  
Adamax: Accuracy = 0.8490264725753947, time elapsed: 13864 milliseconds.  
Nadam: Accuracy = 0.811068116209078, time elapsed: 13728 milliseconds.    
  
**It is important to note that the testing accuracies do not have such a meaningful value as when testing a machine learning algorithm. The purpose of optimizers (such as SGD) is to minimize an error function which is given by the machine learning algorithm on the training set. Therefore, the error measure for an optimizer is the accuracy on the training set. Conversely, the error measure for a machine learning algorithm (logistic regression in this case) is the accuracy on the test set. The machine learning algorithm is the part concern with avoiding overfitting. This is done by modifying the error function (e.g. introducing a penalization for having large weights). This in practice modifies the error that the optimizer is trying to minimze. We do not use regularization parameters in our tests because they are not relevant as explained, but they could be easily introduced.**  

The running times on the training set are equivalent to the ones described previously (the work that SGD does is exactly the same). The training accuracies for the logistic regression algorithm running on a local machine have been the following:    

Iterations = inf  
Precision = 0.001  

Simple: Accuracy = 0.8354115479115479  
Momentum: Accuracy = 0.8486486486486486  
Nesterov: Accuracy = 0.8486486486486486   
Adagrad: Accuracy = 0.8338452088452089  
Adadelta: Accuracy = 0.8436117936117936  
RMSprop: Accuracy = 0.8489864864864864  
Adam: Accuracy = 0.846406633906634  
Adamax: Accuracy = 0.8441953316953317  
Nadam:  Accuracy = 0.8464373464373465  
  
Iterations = 100  
Precision = inf  
  
Simple: Accuracy = 0.832463144963145    
Momentum: Accuracy = 0.8481265356265356  
Nesterov: Accuracy = 0.8481265356265356  
Adagrad: Accuracy = 0.8484950859950859 (LR = 1)  
Adadelta: Accuracy = 0.7621621621621621  
RMSprop: Accuracy = 0.8487100737100737 (LR = 0.01)  
Adam: Accuracy = 0.8481879606879607 (LR = 0.02)  
Adamax: Accuracy = 0.8447481572481572 (LR = 0.02)  
Nadam: Accuracy = 0.8481879606879607 (LR = 0.02)  

The training accuracies for the SVM algorithm running on a local machine have been the following:  

Iterations = inf  
Precision = 0.001  
  
Simple: Accuracy = 0.8391584766584766, time elapsed: 24454 millisecond.  
Momentum: Accuracy = 0.8495085995085995, time elapsed: 38851 millisecond.  
Nesterov: Accuracy = 0.8468980343980343, time elapsed: 26712 millisecond. (LR=0.1)  
Adagrad: Accuracy = 0.8497542997542997, time elapsed: 138820 millisecond.  
Adadelta: Accuracy = 0.8464373464373465, time elapsed: 101239 millisecond.  
RMSprop: Accuracy = 0.8492321867321867, time elapsed: 73808 millisecond.  
Adam: Accuracy = 0.8496621621621622, time elapsed: 33204 millisecond.(LR=0.02)  
Adamax: Accuracy = 0.8467444717444718, time elapsed: 27212 millisecond. (LR=0.02)  
Nadam: Accuracy = 0.8496007371007371, time elapsed: 30380 millisecond. (LR=0.02)  

Iterations = 100  
Precision = inf  
  
Simple: Accuracy = 0.8373157248157248, time elapsed: 14577 millisecond.  
Momentum: Accuracy = 0.849017199017199, time elapsed: 15135 millisecond.  
Nesterov: Accuracy = 0.8495085995085995, time elapsed: 15860 millisecond.  
Adagrad: Accuracy = 0.8428746928746929, time elapsed: 13265 millisecond. (LR=1)  
Adadelta: Accuracy = 0.7591830466830467, time elapsed: 12774 millisecond.  
RMSprop: Accuracy = 0.8085687960687961, time elapsed: 13534 millisecond.  
Adam: Accuracy = 0.8468980343980343, time elapsed: 16008 millisecond.  
Adamax: Accuracy = 0.8493243243243244, time elapsed: 14434 millisecond.  
Nadam: Accuracy = 0.8245700245700246, time elapsed: 16277 millisecond.  
  
Except for where specified (LR=X), the step size chosen each algorihtm during the experiments were:   
  
Simple SGD: 1  
Momentum: 1  
Nesterov: 1  
Adagrad: 0.01  
Adadelta: No learning rate  
RMSProp: 0.001  
Adam: 0.002  
Adamax: 0.002  
Nadam: 0.002  
