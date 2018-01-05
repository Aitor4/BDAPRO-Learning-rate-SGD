# Learning Rate Adaptation Techniques for Stochastic Gradient Descent in Spark
(Developing)

The purpose of this project is the implementation of different stochastic gradient descent algorithms in Spark and the study of their performance using different algorithms (logistic regression, SVM, linear regression) and different datasets (sparse and dense).

So far, we have implemented the different algorithms and tested them in a small, sparse dataset (https://archive.ics.uci.edu/ml/datasets/adult). The results running on a local machine have been the following:

Running ntil convergence (precision = 0.001)

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
Adagrad: Accuracy = 0.8500092131932928, time elapsed: 13749 milliseconds.
Adadelta: Accuracy = 0.7670290522695166, time elapsed: 14163 milliseconds.
RMSProp: Accuracy = 0.7777777777777778, time elapsed: 16001 milliseconds.
Adam: Accuracy = 0.8501934770591487, time elapsed: 14132 milliseconds.
Adamax: Accuracy = 0.8490264725753947, time elapsed: 13864 milliseconds.
Nadam: Accuracy = 0.811068116209078, time elapsed: 13728 milliseconds.

The step size chosen each algorihtm during the experiments were:

Momentum: 1
Nesterov: 1
Adagrad: 0.01
Adadelta: No learning rate
RMSProp: 0.001
Adam: 0.002
Adamax: 0.002
Nadam: 0.002
