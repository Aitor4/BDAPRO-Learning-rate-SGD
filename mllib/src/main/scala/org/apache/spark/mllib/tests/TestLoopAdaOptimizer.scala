package org.apache.spark.mllib.tests

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Test class that executes the selected algorithm for every updater and for every learning rate
  * out of a selected range.
  *
  * The parameters to execute can be changed in the code itself, there are a great variety of them, e.g.:
  * learning rates, algorithm, regularization, early stopping, weigth decay, minibatch fraction, iterations...
  */

object TestLoopAdaOptimizer extends App {

  override def main(args: Array[String]): Unit = {

    //Prepare spark variables etc.
    System.setProperty("hadoop.home.dir", "c:/winutil/")
    System.setProperty("spark.sql.warehouse.dir", "file:///C:/spark-warehouse")
    val sc = new SparkContext(new SparkConf().setAppName("TESTADAOPTIMIZER").setMaster("local[*]"))
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    //Load the data
    val training = MLUtils.loadLibSVMFile(sc, "data/a9a")
    val testing = MLUtils.loadLibSVMFile(sc, "data/a9at")

    //Define the updaters to try
    var updater0 = new SimpleUpdater
    var updater1 = new MomentumUpdater
    var updater2 = new NesterovUpdater
    var updater3 = new AdagradUpdater
    var updater4 = new AdadeltaUpdater
    var updater5 = new RMSpropUpdater
    var updater6 = new AdamUpdater
    var updater7 = new AdamaxUpdater
    var updater8 = new NadamUpdater
    var updater9 = new AMSGradUpdater 
    val updaters = Seq(updater0, updater1, updater2, updater3, updater4, updater5, updater6,
      updater7, updater8, updater9)

    //Define the initial learning rates to try
    val rates = Seq(0.1, 0.1, 0.1, 1.0, 100.0, 0.1, 0.2, 2.0, 0.2, 0.2)
    //Define the variation to try for each initial rate, according to the formula:
    //rate = initial_rate + initial_rate * variation_rate
    //val variation_rates = Seq(9,0,-0.9)
    val variation_rates = Seq(9,6,3,0,-0.3,-0.6,-0.9)

    //Define the algorithm to try, 0=Logistic regression, 1=SVM
    val algorithm = 0

    //Define the parameters for the updaters:
    //Maximum number of iterations to run
    val numIterations = 200
    //Fraction of the dataset to compute the gradient in each iteration
    val miniBatchFraction = 0.25
    //Regularization type, 0:no reg., 1: L1 reg., 2:L2 reg.
    val regType = 0
    //Regularization parameter (only used if not regType=0)
    val regParam = 0
    //Proportion of the training set split used for early stopping. Maximum value 0.5
    //Early stopping only performed if validationSplit>0
    val validationSplit = 0
    //Initial number of iterations to avoid checking the condition of early stopping (to let the optimization "warm up")
    val iterValidation = 50
    //Whether to use weight decay or not
    val decay = false

    //Loop to prepare the optimizer with the appropriate parameters
      if (algorithm==0){ val lr = new LogisticRegressionWithAdaSGD()
        var u = 0
        for(updater<-updaters){
          for (r <- variation_rates) {
            //Choose current rate
            val rate = rates(u) + rates(u) * r
            //Prepare the optimizer
            lr.optimizer
              .setNumIterations(numIterations)
              .setStepSize(rate)
              .setUpdater(updater)
              .setMiniBatchFraction(miniBatchFraction)
              .setRegType(regType)
              .setRegParam(regParam)
              .setIterValidation(iterValidation)
              .setValidationSplit(validationSplit)
              .setDecay(decay)
            //Train measuring training time (for information purposes)
            val currentTime = System.currentTimeMillis()
            //lossHistory contains an array of the losses (can be useful for printing here)
            val (model, lossHistory) = lr.run(training)
            val elapsedTime = System.currentTimeMillis() - currentTime
            // Evaluate on the training set.
            val predictionAndLabels = training.map { case LabeledPoint(label, features) =>
              val prediction = model.predict(features)
              (prediction, label)
            }
            // Get evaluation metrics.
            val metrics = new MulticlassMetrics(predictionAndLabels)
            val accuracy = metrics.accuracy
            //Evaluate on the test set
            val predictionAndLabels2 = testing.map { case LabeledPoint(label, features) =>
              val prediction = model.predict(features)
              (prediction, label)
            }
            // Get evaluation metrics.
            val metrics2 = new MulticlassMetrics(predictionAndLabels2)
            val accuracy2 = metrics2.accuracy

            //Print test and train accuracies and time elapsed
            println(s"Testing accuracy  of updater $u on alg $algorithm with rate $rate = $accuracy2, training accuracy $accuracy, time elapsed: $elapsedTime millisecond.")
          }
          u=u+1
        }
      }
      else {
        val svm = new SVMWithAdaSGD()
        var u = 0
        for (updater <- updaters) {
          for (r <- variation_rates) {
            //Choose current rate
            val rate = rates(u) + rates(u) * r
            //Prepare the optimizer
            svm.optimizer
              .setNumIterations(numIterations)
              .setStepSize(rate)
              .setUpdater(updater)
              .setMiniBatchFraction(miniBatchFraction)
              .setRegType(regType)
              .setRegParam(regParam)
              .setIterValidation(iterValidation)
              .setValidationSplit(validationSplit)
              .setDecay(decay)
            //Train measuring training time (for information purposes)
            val currentTime = System.currentTimeMillis()
            //lossHistory contains an array of the losses (can be useful for printing here)
            val (model, lossHistory) = svm.run(training)
            val elapsedTime = System.currentTimeMillis() - currentTime
            // Evaluate on the training set.
            val predictionAndLabels = training.map { case LabeledPoint(label, features) =>
              val prediction = model.predict(features)
              (prediction, label)
            }
            // Get evaluation metrics.
            val metrics = new MulticlassMetrics(predictionAndLabels)
            val accuracy = metrics.accuracy

            //Evaluate on the test set
            val predictionAndLabels2 = testing.map { case LabeledPoint(label, features) =>
              val prediction = model.predict(features)
              (prediction, label)
            }
            // Get evaluation metrics.
            val metrics2 = new MulticlassMetrics(predictionAndLabels2)
            val accuracy2 = metrics2.accuracy

            //Print test and train accuracies and time elapsed
            println(s"Testing accuracy  of updater $u on alg $algorithm with rate $rate = $accuracy2, training accuracy $accuracy, time elapsed: $elapsedTime millisecond.")
          }
          u = u + 1
        }
      }
    training.unpersist()
    sc.stop()
  }
}

