package org.apache.spark.mllib.tests

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Test class that lets the user call the program with desired arguments, letting decide
  * the dataset to test (within the ones allowed), and the parameters
  *
  * For a more customizable class letting experiment with other aspects (e.g. regularization), see TestLoopAdaOptimizer
  *
  * Format to call:
  * executable filename updater_type learning_rate mini_batch_fraction num_iterations alg_type
  *
  * filename: adult/madelon/news/susy -- sparse small/?/sparse big/dense big
  * updater_type: 0-9 -- 0: simple, 1: momentum, 2: nesterov, 3: adagrad, 4: adadelta,
  * 5: RMSprop, 6: adam, 7: adamax, 8: nadam, 9: AMSgrad
  * learning_rate: double (0-inf)
  * mini_batch_fraction: double (0-1)
  * num_iterations: int (0-inf)
  * alg_type: lr/svm -- logistic regression or SVM
  *
  * Example of call:
  *
  * executable adult 6 0.2 1 200 lr
  */

object TestDatasets extends App {

  override def main(args: Array[String]): Unit = {

    //Prepare spark variables etc.
    System.setProperty("hadoop.home.dir", "c:/winutil/")
    System.setProperty("spark.sql.warehouse.dir", "file:///C:/spark-warehouse")
    val sc = new SparkContext(new SparkConf().setAppName("TESTADAOPTIMIZER").setMaster("local[*]"))
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    if(args.length != 6){
      throw new IllegalArgumentException("6 arugments are needed: filename - updater - " +
        "learning rate - minibatch fraction - number of iterations - algorithm")}

    val filename = args(0)
    val updater_type = args(1).toInt
    val learning_rate = args(2).toFloat
    val mini_batch_fraction = args(3).toFloat
    val num_iterations = args(4).toInt
    val alg_type = args(5)

    var training: RDD[LabeledPoint] = null
    var testing: RDD[LabeledPoint] = null

    //Load the data selected
    filename match {
      case "adult" =>
        println("adult")
        training = MLUtils.loadLibSVMFile(sc, "data/a9a").repartition(4)
        testing = MLUtils.loadLibSVMFile(sc, "data/a9at")
      case "madelon" =>
        println("madelon")
        training = MLUtils.loadLibSVMFile(sc, "data/madelon-training").repartition(4)
        testing = MLUtils.loadLibSVMFile(sc, "data/madelon-testing")
      case "news" =>
        println("news")
        training = MLUtils.loadLibSVMFile(sc, "data/news-training").repartition(4)
        testing = MLUtils.loadLibSVMFile(sc, "data/news-testing")
      case "susy" =>
        println("susy")
        training = MLUtils.loadLibSVMFile(sc, "data/susy-training").repartition(4)
        testing = MLUtils.loadLibSVMFile(sc, "data/susy-testing")
      case _ => throw new IllegalArgumentException("file not found")
    }

    var updater: AdaptiveUpdater = null

    //Load the updater selected
    updater_type match {
      case 0 =>
        updater = new SimpleUpdater
      case 1 =>
        updater = new MomentumUpdater
      case 2 =>
        updater = new NesterovUpdater
      case 3 =>
        updater = new AdagradUpdater //0.01 learning rate
      case 4 =>
        updater = new AdadeltaUpdater
      case 5 =>
        updater = new RMSpropUpdater // 0.001 learning rate
      case 6 =>
        updater = new AdamUpdater //0.002
      case 7 =>
        updater = new AdamaxUpdater //0.002 learning rate
      case 8 =>
        updater = new NadamUpdater //0.002
      case 9 =>
        updater = new AMSGradUpdater //0.002 like adam
      case _ => throw new IllegalArgumentException("updater not found")
    }

    //Prepare the algorithm selected
    alg_type match {
      case "lr" =>
        val lr = new LogisticRegressionWithAdaSGD()
        lr.optimizer
          .setNumIterations(num_iterations)
          .setStepSize(learning_rate)
          .setUpdater(updater)
          .setMiniBatchFraction(mini_batch_fraction)

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
        println(s"Testing accuracy  of updater $updater_type on alg $alg_type with rate $learning_rate = $accuracy2, training accuracy $accuracy, time elapsed: $elapsedTime millisecond.")

      case "svm" =>
        val svm = new SVMWithAdaSGD()
        svm.optimizer
          .setNumIterations(num_iterations)
          .setStepSize(learning_rate)
          .setUpdater(updater)
          .setMiniBatchFraction(mini_batch_fraction)

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
        println(s"Testing accuracy  of updater $updater_type on alg $alg_type with rate $learning_rate = $accuracy2, training accuracy $accuracy, time elapsed: $elapsedTime millisecond.")
    }

    training.unpersist()
    sc.stop()

  }
}
