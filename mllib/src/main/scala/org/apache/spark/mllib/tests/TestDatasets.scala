package org.apache.spark.mllib.tests

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}



object TestDatasets extends App {

  override def main(args: Array[String]): Unit = {

    System.setProperty("hadoop.home.dir", "c:/winutil/")
    System.setProperty("spark.sql.warehouse.dir", "file:///C:/spark-warehouse")
    val sc = new SparkContext(new SparkConf().setAppName("TESTADAOPTIMIZER").setMaster("local[*]"))
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    if(args.length != 6){
      throw new IllegalArgumentException("6 arugments are needed: filename - updater - " +
        "learning rate - minibatch fraction - number of iterations - algorithm")
    }
    else {
        val filename = args(0)
        val updater_type = args(1).toInt
        val learning_rate = args(2).toFloat
        val mini_batch_fraction = args(3).toFloat
        val num_iterations = args(4).toInt
        val alg_type = args(5)
      }
    var training: RDD[LabeledPoint] = null
    var testing: RDD[LabeledPoint] = null

    filename match {
      case "adult" =>
        println("adult")
        training = MLUtils.loadLibSVMFile(sc, "data/a9a").repartition(4)
        testing = MLUtils.loadLibSVMFile(sc, "data/a9at")
      case "madelon" =>
        println("adult")
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

    alg_type match {
      case "lr" =>
        val lr = new LogisticRegressionWithAdaSGD()
        lr.optimizer
          .setNumIterations(num_iterations)
          .setConvergenceTol(0.001)
          .setStepSize(learning_rate)
          .setUpdater(updater)
          .setMiniBatchFraction(mini_batch_fraction)
          .setDecay(true)
        //.setRegType(1)
        //.setRegParam(0.1)
        val currentTime = System.currentTimeMillis()
        val (model, lossHistory) = lr.run(training)
        val elapsedTime = System.currentTimeMillis() - currentTime
        // Compute raw scores on the training set.
        val predictionAndLabels = testing.map { case LabeledPoint(label, features) =>
          val prediction = model.predict(features)
          (prediction, label)
        }
        // Get evaluation metrics.
        val metrics = new MulticlassMetrics(predictionAndLabels)
        val accuracy = metrics.accuracy
        /*for (p <- (1 to lossHistory.size)){
          println("Loss on iteration "+p+" : "+lossHistory(p-1))
        }*/
        println(s"Test accuracy  of updater $updater_type on alg $alg_type with rate $learning_rate = $accuracy, time elapsed: $elapsedTime millisecond.")
      case "svm" =>
        val svm = new SVMWithAdaSGD()
        svm.optimizer
          .setNumIterations(num_iterations)
          .setConvergenceTol(0.001)
          .setStepSize(learning_rate)
          .setUpdater(updater)
          .setMiniBatchFraction(mini_batch_fraction)

        val currentTime = System.currentTimeMillis()
        val (model, lossHistory) = svm.run(training)
        val elapsedTime = System.currentTimeMillis() - currentTime
        // Compute raw scores on the training set.
        val predictionAndLabels = testing.map { case LabeledPoint(label, features) =>
          val prediction = model.predict(features)
          (prediction, label)
        }
        // Get evaluation metrics.
        val metrics = new MulticlassMetrics(predictionAndLabels)
        val accuracy = metrics.accuracy
        /*for (p <- (1 to lossHistory.size)){
              println("Loss on iteration "+p+" : "+lossHistory(p-1))
            }*/
        println(s"Test accuracy  of updater $updater_type on alg $alg_type with rate $learning_rate = $accuracy, time elapsed: $elapsedTime millisecond.")

    }

    training.unpersist()
    sc.stop()

  }
}
