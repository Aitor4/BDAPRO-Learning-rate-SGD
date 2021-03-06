
package org.apache.spark.mllib.classification

import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD



/**
  * Classification model trained using Binary Logistic Regression with an adaptive updater
  * and extending our GeneralizedLinearAlgorithmWithAdaSGD class
  *
  *  The class is based on the trait "LogisticRegression" from org.apache.spark.mllib.classification
  *  Most of the comments and code are from the original class
  *
  * @param stepSize - The step size (learning rate) to be used
  * @param numIterations - maximum number of iterations to train
  * @param regParam - the regularization parameter to use
  * @param miniBatchFraction - the fraction of the batch to compute at every iteration
  * @param updater - the adaptive updater to be used to compute the weight updates
  */

class LogisticRegressionWithAdaSGD private[mllib](
                                                 private var stepSize: Double,
                                                 private var numIterations: Int,
                                                 private var regParam: Double,
                                                 private var miniBatchFraction: Double,
                                                 private var updater: AdaptiveUpdater)
  extends GeneralizedLinearAlgorithmWithAdaSGD[LogisticRegressionModel] with Serializable {

  private val gradient = new LogisticGradient()

  //optimizer with default values
  override val optimizer = new GradientDescentAlg(gradient, updater)
    .setStepSize(stepSize)
    .setNumIterations(numIterations)
    .setRegParam(regParam)
    .setMiniBatchFraction(miniBatchFraction)

  override protected val validators = List(DataValidators.binaryLabelValidator)
  /**
    * Construct a LogisticRegression object with default parameters: {stepSize: 1.0,
    * numIterations: 100, regParm: 0, miniBatchFraction: 1.0}.
    */
  @Since("2.1.0")
  def this() = this(1.0, 100, 0, 1.0, new AdagradUpdater)

  override protected[mllib] def createModel(weights: Vector, intercept: Double) = {
    new LogisticRegressionModel(weights, intercept)
  }
}

/**
  * Top-level methods for calling Logistic Regression.
  * NOTE: Labels used in Logistic Regression should be {0, 1}
  */
@deprecated("Use ml.classification.LogisticRegression or LogisticRegressionWithLBFGS", "2.0.0")
object LogisticRegressionWithAdaSGD {
  // NOTE(shivaram): We use multiple train methods instead of default arguments to support
  // Java programs.

  /**
    * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
    * number of iterations of gradient descent using the specified step size. Each iteration uses
    * `miniBatchFraction` fraction of the data to calculate the gradient. The weights used in
    * gradient descent are initialized using the initial weights provided.
    * NOTE: Labels used in Logistic Regression should be {0, 1}
    *
    * @param input RDD of (label, array of features) pairs.
    * @param numIterations Number of iterations of gradient descent to run.
    * @param stepSize Step size to be used for each iteration of gradient descent.
    * @param miniBatchFraction Fraction of data to be used per iteration.
    * @param initialWeights Initial set of weights to be used. Array should be equal in size to
    *        the number of features in the data.
    */
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             miniBatchFraction: Double,
             initialWeights: Vector,
             updater: AdaptiveUpdater): LogisticRegressionModel = {
    new LogisticRegressionWithAdaSGD(stepSize, numIterations, 0.0, miniBatchFraction, updater)
      .run(input, initialWeights)._1
  }

  /**
    * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
    * number of iterations of gradient descent using the specified step size. Each iteration uses
    * `miniBatchFraction` fraction of the data to calculate the gradient.
    * NOTE: Labels used in Logistic Regression should be {0, 1}
    *
    * @param input RDD of (label, array of features) pairs.
    * @param numIterations Number of iterations of gradient descent to run.
    * @param stepSize Step size to be used for each iteration of gradient descent.

    * @param miniBatchFraction Fraction of data to be used per iteration.
    */
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             miniBatchFraction: Double,
             updater: AdaptiveUpdater): LogisticRegressionModel = {
    new LogisticRegressionWithAdaSGD(stepSize, numIterations, 0.0, miniBatchFraction, updater)
      .run(input)._1
  }

  /**
    * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
    * number of iterations of gradient descent using the specified step size. We use the entire data
    * set to update the gradient in each iteration.
    * NOTE: Labels used in Logistic Regression should be {0, 1}
    *
    * @param input RDD of (label, array of features) pairs.
    * @param stepSize Step size to be used for each iteration of Gradient Descent.

    * @param numIterations Number of iterations of gradient descent to run.
    * @return a LogisticRegressionModel which has the weights and offset from training.
    */
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             updater: AdaptiveUpdater): LogisticRegressionModel = {
    train(input, numIterations, stepSize, 1.0, updater)
  }

  /**
    * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
    * number of iterations of gradient descent using a step size of 1.0. We use the entire data set
    * to update the gradient in each iteration.
    * NOTE: Labels used in Logistic Regression should be {0, 1}
    *
    * @param input RDD of (label, array of features) pairs.
    * @param numIterations Number of iterations of gradient descent to run.
    * @return a LogisticRegressionModel which has the weights and offset from training.
    */
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             updater: AdaptiveUpdater): LogisticRegressionModel = {
    train(input, numIterations, 1.0, 1.0, updater)
  }
}

