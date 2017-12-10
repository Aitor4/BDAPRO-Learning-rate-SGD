package bdapro.project.sgd

import java.io.Serializable

import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.ml.linalg.DenseMatrix
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.classification.impl.GLMClassificationModel
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.linalg.BLAS.dot
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.pmml.PMMLExportable
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.{DataValidators, Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

/**
  * Train a classification model for Binary Logistic Regression
  * using Stochastic Gradient Descent. By default L2 regularization is used,
  * which can be changed via `LogisticRegressionWithSGD.optimizer`.
  *
  * Using LogisticRegressionWithBLHD is recommended over this.
  *
  * @note Labels used in Logistic Regression should be {0, 1, ..., k - 1}
  * for k classes multi-label classification problem.
  */
class LogisticRegressionWithSGD private[mllib] (
                                                 private var stepSize: Double,
                                                 private var numIterations: Int,
                                                 private var regParam: Double,
                                                 private var miniBatchFraction: Double)
  extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {

  private val gradient = new LogisticGradient()
  private val updater = new SquaredL2Updater()
  override val optimizer = new GradientDescent(gradient, updater)
    .setStepSize(stepSize)
    .setNumIterations(numIterations)
    .setRegParam(regParam)
    .setMiniBatchFraction(miniBatchFraction)
  override protected val validators = List(DataValidators.binaryLabelValidator)

  /**
    * Construct a LogisticRegression object with default parameters: {stepSize: 1.0,
    * numIterations: 100, regParm: 0.01, miniBatchFraction: 1.0}.
    */
  def this() = this(1.0, 100, 0.01, 1.0)

  override protected[mllib] def createModel(weights: Vector, intercept: Double) = {
    new LogisticRegressionModel(weights, intercept)
  }
}

/**
  * Top-level methods for calling Logistic Regression using Stochastic Gradient Descent.
  *
  * @note Labels used in Logistic Regression should be {0, 1}
  */
object LogisticRegressionWithSGD {
  // NOTE(shivaram): We use multiple train methods instead of default arguments to support
  // Java programs.

  /**
    * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
    * number of iterations of gradient descent using the specified step size. Each iteration uses
    * `miniBatchFraction` fraction of the data to calculate the gradient. The weights used in
    * gradient descent are initialized using the initial weights provided.
    *
    * @param input RDD of (label, array of features) pairs.
    * @param numIterations Number of iterations of gradient descent to run.
    * @param stepSize Step size to be used for each iteration of gradient descent.
    * @param miniBatchFraction Fraction of data to be used per iteration.
    * @param initialWeights Initial set of weights to be used. Array should be equal in size to
    *        the number of features in the data.
    *
    * @note Labels used in Logistic Regression should be {0, 1}
    */
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             miniBatchFraction: Double,
             initialWeights: Vector): LogisticRegressionModel = {
    new LogisticRegressionWithSGD(stepSize, numIterations, 0.0, miniBatchFraction)
      .run(input, initialWeights)
  }

  /**
    * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
    * number of iterations of gradient descent using the specified step size. Each iteration uses
    * `miniBatchFraction` fraction of the data to calculate the gradient.
    *
    * @param input RDD of (label, array of features) pairs.
    * @param numIterations Number of iterations of gradient descent to run.
    * @param stepSize Step size to be used for each iteration of gradient descent.
    * @param miniBatchFraction Fraction of data to be used per iteration.
    *
    * @note Labels used in Logistic Regression should be {0, 1}
    */
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             miniBatchFraction: Double): LogisticRegressionModel = {
    new LogisticRegressionWithSGD(stepSize, numIterations, 0.0, miniBatchFraction)
      .run(input)
  }

  /**
    * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
    * number of iterations of gradient descent using the specified step size. We use the entire data
    * set to update the gradient in each iteration.
    *
    * @param input RDD of (label, array of features) pairs.
    * @param stepSize Step size to be used for each iteration of Gradient Descent.
    * @param numIterations Number of iterations of gradient descent to run.
    * @return a LogisticRegressionModel which has the weights and offset from training.
    *
    * @note Labels used in Logistic Regression should be {0, 1}
    */
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double): LogisticRegressionModel = {
    train(input, numIterations, stepSize, 1.0)
  }

  /**
    * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
    * number of iterations of gradient descent using a step size of 1.0. We use the entire data set
    * to update the gradient in each iteration.
    *
    * @param input RDD of (label, array of features) pairs.
    * @param numIterations Number of iterations of gradient descent to run.
    * @return a LogisticRegressionModel which has the weights and offset from training.
    *
    * @note Labels used in Logistic Regression should be {0, 1}
    */
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int): LogisticRegressionModel = {
    train(input, numIterations, 1.0, 1.0)
  }
}