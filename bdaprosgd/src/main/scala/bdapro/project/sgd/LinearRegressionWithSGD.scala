package bdapro.project.sgd

import java.io.Serializable

import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.pmml.PMMLExportable
import org.apache.spark.mllib.regression.{GeneralizedLinearAlgorithm, LabeledPoint, LinearRegressionModel}
import org.apache.spark.mllib.regression.impl.GLMRegressionModel
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD

  /**
    * Train a linear regression model with no regularization using Stochastic Gradient Descent.
    * This solves the least squares regression formulation
    *              f(weights) = 1/n ||A weights-y||^2^
    * (which is the mean squared error).
    * Here the data matrix has n rows, and the input RDD holds the set of rows of A, each with
    * its corresponding right hand side label y.
    * See also the documentation for the precise formulation.
    */
  class LinearRegressionWithSGD private[mllib] (
                                                 private var stepSize: Double,
                                                 private var numIterations: Int,
                                                 private var regParam: Double,
                                                 private var miniBatchFraction: Double)
    extends GeneralizedLinearAlgorithm[LinearRegressionModel] with Serializable {

    private val gradient = new LeastSquaresGradient()
    private val updater = new SimpleUpdater()
    override val optimizer = new GradientDescent(gradient, updater)
      .setStepSize(stepSize)
      .setNumIterations(numIterations)
      .setRegParam(regParam)
      .setMiniBatchFraction(miniBatchFraction)

    /**
      * Construct a LinearRegression object with default parameters: {stepSize: 1.0,
      * numIterations: 100, miniBatchFraction: 1.0}.
      */

    def this() = this(1.0, 100, 0.0, 1.0)

    override protected[mllib] def createModel(weights: Vector, intercept: Double) = {
      new LinearRegressionModel(weights, intercept)
    }
  }

  /**
    * Top-level methods for calling LinearRegression.
    *
    */
  object LinearRegressionWithSGD {

    /**
      * Train a Linear Regression model given an RDD of (label, features) pairs. We run a fixed number
      * of iterations of gradient descent using the specified step size. Each iteration uses
      * `miniBatchFraction` fraction of the data to calculate a stochastic gradient. The weights used
      * in gradient descent are initialized using the initial weights provided.
      *
      * @param input RDD of (label, array of features) pairs. Each pair describes a row of the data
      *              matrix A as well as the corresponding right hand side label y
      * @param numIterations Number of iterations of gradient descent to run.
      * @param stepSize Step size to be used for each iteration of gradient descent.
      * @param miniBatchFraction Fraction of data to be used per iteration.
      * @param initialWeights Initial set of weights to be used. Array should be equal in size to
      *        the number of features in the data.
      *
      */
    def train(
               input: RDD[LabeledPoint],
               numIterations: Int,
               stepSize: Double,
               miniBatchFraction: Double,
               initialWeights: Vector): LinearRegressionModel = {
      new LinearRegressionWithSGD(stepSize, numIterations, 0.0, miniBatchFraction)
        .run(input, initialWeights)
    }

    /**
      * Train a LinearRegression model given an RDD of (label, features) pairs. We run a fixed number
      * of iterations of gradient descent using the specified step size. Each iteration uses
      * `miniBatchFraction` fraction of the data to calculate a stochastic gradient.
      *
      * @param input RDD of (label, array of features) pairs. Each pair describes a row of the data
      *              matrix A as well as the corresponding right hand side label y
      * @param numIterations Number of iterations of gradient descent to run.
      * @param stepSize Step size to be used for each iteration of gradient descent.
      * @param miniBatchFraction Fraction of data to be used per iteration.
      *
      */
    def train(
               input: RDD[LabeledPoint],
               numIterations: Int,
               stepSize: Double,
               miniBatchFraction: Double): LinearRegressionModel = {
      new LinearRegressionWithSGD(stepSize, numIterations, 0.0, miniBatchFraction).run(input)
    }

    /**
      * Train a LinearRegression model given an RDD of (label, features) pairs. We run a fixed number
      * of iterations of gradient descent using the specified step size. We use the entire data set to
      * compute the true gradient in each iteration.
      *
      * @param input RDD of (label, array of features) pairs. Each pair describes a row of the data
      *              matrix A as well as the corresponding right hand side label y
      * @param stepSize Step size to be used for each iteration of Gradient Descent.
      * @param numIterations Number of iterations of gradient descent to run.
      * @return a LinearRegressionModel which has the weights and offset from training.
      *
      */
    def train(
               input: RDD[LabeledPoint],
               numIterations: Int,
               stepSize: Double): LinearRegressionModel = {
      train(input, numIterations, stepSize, 1.0)
    }

    /**
      * Train a LinearRegression model given an RDD of (label, features) pairs. We run a fixed number
      * of iterations of gradient descent using a step size of 1.0. We use the entire data set to
      * compute the true gradient in each iteration.
      *
      * @param input RDD of (label, array of features) pairs. Each pair describes a row of the data
      *              matrix A as well as the corresponding right hand side label y
      * @param numIterations Number of iterations of gradient descent to run.
      * @return a LinearRegressionModel which has the weights and offset from training.
      *
      */
    def train(
               input: RDD[LabeledPoint],
               numIterations: Int): LinearRegressionModel = {
      train(input, numIterations, 1.0, 1.0)
    }
  }

