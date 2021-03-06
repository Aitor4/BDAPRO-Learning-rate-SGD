package org.apache.spark.mllib.optimization

import breeze.linalg.{norm, DenseVector => BDV}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import breeze.numerics.{sqrt => sqrt}
import org.apache.spark.mllib.linalg.BLAS.{axpy, dot, scal}

/**
  * Class used to solve an optimization problem using Gradient Descent.
  * Based on the original class "GradientDescent" from org.apache.spark.mllib.optimization.
  * Some comments are from the original class, we therefore indicate our comments with
  * "::Extension::" in the beginning
  *
  * @param gradient Gradient function to be used.
  * @param updater Updater to be used to update weights after every iteration.
  */
class GradientDescentAlg private[spark](
                                         private var gradient: Gradient,
                                         private var updater: AdaptiveUpdater)
  extends OptimizerWithAdaSGD with Logging {

  private var learningRate: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
  private var convergenceTol: Double = 0.001
  // ::Extension:: parameters of different updaters initialized to their default recommended values
  private var momentumFraction: Double = 0.9
  private var smoothingTerm :Double = 0.00000001
  private var beta : Double = 0.9
  private var betaS : Double = 0.999
  // ::Extension:: by default, no regularization nor weight decay is performed
  private var regType: Int = 0
  private var decay :Boolean = false
  // ::Extension:: variables used to perform early stopping (if validationSplit is set to >0)
  private var validationSplit:Double = 0
  private var iterValidation:Int = 50

  /**
    * Set the initial step size for the first step. Default 1.0.
    */
  def setStepSize(step: Double): this.type = {
    require(step > 0,
      s"Initial step size must be positive but got ${step}")
    this.learningRate = step
    this
  }

  /**
    * Set fraction of data to be used for each iteration.
    * Default 1.0 (corresponding to deterministic/classical gradient descent)
    */
  def setMiniBatchFraction(fraction: Double): this.type = {
    require(fraction > 0 && fraction <= 1.0,
      s"Fraction for mini-batch must be in range (0, 1] but got ${fraction}")
    this.miniBatchFraction = fraction
    this
  }

  /**
    * Set the number of iterations. Default 100.
    */
  def setNumIterations(iters: Int): this.type = {
    require(iters >= 0,
      s"Number of iterations must be nonnegative but got ${iters}")
    this.numIterations = iters
    this
  }

  /**
    * Set the regularization parameter. Default 0.0.
    */
  def setRegParam(regParam: Double): this.type = {
    require(regParam >= 0,
      s"Regularization parameter must be nonnegative but got ${regParam}")
    this.regParam = regParam
    this
  }

  /**
    * Set the convergence tolerance. Default 0.001
    * convergenceTol is a condition which decides iteration termination.
    * The end of iteration is decided based on below logic.
    *
    *  - If the norm of the new solution vector is >1, the diff of solution vectors
    *    is compared to relative tolerance which means normalizing by the norm of
    *    the new solution vector.
    *  - If the norm of the new solution vector is <=1, the diff of solution vectors
    *    is compared to absolute tolerance which is not normalizing.
    *
    * Must be between 0.0 and 1.0 inclusively.
    */
  def setConvergenceTol(tolerance: Double): this.type = {
    require(tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
    this.convergenceTol = tolerance
    this
  }

  /**
    * Set the gradient function (of the loss function of one single data example)
    */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
    * Set the updater function to actually perform a gradient step in a given direction.
    * The updater is responsible to perform the update from the regularization term as well,
    * and therefore determines what kind or regularization is used, if any.
    */
  def setUpdater(updater: AdaptiveUpdater): this.type = {
    this.updater = updater
    this
  }

  /**
    * ::Extension:: Set the smoothing term. Default 1*e-8
    */
  def setSmoothingTerm(term: Double): this.type = {
    this.smoothingTerm = term
    this
  }
  /**
    * ::Extension:: Set the beta parameter. Default 0.9
    */
  def setBeta(b: Double): this.type = {
    require(beta > 0 && beta <= 1.0,
      "Beta for mini-batch SGD must be in range (0, 1] but got ${beta}")
    this.beta = b
    this
  }

  /**
    * ::Extension:: Set the beta2 (beta squared) parameter. Default 0.999
    */
  def setBetaS(bS: Double): this.type = {
    require(betaS > 0 && betaS <= 1.0,
      "Beta squared for mini-batch SGD must be in range (0, 1] but got ${betaS}")
    this.betaS = bS
    this
  }

  /**
    * ::Extension:: Set the momentum fraction. Default 0.9
    * It is also the variable used for "rho" in adadelta since it has equivalent semantics
    */
  def setMomentumFraction(momentumFraction:Double) : this.type  = {
    require(momentumFraction > 0 && momentumFraction <= 1.0,
      "Momentum fraction for mini-batch SGD must be in range (0, 1] but got ${momentumFraction}")
    this.momentumFraction = momentumFraction
    this
  }

  /**
    * ::Extension:: Set the decay variable. Default false
    * It indicates whether to decay the initial learning rate dividing it by the square root
    *  of the number of iterations
    */
  def setDecay(decay: Boolean): this.type = {
    this.decay = decay
    this
  }

  /**
    * ::Extension:: Set the regularization type. Default 0
    * 0 indicates no regularization, 1 indicates L1 regularization, 2 indicates L2 regularization
    */
  def setRegType(regType: Int): this.type = {
    require(regType >= 0 && regType <= 2,
      "regType must be = (no regularization), 1 (L1), or 2 (L2)")
    this.regType = regType
    this
  }

  /**
    * ::Extension:: Set the proportion of the data used for validation and early stopping.
    * Default value: 0. If the value is not 0, early stopping is performed with that proportion of
    * the data passed to the algorithm as a validation set.
    */
  def setValidationSplit(split: Double): this.type = {
    require(split >= 0 && split <= 0.5,
      "validationSplit must be >= 0 and <= 0.5")
    this.validationSplit = split
    this
  }

  /**
    * ::Extension:: Set the number of iterations which are ignored during early stopping
    * Default value: 50. The early stopping condition will only be checked after that many iterations
    */
  def setIterValidation(iter: Int): this.type = {
    require(iter >= 1,
      "iterValidation must be >= 1")
    this.iterValidation = iter
    this
  }

  /**
    * :: DeveloperApi ::
    * Runs gradient descent on the given training data.
    *
    * @param data training data
    * @param initialWeights initial weights
    * @return solution vector
    */
  @DeveloperApi
  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): (Vector, Array[Double]) = {
    val (weights, loss) = GradientDescentAlg.runMiniBatch(
      data,
      gradient,
      updater,
      momentumFraction,
      learningRate,
      numIterations,
      regParam,
      miniBatchFraction,
      initialWeights,
      smoothingTerm,
      beta,
      betaS,
      convergenceTol,
      regType,
      decay,
      validationSplit,
      iterValidation)
    (weights, loss)
  }

}

/**
  * :: DeveloperApi ::
  * Top-level method to run gradient descent.
  */
@DeveloperApi
object GradientDescentAlg extends Logging {
  /**
    * ::Extension:: Including the appropriate variables we defined
    *
    * In each iteration, we sample a subset (fraction miniBatchFraction) of the total data
    * in order to compute a gradient estimate.
    * Sampling, and averaging the subgradients over this subset is performed using one standard
    * spark map-reduce in each iteration.
    *
    * @param data Input data. RDD of the set of data examples, each of
    *             the form (label, [feature values]).
    * @param gradient Gradient object (used to compute the gradient of the loss function of
    *                 one single data example)
    * @param updater Updater function to actually perform a gradient step in a given direction.
    * @param learningRate initial step size for the first step
    * @param numIterations number of iterations.
    * @param regParam regularization parameter
    * @param miniBatchFraction fraction of the input data set that should be used for
    *                          one iteration. Default value 1.0.
    * @param smoothingTerm - the smoothing term to be used to avoid division by 0 in the corresponding updaters
    * @param beta - the beta parameter to be used in the corresponding updaters
    * @param betaS - the beta2 (beta squared) parameter to be used in the correspodning updaters
    * @param convergenceTol Minibatch iteration will end before numIterations if the relative
    *                       difference between the current weight and the previous weight is less
    *                       than this value. In measuring convergence, L2 norm is calculated.
    *                       Default value 0.001. Must be between 0.0 and 1.0 inclusively.
    * @param regType - the type of regularization to be applied
    * @param decay - whether to apply weight decay or not
    * @param validationSplit - the proportion of the data that is used for validation and early stopping
    * @param iterValidation - how many initial iterations are avoided for early stopping
    * @return A tuple containing two elements. The first element is a column matrix containing
    *         weights for every feature, and the second element is an array containing the
    *         stochastic loss computed for every iteration.
    */
  def runMiniBatch(
                    data: RDD[(Double, Vector)],
                    gradient: Gradient,
                    updater: AdaptiveUpdater,
                    momentumFraction: Double,
                    learningRate: Double,
                    numIterations: Int,
                    regParam: Double,
                    miniBatchFraction: Double,
                    initialWeights: Vector,
                    smoothingTerm: Double,
                    beta: Double,
                    betaS: Double,
                    convergenceTol: Double,
                    regType: Int,
                    decay:Boolean,
                    validationSplit:Double,
                    iterValidation: Int): (Vector, Array[Double]) = {

    // convergenceTol should be set with non minibatch settings
    if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
      logWarning("Testing against a convergenceTol when using miniBatchFraction " +
        "< 1.0 can be unstable because of the stochasticity in sampling.")
    }

    if (numIterations * miniBatchFraction < 1.0) {
      logWarning("Not all examples will be used if numIterations * miniBatchFraction < 1.0: " +
        s"numIterations=$numIterations and miniBatchFraction=$miniBatchFraction")
    }

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    //::Extension:: validationLoss used to check when to stop
    val validationLoss = new ArrayBuffer[Double](numIterations)
    // Record previous weight and current one to calculate solution vector difference

    var previousWeights: Option[Vector] = None
    var currentWeights: Option[Vector] = None

    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("GradientDescent.runMiniBatch returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    if (numExamples * miniBatchFraction < 1) {
      logWarning("The miniBatchFraction is too small")
    }

    // Initialize weights as a column vector
    var weights: Vector = Vectors.dense(initialWeights.toArray)
    //::Extension:: weightsShifted is used for nesterov (it representes the "predicted future weights", not the current weight
    //position).
    var weightsShifted = weights
    val n = weights.size

    var regVal = 0.0
    //::Extension:: check which updater is being used and call its corresponding function with the appropriate parameters
    //Note that this initial calculation is only used for getting the corret regularization value for the initial loss
    updater match {
      case _: SimpleUpdater =>
        regVal = updater.asInstanceOf[SimpleUpdater].compute(
          weights, Vectors.zeros(weights.size), 0, 0, regParam,regType,decay)._2
      case _: AdamUpdater =>
        regVal = updater.asInstanceOf[AdamUpdater].compute(
          weights, Vectors.zeros(weights.size), 0, smoothingTerm, beta, betaS, 0, regParam, regType, decay)._2
      case _: AdagradUpdater =>
        regVal = updater.asInstanceOf[AdagradUpdater].compute(
          weights, Vectors.zeros(weights.size), 0, smoothingTerm, 0, regParam,regType,decay)._2
      case _: MomentumUpdater =>
        regVal = updater.asInstanceOf[MomentumUpdater].compute(
          weights, Vectors.zeros(weights.size), 0, 0, 1, regParam,regType,decay)._2
      case _: NesterovUpdater =>
        regVal = updater.asInstanceOf[NesterovUpdater].compute(
          weights, Vectors.zeros(weights.size), 0, 0, 1, regParam,regType,decay)._2
      case _: AdamaxUpdater =>
        regVal = updater.asInstanceOf[AdamaxUpdater].compute(
          weights, Vectors.zeros(weights.size), 0, smoothingTerm, beta, betaS, 0, regParam,regType,decay)._2
      case _: AdadeltaUpdater =>
        regVal = updater.asInstanceOf[AdadeltaUpdater].compute(
          weights, Vectors.zeros(weights.size), 0, smoothingTerm, 0, 0, regParam,regType,decay)._2
      case _: RMSpropUpdater =>
        regVal = updater.asInstanceOf[RMSpropUpdater].compute(
          weights, Vectors.zeros(weights.size), 0, smoothingTerm, 0, regParam,regType,decay)._2
      case _:NadamUpdater =>
        regVal = updater.asInstanceOf[NadamUpdater].compute(
          weights, Vectors.zeros(weights.size), 0, smoothingTerm, beta, betaS, 0, regParam,regType,decay)._2
      case _: AMSGradUpdater =>
        regVal = updater.asInstanceOf[AMSGradUpdater].compute(
          weights, Vectors.zeros(weights.size), 0, smoothingTerm, beta, betaS, 0, regParam,regType,decay)._2
    }


    var converged = false // indicates whether converged based on convergenceTol
    var i = 1

    //::Extension:: split the data between training and validation according to validationSplit
    val Array(train,validation) = data.randomSplit(Array(1-validationSplit, validationSplit), seed = 41);

    //::Extension:: variable that indicates whether the early stopping condition has been triggered
    var earlyStopped = false

    //::Extension:: Iterate while not converged, not maximum iterations reached and (extension) not stopped early
    while (!converged && i <= numIterations && !earlyStopped) {

      var bcWeights: org.apache.spark.broadcast.Broadcast[Vector] = null
      //::Extension:: For the case of nesterov, we have to broadcast the "predicted future weights" to calculate the
      //gradient
      if (updater.isInstanceOf[NesterovUpdater]){
        bcWeights = data.context.broadcast(weightsShifted)
      }
      //::Extension:: Otherwise, the actual weights is what needs to be broadcasted
      else{
        bcWeights = data.context.broadcast(weights)
      }

      // Sample a subset (fraction miniBatchFraction) of the total data
      // compute and sum up the subgradients on this subset (this is one map-reduce)
      val (gradientSum, lossSum, miniBatchSize) = train.sample(false, miniBatchFraction, 42 + i)
        .treeAggregate((BDV.zeros[Double](n), 0.0, 0L)) (
          seqOp = (c, v) => {
            // c: (grad, loss, count), v: (label, features)
            val l = gradient.compute(v._2, v._1, bcWeights.value, Vectors.fromBreeze(c._1))
            (c._1, c._2 + l, c._3 + 1)
          },
          combOp = (c1, c2) => {
            // c: (grad, loss, count)
            (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
          })
      //::Extension:: If the validation split is larger than 0, calculate the loss on the validation set to check
      //later the early stopping condition
      var lossSumVal=20
      var miniBatchSizeVal=1
      var gradientSumVal=0
      if(validationSplit>0) {
        var (gradientSumVal, lossSumVal, miniBatchSizeVal) = validation.sample(false, miniBatchFraction, 42 + i)
          .treeAggregate((BDV.zeros[Double](n), 0.0, 0L))(
            seqOp = (c, v) => {
              // c: (grad, loss, count), v: (label, features)
              val l = gradient.compute(v._2, v._1, bcWeights.value, Vectors.fromBreeze(c._1))
              (c._1, c._2 + l, c._3 + 1)
            },
            combOp = (c1, c2) => {
              // c: (grad, loss, count)
              (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
            })
        //::Extension:: Accumulate validation loss
        validationLoss += lossSumVal / miniBatchSizeVal + regVal
      }

      if (miniBatchSize > 0) {
        /**
          * lossSum is computed using the weights from the previous iteration
          * and regVal is the regularization value computed in the previous iteration as well.
          */
        stochasticLossHistory += lossSum / miniBatchSize + regVal
        //::Extension:: Update the weights with the corresponding updater call with the appropriate parameters
        // Only performing the weight update if the early stopping condition is not met. Otherwise the
        //weights from the previous iteration need to be returned.
        if(validationSplit==0||i<=iterValidation||(validationLoss(i-1) < validationLoss(i-2))) {
          updater match {
            case _: SimpleUpdater =>
              val update = updater.asInstanceOf[SimpleUpdater].compute(
                weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble), learningRate, i, regParam, regType, decay)
              weights = update._1
              regVal = update._2
            case _: AdamUpdater =>
              val update = updater.asInstanceOf[AdamUpdater].compute(
                weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble), learningRate, smoothingTerm, beta, betaS,
                i, regParam, regType, decay)
              weights = update._1
              regVal = update._2
            case _: AdagradUpdater =>
              val update = updater.asInstanceOf[AdagradUpdater].compute(
                weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble), learningRate, smoothingTerm,
                i, regParam, regType, decay)
              weights = update._1
              regVal = update._2
            case _: MomentumUpdater =>
              val update = updater.asInstanceOf[MomentumUpdater].compute(
                weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble), momentumFraction,
                learningRate, i, regParam, regType, decay)
              weights = update._1
              regVal = update._2
            case _: NesterovUpdater =>
              val update = updater.asInstanceOf[NesterovUpdater].compute(
                weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble), momentumFraction,
                learningRate, i, regParam, regType, decay)
              weights = update._1
              regVal = update._2
              weightsShifted = update._3
            case _: AdamaxUpdater =>
              val update = updater.asInstanceOf[AdamaxUpdater].compute(
                weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble), learningRate, smoothingTerm, beta, betaS,
                i, regParam, regType, decay)
              weights = update._1
              regVal = update._2
            case _: AdadeltaUpdater =>
              val update = updater.asInstanceOf[AdadeltaUpdater].compute(
                weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble), learningRate, smoothingTerm,
                i, momentumFraction, regParam, regType, decay)
              weights = update._1
              regVal = update._2
            case _: RMSpropUpdater =>
              val update = updater.asInstanceOf[RMSpropUpdater].compute(
                weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble), learningRate, smoothingTerm,
                i, regParam, regType, decay)
              weights = update._1
              regVal = update._2
            case _: NadamUpdater =>
              val update = updater.asInstanceOf[NadamUpdater].compute(
                weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble), learningRate, smoothingTerm, beta, betaS,
                i, regParam, regType, decay)
              weights = update._1
              regVal = update._2
            case _: AMSGradUpdater =>
              val update = updater.asInstanceOf[AMSGradUpdater].compute(
                weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble), learningRate, smoothingTerm, beta, betaS,
                i, regParam, regType, decay)
              weights = update._1
              regVal = update._2
          }
        }
        else {
          earlyStopped = true
        }
        previousWeights = currentWeights
        currentWeights = Some(weights)

        if (previousWeights != None && currentWeights != None) {
          converged = isConverged(previousWeights.get,
            currentWeights.get, convergenceTol)
        }
      } else {
        logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
      }
      //Print loss of current iteration
      println("Loss in iteration "+i+" : "+(lossSum / miniBatchSize + regVal))
      //Print validation loss of current iteration (optional)
      //if(validationSplit>0) println("Validation loss in iteration "+i+" : "+validationLoss(i-1))

      i += 1
    }

    logInfo("GradientDescent.runMiniBatch finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))

    //return the weights and the training loss of all iterations
    (weights, stochasticLossHistory.toArray)

  }

  /**
    * Alias of [[runMiniBatch]] with convergenceTol set to default value of 0.001.
    */
  def runMiniBatch(
                    data: RDD[(Double, Vector)],
                    gradient: Gradient,
                    updater: AdaptiveUpdater,
                    momentumFraction: Double,
                    learningRate: Double,
                    numIterations: Int,
                    regParam: Double,
                    miniBatchFraction: Double,
                    smoothingTerm: Double,
                    beta: Double,
                    betaS: Double,
                    initialWeights: Vector,
                    regType:Int,
                    decay:Boolean,
                    validationSplit:Double,
                    iterValidation: Int): (Vector, Array[Double]) =
    GradientDescentAlg.runMiniBatch(data, gradient, updater, momentumFraction, learningRate, numIterations,
      regParam, miniBatchFraction, initialWeights, beta, betaS, smoothingTerm, 0.001,regType,decay,validationSplit,
      iterValidation)


  private def isConverged(
                           previousWeights: Vector,
                           currentWeights: Vector,
                           convergenceTol: Double): Boolean = {
    // To compare with convergence tolerance.
    val previousBDV = previousWeights.asBreeze.toDenseVector
    val currentBDV = currentWeights.asBreeze.toDenseVector

    // This represents the difference of updated weights in the iteration.
    val solutionVecDiff: Double = norm(previousBDV - currentBDV)
    solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
  }


}