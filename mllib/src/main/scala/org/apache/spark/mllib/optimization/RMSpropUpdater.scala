package org.apache.spark.mllib.optimization

import breeze.linalg.{DenseVector, norm => brzNorm}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.math.{abs, max, signum}

/**
  * Class that implements the RMSprop updater
  *
  *
  * For more information about is theoretical background,
  * refer to the original source (note that it is an unpublished updater):
  * https://es.coursera.org/learn/deep-neural-network/lecture/BhJlm/rmsprop
  */
class RMSpropUpdater extends AdaptiveUpdater{

  //Averaged gradient to be used across iterations
  private [this] var accGradient: DenseVector[Double] = null

  /**
    * Compute an updated value for weights given the parameters. Performs one update step at a time
    *
    * @param weightsOld - Column matrix of size dx1 where d is the number of features.
    * @param gradient - Column matrix of size dx1 where d is the number of features.
    * @param inStepSize - step size parameter (also named learning rate) which indicates the strength of the
    *                   update based on the gradient.
    * @param smoothingTerm - Smoothing term in the denominator of the equation that avoids division by 0.
    *                      It is usually a very small value (e.g. 1e-6)
    * @param iter - Iteration number (used if we decide to decay the learning rate across iterations)
    * @param regParam - Regularization parameter (only used if the type of regularization is not 0)
    * @param regType - Which type of regularization is used:
    *                0: no regularization, 1: L1 regularization, 2: L2 regularization
    * @param decay - Whether to decay or not the stepSize across iterations (divide by the square root of
    *              the iterations)
    * @return A tuple of 2 elements. The first element is a column matrix containing updated weights,
    *         and the second element is the regularization value computed using updated weights (useful for the
    *         regularized loss)
    */
  def compute(weightsOld: Vector,
              gradient: Vector,
              inStepSize: Double,
              smoothingTerm: Double,
              iter: Int,
              regParam : Double,
              regType: Int,
              decay: Boolean): (Vector, Double) = {
    //Decay or not the stepSize according to the variable decay
    var stepSize : Double = 0
    if(decay) stepSize = inStepSize / math.sqrt(iter)
    else stepSize = inStepSize

    val brzWeights: DenseVector[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient: DenseVector[Double] = gradient.asBreeze.toDenseVector
    //Initialize the average gradient and updates in the first iteration
    if (accGradient == null) accGradient = DenseVector.zeros(gradient.size)

    //accumulate gradient
    accGradient = 0.9*accGradient + 0.1 * (brzGradient :* brzGradient)

    //compute update
    val denom: DenseVector[Double] = brzSqrt(accGradient + smoothingTerm)
    val mult =  DenseVector.fill(weightsOld.size){ stepSize }/ denom
    val update: DenseVector[Double] =  mult :* brzGradient

    //L1 Regularization
    if(regType==1) {
      //Apply updates
      val weightsNew = brzWeights - update
      // Apply proximal operator (soft thresholding) according to L1
      val shrinkageVal = regParam * stepSize
      var i = 0
      val len = brzWeights.length
      while (i < len) {
        val wi = weightsNew(i)
        weightsNew(i) = signum(wi) * max(0.0, abs(wi) - shrinkageVal)
        i += 1
      }
      (Vectors.fromBreeze(weightsNew), brzNorm(weightsNew, 1.0) * regParam)
    }
    //L2 regularization
    else if (regType==2){
      //Update weights according to L2
      brzWeights :*= (1.0 - stepSize* regParam)
      //Apply updates
      val weightsNew = brzWeights - update
      val norm = brzNorm(weightsNew, 2.0)

      (Vectors.fromBreeze(weightsNew), 0.5 * regParam * norm * norm)
    }
    //No regularization
    else{
      //Apply updates
      val weightsNew = brzWeights - update
      (Vectors.fromBreeze(weightsNew), 0)
    }
  }

}