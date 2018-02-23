package org.apache.spark.mllib.optimization

import breeze.linalg.{DenseVector, norm => brzNorm}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.math.{abs, max, signum}


/**
  * Class that implements the Adadelta updater
  *
  *
  * For more information about is theoretical background,
  * refer to the original paper: Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method.
  */
class AdadeltaUpdater extends AdaptiveUpdater{

  //Averaged gradient and updates to be used across iterations
  private [this] var accGradient: DenseVector[Double] = null
  private [this] var accUpdates: DenseVector[Double] = null
  /**
    * Compute an updated value for weights given the parameters. Performs one update step at a time
    *
    * @param weightsOld - Column matrix of size dx1 where d is the number of features.
    * @param gradient - Column matrix of size dx1 where d is the number of features.
    * @param inStepSize - step size parameter (also named learning rate). Note that for Adadelta, it does not
    *                   indicate the strenght of the update (automatically calculated by the algorithm)
    *                   but rather the strength of the regularization applied (if present)
    * @param smoothingTerm - Smoothing term in the denominator of the equation that avoids division by 0.
    *                      It is usually a very small value (e.g. 1e-6)
    * @param iter - Iteration number (used if we decide to decay the learning rate across iterations)
    * @param rho - The rho parameter of Adadelta (indicates how fast the average decay is)
    * @param regParam - Regularization parameter (only used if the type of regularization is not 0)
    * @param regType - Which type of regularization is used:
    *                0: no regularization, 1: L1 regularization, 2: L2 regularization
    * @param decay - Whether to decay or not the stepSize across iterations (divide by the square root of
    *              the iterations). Again, only useful for regularization in Adadelta, not used for the
    *              actual update of the weights
    * @return A tuple of 2 elements. The first element is a column matrix containing updated weights,
    *         and the second element is the regularization value computed using updated weights (useful for the
    *         regularized loss)
    */
  def compute(weightsOld: Vector,
              gradient: Vector,
              inStepSize: Double,
              smoothingTerm: Double,
              iter: Int,
              rho: Double,
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
    if (accUpdates == null) accUpdates = DenseVector.zeros(gradient.size)

    //accumulate gradient
    accGradient = rho*accGradient + (1-rho) * (brzGradient :* brzGradient)
    //compute update
    val update = (brzSqrt(accUpdates + smoothingTerm) / brzSqrt(accGradient + smoothingTerm)) :* brzGradient
    //accumulate updates
    accUpdates = rho*accUpdates + (1-rho) * (update :* update)

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
      //Modify weights according to L2 reg.
      brzWeights :*= (1.0 - stepSize* regParam)
      //Apply updates
      val weightsNew = brzWeights - update

      val norm = brzNorm(weightsNew, 2.0)
      (Vectors.fromBreeze(weightsNew), 0.5 * regParam * norm * norm)
    }
    //No regularization
    else{
      //Simply apply updates
      val weightsNew = brzWeights - update
      (Vectors.fromBreeze(weightsNew), 0)
    }
  }

}
