package org.apache.spark.mllib.optimization

import breeze.linalg.{Vector => BV, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.math.{abs, max, signum}

/**
  * Class that implements the simple updater, based on the original class "Updater" from org.apache.spark.mllib.optimization
  *
  *
  * Combines the three updaters (no regularization, L1 and L2) in one method and makes it inherit from
  * our AdaptiveUpdater interface. The operations themselves are equal as the ones in that class
  */
class SimpleUpdater extends AdaptiveUpdater {

  /**
    * Compute an updated value for weights given the parameters. Performs one update step at a time
    *
    * @param weightsOld - Column matrix of size dx1 where d is the number of features.
    * @param inStepSize - step size parameter (also named learning rate) which indicates the strength of the
    *                   update based on the gradient.
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
  def compute(
                        weightsOld: Vector,
                        gradient: Vector,
                        inStepSize: Double,
                        iter: Int,
                        regParam: Double,
                        regType: Int,
                        decay: Boolean): (Vector, Double) = {
    //Decay or not the stepSize according to the variable decay
    var stepSize : Double = 0
    if(decay) stepSize = inStepSize / math.sqrt(iter)
    else stepSize = inStepSize

    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient = gradient.asBreeze.toDenseVector
    //L1 Regularization
    if(regType==1) {
      brzAxpy(-stepSize, gradient.asBreeze, brzWeights)
      // Apply proximal operator (soft thresholding)
      val shrinkageVal = regParam * stepSize
      var i = 0
      val len = brzWeights.length
      while (i < len) {
        val wi = brzWeights(i)
        brzWeights(i) = signum(wi) * max(0.0, abs(wi) - shrinkageVal)
        i += 1
      }

      (Vectors.fromBreeze(brzWeights), brzNorm(brzWeights, 1.0) * regParam)
    }
    //L2 regularization
    else if (regType==2){
      brzWeights :*= (1.0 - stepSize * regParam)
      brzAxpy(-stepSize, gradient.asBreeze, brzWeights)
      val norm = brzNorm(brzWeights, 2.0)

      (Vectors.fromBreeze(brzWeights), 0.5 * regParam * norm * norm)
    }
    //No regularization
    else{
      brzAxpy(-stepSize, gradient.asBreeze, brzWeights)
      (Vectors.fromBreeze(brzWeights), 0)
    }
  }
}
