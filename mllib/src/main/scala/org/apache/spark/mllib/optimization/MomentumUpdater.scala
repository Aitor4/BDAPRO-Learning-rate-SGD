package org.apache.spark.mllib.optimization

import scala.math._
import breeze.linalg.{DenseVector, Vector => BV, norm => brzNorm}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
  * Class that implements the momentum updater
  *
  *
  * For more information about is theoretical background,
  * refer to the original paper: Qian, N. (1999). On the momentum term in gradient descent learning algorithms.
  * Neural Networks : The Official Journal of the International Neural Network Society, 12(1), 145–151.
  */
class MomentumUpdater extends AdaptiveUpdater{

  //Momentum variable to be accumulated across iterations
  private [this] var momentumOld: BV[Double] = null

  /**
    * Compute an updated value for weights given the parameters. Performs one update step at a time
    *
    * @param weightsOld - Column matrix of size dx1 where d is the number of features.
    * @param gradient - Column matrix of size dx1 where d is the number of features.
    * @param inStepSize - step size parameter (also named learning rate) which indicates the strength of the
    *                   update based on the gradient.
    * @param momentumFraction - The proportion of the update that corresponds to the accumulated momentum
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
                        momentumFraction: Double,
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
    val brzGradient: BV[Double] = gradient.asBreeze.toDenseVector

    //Initialize the accumulated momentum
    if(momentumOld == null) {momentumOld = DenseVector.zeros[Double](weightsOld.size)}
    //Update it
    val momentumNew = momentumOld * momentumFraction + brzGradient * stepSize
    momentumOld = momentumNew


    //L1 Regularization
    if(regType==1) {
      //Apply update
      val weightsNew = brzWeights - momentumNew
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
      //Apply update
      val weightsNew = brzWeights - momentumNew

      val norm = brzNorm(weightsNew, 2.0)
      (Vectors.fromBreeze(weightsNew), 0.5 * regParam * norm * norm)
    }
    //No regularization
    else{
      //Simply apply update
      val weightsNew = brzWeights - momentumNew
      (Vectors.fromBreeze(weightsNew), 0)
    }
  }
}
