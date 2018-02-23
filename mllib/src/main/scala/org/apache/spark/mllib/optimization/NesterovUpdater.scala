package org.apache.spark.mllib.optimization

import scala.math._
import breeze.linalg.{DenseVector, Vector => BV, norm => brzNorm}
import org.apache.spark.mllib.linalg.{Vector, Vectors}


/**
  * Class that implements the momentum updater
  *
  *
  * For more information about is theoretical background,
  * refer to the original paper: Nesterov, Y. (1983). A method for unconstrained convex minimization problem
  * with the rate of convergence o(1/k2). Doklady ANSSSR (translated as Soviet.Math.Docl.), vol. 269, pp. 543– 547.
  */
class NesterovUpdater extends AdaptiveUpdater{

  //Momentum variable to be accumulated across iterations
  private [this] var momentumOld: BV[Double] = null

  /**
    * Compute an updated value for weights given the parameters. Performs one update step at a time
    *
    * @param weightsOld - Column matrix of size dx1 where d is the number of features.
    * @param gradientShifted - Column matrix of size dx1 where d is the number of features.
    *                        Note that it is calculated not in the current position of the weights, but
    *                        in its "predicted future position" according to the definition of Nesterov
    * @param inStepSize - step size parameter (also named learning rate) which indicates the strength of the
    *                   update based on the gradient.
    * @param momentumFraction - The proportion of the update that corresponds to the accumulated momentum
    * @param iter - Iteration number (used if we decide to decay the learning rate across iterations)
    * @param regParam - Regularization parameter (only used if the type of regularization is not 0)
    * @param regType - Which type of regularization is used:
    *                0: no regularization, 1: L1 regularization, 2: L2 regularization
    * @param decay - Whether to decay or not the stepSize across iterations (divide by the square root of
    *              the iterations)
    * @return A tuple of 3 elements. The first element is a column matrix containing updated weights,
    *         the second element is the regularization value computed using updated weights (useful for the
    *         regularized loss) and the third element is the "predicted future position" according to the new
    *         momentum term and the position of the weights (where the gradient is actually calculated for the
    *         next step)
    */
  def compute(weightsOld: Vector,
                        gradientShifted: Vector,
                        momentumFraction: Double,
                        inStepSize: Double,
                        iter: Int,
                        regParam: Double,
                        regType: Int,
                        decay: Boolean): (Vector, Double, Vector) = {
    //Decay or not the stepSize according to the variable decay
    var stepSize : Double = 0
    if(decay) stepSize = inStepSize / math.sqrt(iter)
    else stepSize = inStepSize

    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient: BV[Double] = gradientShifted.asBreeze.toDenseVector

    //Initialize the accumulated momentum
    if(momentumOld == null) {momentumOld = DenseVector.zeros[Double](weightsOld.size)}
    val momentumNew = momentumOld * momentumFraction + brzGradient * stepSize
    //Update it
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
      (Vectors.fromBreeze(weightsNew), brzNorm(weightsNew, 1.0) * regParam,Vectors.fromBreeze(weightsNew-momentumFraction*momentumNew))
    }
    //L2 regularization
    else if (regType==2){
      //Update weights according to L2
      brzWeights :*= (1.0 - stepSize* regParam)
      //Apply update
      val weightsNew = brzWeights - momentumNew

      val norm = brzNorm(weightsNew, 2.0)
      (Vectors.fromBreeze(weightsNew), 0.5 * regParam * norm * norm,Vectors.fromBreeze(weightsNew-momentumFraction*momentumNew))
    }
    //No regularization
    else{
      //Simply apply update
      val weightsNew = brzWeights - momentumNew
      (Vectors.fromBreeze(weightsNew), 0,Vectors.fromBreeze(weightsNew-momentumFraction*momentumNew))
    }
  }
}