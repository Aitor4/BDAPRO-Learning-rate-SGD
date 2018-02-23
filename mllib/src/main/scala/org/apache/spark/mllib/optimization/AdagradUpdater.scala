package org.apache.spark.mllib.optimization


import scala.math._
import breeze.linalg.{DenseVector, norm => brzNorm}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.mllib.linalg.{Vector, Vectors}


/**
  * Class that implements the Adagrad updater
  *
  *
  * For more information about is theoretical background,
  * refer to the original paper: Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for
  * Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 12, 2121–2159.
  */
class AdagradUpdater extends AdaptiveUpdater {

  //Accumulated gradients to be used across iterations
  private [this] var squaredGradients: DenseVector[Double] = null

  /**
    * Compute an updated value for weights given the parameters. Performs one update step at a time
    *
    * @param weightsOld - Column matrix of size dx1 where d is the number of features.
    * @param gradient - Column matrix of size dx1 where d is the number of features.
    * @param inStepSize - step size parameter (also named learning rate) which indicates the strength of the
    *                   update based on the gradient.
    * @param smoothingTerm - Smoothing term in the denominator of the equation that avoids division by 0.
    *                  It is usually a very small value (e.g. 1e-6)
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
    //Initialize the accumulated gradients or update them
    if(squaredGradients == null) squaredGradients = brzGradient :* brzGradient
    else squaredGradients = squaredGradients + (brzGradient :* brzGradient)
    //Prepare the equation terms
    val denom: DenseVector[Double] = squaredGradients + smoothingTerm
    val root=brzSqrt(denom)
    val mult = DenseVector.fill(weightsOld.size){stepSize} / root
    /*Even though the original equation shows a multiplication of a diagonal matrix and a vector,
  here it is transformed to an element-wise multiplication of the diagonal (in vector representation)
  and the vector. The operation is equivalent and it saves memory space by using a vector*/
    val update: DenseVector[Double] =  mult :* brzGradient

    //L1 Regularization
    if(regType==1) {
      //Apply update
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
      //Modify weights according to L2
      brzWeights :*= (1.0 - stepSize* regParam)
      //Apply update
      val weightsNew = brzWeights - update

      val norm = brzNorm(weightsNew, 2.0)
      (Vectors.fromBreeze(weightsNew), 0.5 * regParam * norm * norm)
    }
    //No regularization
    else{
      //Simply apply update
      val weightsNew = brzWeights - update
      (Vectors.fromBreeze(weightsNew), 0)
    }
  }

}

