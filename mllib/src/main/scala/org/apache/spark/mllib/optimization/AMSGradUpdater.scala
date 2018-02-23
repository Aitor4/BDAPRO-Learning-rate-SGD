package org.apache.spark.mllib.optimization

import scala.math._
import breeze.linalg.{DenseVector, norm => brzNorm , max=> brzMax}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
  * Class that implements the AMSGrad updater
  *
  *
  * For more information about is theoretical background,
  * refer to the original paper: Reddi, Sashank J., Kale, Satyen, & Kumar, Sanjiv.
  * On the Convergence of Adam and Beyond. Proceedings of ICLR 2018.
  */
class AMSGradUpdater extends AdaptiveUpdater {

  //Variables to accumulate gradients, squared gradients, and maximum squared gradient variables across iterations
  private [this] var squaredGradients: DenseVector[Double] = null
  private [this] var gradients: DenseVector[Double] = null
  private [this] var maxSquaredGradients: DenseVector[Double] = null

  /**
    * Compute an updated value for weights given the parameters. Performs one update step at a time
    *
    * @param weightsOld - Column matrix of size dx1 where d is the number of features.
    * @param gradient - Column matrix of size dx1 where d is the number of features.
    * @param inStepSize - step size parameter (also named learning rate) which indicates the strength of the
    *                   update based on the gradient.
    * @param smoothingTerm - Smoothing term in the denominator of the equation that avoids division by 0.
    *                      It is usually a very small value (e.g. 1e-6)
    * @param beta - Beta1 parameter of the updater. It indicates how fast the gradient decay is.
    * @param betaS - Beta2 parameter of the updater. It indicates how fast the squared gradient decay is.
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
              beta: Double,
              betaS: Double,
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
    //Initialize the average gradient and squared gradient, max squared gradient variables
    if(squaredGradients == null) {
      squaredGradients = (1-betaS) * (brzGradient :* brzGradient)
      maxSquaredGradients = squaredGradients
      gradients = (1-beta) * brzGradient
    }
      //Update those variables
    else {
      squaredGradients = betaS * squaredGradients + (1-betaS) * (brzGradient :* brzGradient)
      maxSquaredGradients = brzMax(squaredGradients,maxSquaredGradients)
      gradients = beta*gradients + (1-beta)* brzGradient
    }
    //Prepare the terms of the equation
    val denom: DenseVector[Double] = brzSqrt(squaredGradients) + smoothingTerm
    val mult = DenseVector.fill(weightsOld.size){stepSize} / denom
    val update: DenseVector[Double] =  mult :* gradients


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
      //Update weights according to L2
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