package org.apache.spark.mllib.optimization

import breeze.linalg.{DenseVector, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.mllib.linalg.{Vector, Vectors}


class RMSpropUpdater{

  private [this] var accGradient: DenseVector[Double] = null


  def compute(weightsOld: Vector,
              gradient: Vector,
              stepSize: Double,
              smoothingTerm: Double,
              iter: Int,
              rho: Double,
              regParam : Double): (Vector, Double) = {
    val brzWeights: DenseVector[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient: DenseVector[Double] = gradient.asBreeze.toDenseVector

    if (accGradient == null) accGradient = DenseVector.zeros(gradient.size)

    //accumulate gradient
    accGradient = rho*accGradient + (1-rho) * (brzGradient :* brzGradient)

    //compute update
    //A good default value for the learning rate (stepSize) is 0.001 for this alg.
    val denom: DenseVector[Double] = brzSqrt(accGradient + smoothingTerm)
    val optStepSize = 0.001
    val mult =  DenseVector.fill(weightsOld.size){ optStepSize }/ denom
    val update: DenseVector[Double] =  mult :* brzGradient

    val weightsNew = brzWeights - update

    (Vectors.fromBreeze(weightsNew), 0)
  }

}