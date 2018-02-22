package org.apache.spark.mllib.optimization

import breeze.linalg.{DenseVector, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.math.{abs, max, signum}


class RMSpropUpdater extends AdaptiveUpdater{

  private [this] var accGradient: DenseVector[Double] = null


  def compute(weightsOld: Vector,
              gradient: Vector,
              inStepSize: Double,
              smoothingTerm: Double,
              iter: Int,
              regParam : Double,
              regType: Int,
              decay: Boolean): (Vector, Double) = {
    var stepSize : Double = 0
    if(decay) stepSize = inStepSize / math.sqrt(iter)
    else stepSize = inStepSize

    val brzWeights: DenseVector[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient: DenseVector[Double] = gradient.asBreeze.toDenseVector

    if (accGradient == null) accGradient = DenseVector.zeros(gradient.size)

    //accumulate gradient
    accGradient = 0.9*accGradient + 0.1 * (brzGradient :* brzGradient)

    //compute update
    val denom: DenseVector[Double] = brzSqrt(accGradient + smoothingTerm)
    val mult =  DenseVector.fill(weightsOld.size){ stepSize }/ denom
    val update: DenseVector[Double] =  mult :* brzGradient

    //L1 Regularization
    if(regType==1) {
      val weightsNew = brzWeights - update
      // Apply proximal operator (soft thresholding)
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
      brzWeights :*= (1.0 - stepSize* regParam)
      val weightsNew = brzWeights - update
      val norm = brzNorm(weightsNew, 2.0)

      (Vectors.fromBreeze(weightsNew), 0.5 * regParam * norm * norm)
    }
    //No regularization
    else{
      val weightsNew = brzWeights - update
      (Vectors.fromBreeze(weightsNew), 0)
    }
  }

}