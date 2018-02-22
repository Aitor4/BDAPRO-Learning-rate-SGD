package org.apache.spark.mllib.optimization

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.math._
import breeze.linalg.{DenseVector, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}

class MomentumUpdater extends AdaptiveUpdater{

  private [this] var momentumOld: BV[Double] = null

  def compute(
                        weightsOld: Vector,
                        gradient: Vector,
                        momentumFraction: Double,
                        inStepSize: Double,
                        iter: Int,
                        regParam: Double,
                        regType: Int,
                        decay: Boolean): (Vector, Double) = {
    var stepSize : Double = 0
    if(decay) stepSize = inStepSize / math.sqrt(iter)
    else stepSize = inStepSize

    if(momentumOld == null) {momentumOld = DenseVector.zeros[Double](weightsOld.size)}
    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient: BV[Double] = gradient.asBreeze.toDenseVector
    val momentumNew = momentumOld * momentumFraction + brzGradient * stepSize
    momentumOld = momentumNew


    //L1 Regularization
    if(regType==1) {
      val weightsNew = brzWeights - momentumNew
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
      val weightsNew = brzWeights - momentumNew
      val norm = brzNorm(weightsNew, 2.0)

      (Vectors.fromBreeze(weightsNew), 0.5 * regParam * norm * norm)
    }
    //No regularization
    else{
      val weightsNew = brzWeights - momentumNew
      (Vectors.fromBreeze(weightsNew), 0)
    }
  }
}
