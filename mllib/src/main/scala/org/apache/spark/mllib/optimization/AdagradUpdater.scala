package org.apache.spark.mllib.optimization

import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.math._
import breeze.linalg.{DenseMatrix, DenseVector, diag, inv, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}

class AdagradUpdater extends AdaptiveUpdater {

  private [this] var squaredGradients: DenseVector[Double] = null

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
    if(squaredGradients == null) squaredGradients = brzGradient :* brzGradient
    else squaredGradients = squaredGradients + (brzGradient :* brzGradient)
    val denom: DenseVector[Double] = squaredGradients + smoothingTerm
    val root=brzSqrt(denom)
    val mult = DenseVector.fill(weightsOld.size){stepSize} / root
    /*Even though the equation shows a multiplication of a diagonal matrix and a svector,
  here it is transformed to an element-wise multiplication of the diagonal (in vector representation)
  and the vector. The operation is equivalent and it saves memory space and computation time*/
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

