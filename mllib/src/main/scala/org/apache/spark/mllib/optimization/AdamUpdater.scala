package org.apache.spark.mllib.optimization

import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.math._
import breeze.linalg.{DenseMatrix, DenseVector, diag, inv, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}

class AdamUpdater {

  private [this] var squaredGradients: DenseVector[Double] = null
  private [this] var gradients: DenseVector[Double] = null
  private [this] var betaPower: Double = 0
  private [this] var betaSPower: Double = 0

  def compute(weightsOld: Vector,
              gradient: Vector,
              stepSize: Double,
              smoothingTerm: Double,
              beta: Double,
              betaS: Double,
              iter: Int,
              regParam : Double): (Vector, Double) = {
    val brzWeights: DenseVector[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient: DenseVector[Double] = gradient.asBreeze.toDenseVector
    if(squaredGradients == null) {
      squaredGradients = betaS * (brzGradient :* brzGradient)
      gradients = beta * brzGradient
    }
    else {
      squaredGradients = (1-betaS) * squaredGradients + betaS * (brzGradient :* brzGradient)
      gradients = (1-beta)*gradients + beta* brzGradient
    }
    betaPower = betaPower * beta
    betaSPower = betaSPower * betaS
    val m =  (1/(1-betaPower)) * gradients
    val v = (1/(1-betaSPower)) * squaredGradients
    val denom: DenseVector[Double] = brzSqrt(v) + smoothingTerm
    val mult = DenseVector.fill(weightsOld.size){stepSize} / denom
    val update: DenseVector[Double] =  mult :* m
    val weightsNew = brzWeights - update
    (Vectors.fromBreeze(weightsNew), 0)
  }

}