package org.apache.spark.mllib.optimization

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.math._
import breeze.linalg.{DenseVector, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}

class MomentumUpdater{

    //TODO: test if it works fastly in a dense dataset
    //TODO: pass momentumFraction as an argument (need to define a new GradientDescent class to pass new parameters)

  private [this] var momentumOld: BV[Double] = _
  private [this] var momentumFraction: Double = 0.9

  def compute(
                        weightsOld: Vector,
                        gradient: Vector,
                        momentumFraction: Double,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (Vector, Double) = {
    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient: BV[Double] = gradient.asBreeze.toDenseVector
    val momentumNew = momentumOld :*= momentumFraction + stepSize :*= brzGradient
    val weightsNew = brzWeights - momentumNew
    momentumOld = momentumNew
    (Vectors.fromBreeze(weightsNew), 0)
  }
}
