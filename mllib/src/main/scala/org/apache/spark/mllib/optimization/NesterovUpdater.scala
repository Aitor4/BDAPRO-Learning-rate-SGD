package org.apache.spark.mllib.optimization

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.math._
import breeze.linalg.{DenseVector, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}


class NesterovUpdater extends Updater {
  /* TODO: Define a new GradientDescent
  TODO: pass momentumFraction as an argument instead of defining it
  TODO: gradientShifted has to be calculated from the new position using the old momentum in GradientDescent
  TODO: Test for correctness
   */


  private [this] var momentumOld: BV[Double] = _
  private [this] var momentumFraction: Double = 0.9

  override def compute(weightsOld: Vector,
                        gradientShifted: Vector,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (Vector, Double) = {


    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient: BV[Double] = gradientShifted.asBreeze.toDenseVector
    val momentumNew = momentumOld :*= momentumFraction + stepSize :*= brzGradient
    val weightsNew = brzWeights - momentumNew
    momentumOld = momentumNew
    (Vectors.fromBreeze(weightsNew), 0)
  }
}