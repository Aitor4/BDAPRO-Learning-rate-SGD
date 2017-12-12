package org.apache.spark.mllib.optimization

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.math._
import breeze.linalg.{DenseVector, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}


class NesterovUpdater{
  /*
  TODO: Test for correctness
   */


  private [this] var momentumOld: BV[Double] = _

  def compute(weightsOld: Vector,
                        gradientShifted: Vector,
                        momentumFraction: Double,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (Vector, Double, Vector) = {


    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient: BV[Double] = gradientShifted.asBreeze.toDenseVector
    val momentumNew = momentumOld :*= momentumFraction + stepSize :*= brzGradient
    val weightsNew = brzWeights - momentumNew
    momentumOld = momentumNew
    (Vectors.fromBreeze(weightsNew), 0, Vectors.fromBreeze(weightsNew-momentumNew))
  }
}