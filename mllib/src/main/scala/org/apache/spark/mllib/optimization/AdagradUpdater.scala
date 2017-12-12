package org.apache.spark.mllib.optimization

import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.math._
import breeze.linalg.{DenseMatrix, DenseVector, diag, inv, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}

//TODO: Create Gradient descent class for Adagrad
//TODO: Check that matrix multiplication size is perormed correctly
//TODO: Test
class AdagradUpdaterNaive {

  def compute(
                        weightsOld: Vector,
                        gradient: Vector,
                        stepSize: Double,
                        momentumOld: Vector,
                        momentumFraction: Double,
                        squareGradients : Vector,
                        smoothingTerm: Double): (Vector, Double) = {
    val brzSquared: BV[Double] = squareGradients.asBreeze.toDenseVector
    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient: BV[Double] = gradient.asBreeze.toDenseVector

    val root = brzSqrt(brzSquared + smoothingTerm)
    val matrix = diag(root)
    val smoothed = brzGradient *:* stepSize
    val step =  smoothed * inv(matrix)
    val weightsNew = brzWeights - step

    (Vectors.fromBreeze(weightsNew), 0)
  }
}

