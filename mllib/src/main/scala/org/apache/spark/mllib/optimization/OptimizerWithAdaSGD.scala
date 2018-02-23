
package org.apache.spark.mllib.optimization

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

/**
  * Trait for optimization problem solvers, modified to return not only the weights but also the
  * training loss (in order to return the loss when training the algorithm).
  *
  * The trait is based on the trait "Optimizer" from org.apache.spark.mllib.optimization
  */
@DeveloperApi
trait OptimizerWithAdaSGD extends Serializable {

  /**
    * Solve the provided convex optimization problem.
    */
  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): (Vector, Array[Double])
}