package org.apache.spark.mllib.optimization

/**
  * Trait that serves as interface for all of our implemented adaptive updaters
  *
  *
  * Note that we cannot define the "compute" method because it has variable of input and output arguments
  * depending on the particular updater
  */
trait AdaptiveUpdater extends Serializable {

}
