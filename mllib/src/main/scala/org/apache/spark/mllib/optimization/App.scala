package org.apache.spark.mllib.optimization

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.PipelineModel

// Spark application that runs linear regression for predicting the arrival delay of USA commercial flights.
// The dataset is large (200MB or so per year) and is not sparse


object App {
  def main(args : Array[String]) {


    // Disabling debug option.
    Logger.getRootLogger().setLevel(Level.WARN)
    //Prepare spark session
    val spark = SparkSession
      .builder()
      .appName("Spark Flights Delay")
      .getOrCreate()


    // Loading the data from HDFS

    val targetVariable = "ArrDelay"
    val datasetsPath = args(0)
    // Create a Flights object (from the other implemented class), with the DataFrame and the Machine Learning methods.
    var flights = new Flights(spark, targetVariable)
    flights.load("hdfs://"+datasetsPath+"*.csv")


    // Data Manipulation



    //Drop rows with null values in the target variable	(not allowed in supervised learning).
    flights.df = flights.df.na.drop(Array("ArrDelay"))

    // Transformation of variables for the learning phase.
    flights.variablesTransformation()


    //Discarding unused variables for the prediction
    flights.df = flights.df.drop("DepTime").drop("Cancelled")
      .drop("CancellationCode").drop("FlightNum")
      .drop("TailNum").drop("UniqueCarrier")
      .drop("Year").drop("DayOfMonth")
      .drop("Origin").drop("Dest")


    // Null treatment.
    // We discard all the rows with at least one null value
    flights.df = flights.df.na.drop()

    // We will take the standard deviation to use it as a baseline.
    // To compare the linear regression method vs a naive method (taking the mean) and check that everything is OK
    val dStDev = flights.df.select(stddev("ArrDelay")).take(1)(0)(0)

    // Linear Regression method needs to define a special transformation for categorical variables.
    //OneHotEncoder to create dummy variables for carrier, month and day of the week
    val dayEncoder = new OneHotEncoder().setInputCol("DayOfWeek").setOutputCol("dummyDayOfWeek")
    val monthEncoder = new OneHotEncoder().setInputCol("Month").setOutputCol("dummyMonth")
    val carrierEncoder = new OneHotEncoder().setInputCol("UniqueCarrierInt").setOutputCol("dummyUniqueCarrier")

    flights.df = dayEncoder.transform(flights.df)
    flights.df = monthEncoder.transform(flights.df)
    flights.df = carrierEncoder.transform(flights.df)


    // Training and Test datasets
    // Split the data into training and test sets ( we choose 30% to be held out for testing).
    var Array(trainingData, testData) = flights.df.randomSplit(Array(0.7, 0.3))

    var trainingDataR = trainingData
    var testDataR = testData

    // Drop old variables (the dummy variables will substitute them)
    trainingDataR = trainingDataR.drop("DayOfWeek")
      .drop("Month").drop("UniqueCarrierInt")
    testDataR = testDataR.drop("DayOfWeek")
      .drop("Month").drop("UniqueCarrierInt")


    /* Linear Regression Model
     *
     * We would like to tune the regularisaiton hyperparameter in order to avoid overfitting. We perform a grid search
       * of three parameters using 3-fold cross-validation to find out the best possible combination of parameters.
     *
     * We set the elastic parameter to 1 (lasso regularization)
     */
    val lrMaxNumIterations = 100
    val k = 3
    val regularisations = Array(0.1, 1.0, 10.0)
    flights.linearRegression(trainingDataR, lrMaxNumIterations, k, regularisations)

    // Training the model
    val lrModel = flights.linearRegressionModel.fit(trainingDataR)

    // Retrieving the best model of tuning selection.
    val pipeline = lrModel.bestModel.asInstanceOf[PipelineModel]
    val bestRegularizer = pipeline.stages(1).asInstanceOf[LinearRegressionModel].getRegParam

    // Validation
    val lrPredictions = lrModel.transform(testDataR)
    val rmseRegression = flights.evaluator.evaluate(lrPredictions)


    // Baseline to improve
    println("Standard deviation of arrival delay = "+dStDev)

    // RMSE of Linear regression
    println("Linear regression = "+rmseRegression)

  }
}

