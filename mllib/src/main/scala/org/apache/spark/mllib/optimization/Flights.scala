package org.apache.spark.mllib.optimization

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}


//targetVariable is the variable we want to predict (our "y")
class Flights(spark: SparkSession, targetVariable: String) {

  import spark.implicits._

  var df: DataFrame = null
  var evaluator: RegressionEvaluator = null
  var linearRegressionModel: CrossValidator = null

  //Evaluator using the RMSE metric on the predicted variable
  evaluator = new RegressionEvaluator()
    .setLabelCol(targetVariable)
    .setPredictionCol("prediction")
    .setMetricName("rmse")

  /* Read all csv files with headers (they have to be placed in HDFS).
   *
   * The valid columns are selected, casting them to their correct type (the default type is String).
   *
   * @param: hdfsPath, the hdfs path of the datasets.
   */
  def load(hdfsPath: String){
    df = spark.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .load(hdfsPath)
      .select(col("Year").cast(StringType),
        col("Month").cast(StringType),
        col("DayOfMonth").cast(StringType),
        col("DayOfWeek").cast(DoubleType),
        col("DepTime").cast(DoubleType),
        col("CRSDepTime").cast(StringType),
        col("CRSArrtime").cast(StringType),
        col("UniqueCarrier").cast(StringType),
        col("FlightNum").cast(StringType),
        col("TailNum").cast(StringType),
        col("CRSElapsedTime").cast(DoubleType),
        col("ArrDelay").cast(DoubleType),
        col("DepDelay").cast(DoubleType),
        col("Origin").cast(StringType),
        col("Dest").cast(StringType),
        col("Distance").cast(DoubleType),
        col("TaxiOut").cast(DoubleType),
        col("Cancelled").cast(BooleanType),
        col("CancellationCode").cast(StringType))
  }

  /* Transform a column date in "dd/MM/yyyy HHmm" format to a Unix TimeStamp column (uniform)
   *
   * @param: df, the original dataframe
   * @param: columnName, the column name to transform.
   * @return: new dataframe with the TimeStamp column.
   */
  def dateToTimeStamp(df: org.apache.spark.sql.DataFrame, columnName: String) : org.apache.spark.sql.DataFrame = {
    return df.withColumn(columnName,
      unix_timestamp(concat(col("DayOfMonth"), lit("/"), col("Month"), lit("/"), col("Year"), lit(" "), col(columnName)),
        "dd/MM/yyyy HHmm"))
  }

  /* Transformation of initial variables to be suitable for the learning phase.
   *
   * - CRSDepTime and CRSArrTime are converted to TimeStamp.
   * - Numerical variables changed to Double variables due to regression models limitations.
   * - UniqueCarrier variable from String to Categorical using StringIndexer.
   *
   */
  def variablesTransformation(){
    //Convert scheduled departure and arrival time to TimeStamp
    df = dateToTimeStamp(df, "CRSDepTime")
    df = dateToTimeStamp(df, "CRSArrTime")

    // Normalize UNIX time, we take as reference point the earliest date in the database.
    val timeStampReference = unix_timestamp(lit("01/01/1987"), "dd/MM/yy")
    df = df.withColumn("CRSDepTime", $"CRSDepTime" - timeStampReference)
    df = df.withColumn("CRSArrTime", $"CRSArrTime" - timeStampReference)

    //Cast variables to Double to input in the machine learning methods
    df = df.withColumn("DayOfMonth", col("DayOfMonth").cast(DoubleType))
    df = df.withColumn("CRSDepTime", col("CRSDepTime").cast(DoubleType))
    df = df.withColumn("CRSArrTime", col("CRSArrTime").cast(DoubleType))
    df = df.withColumn("Year", col("Year").cast(DoubleType))
    df = df.withColumn("Month", col("Month").cast(DoubleType))

    //StringIndexer to transform the UniqueCarrier string to integer for using it as a categorical variable.
    val sIndexer = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol("UniqueCarrierInt")
    df = sIndexer.fit(df).transform(df)
  }

  /* Linear Regression method.
   *
   * @param: trainingData, the training data.
   * @param: maxIter, max. number of iterations (if it does not converge before).
   * @param: elasticNetParameter, the elastic net parameter from 0 to 1.
   * @param: k, the number of folds in cross validation.
   * @param: hyperaparameters, set of regularizer variables to be tune by cross validation.
   */
  def linearRegression(trainingData: DataFrame, maxIter: Int, k: Int, hyperparameters: Array[Double]){
    //Prepare the assembler that will transform the remaining variables to a feature vector for the ML algorithms
    val assembler = new VectorAssembler()
      .setInputCols(trainingData.drop(targetVariable).columns)
      .setOutputCol("features")

    // Defining the model.
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol(targetVariable)
      .setMaxIter(maxIter)

    //Preparing the pipeline to train and test the data with the regression algorithm.
    val regressionPipeline = new Pipeline().setStages(Array(assembler, lr))

    //To tune the parameters of the model.
    var paramGrid = new ParamGridBuilder()
      .addGrid(lr.getParam("regParam"), hyperparameters)
      .build()

    // Cross validation to select the best hyperparameter.
    linearRegressionModel = new CrossValidator()
      .setEstimator(regressionPipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(k)
  }

}
