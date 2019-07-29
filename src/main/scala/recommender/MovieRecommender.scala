package recommender

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.rdd.RDD

object MovieRecommender extends App {

  val spark = SparkSession
    .builder()
    .appName("Movie Reccommender")
    .config("spark.master", "local")
    .getOrCreate()

  import spark.implicits._

  val moviesDir = "/home/tatjana/code/spark/datasets/ml-latest/"

  val rdf: DataFrame = spark.read.format("csv").option("header", "true").load(moviesDir + "ratings.csv")
  //rdf.show()
  //Exception when fitting ALS: Column userId must be of type numeric but was actually of type string.
  //val Array(training, test) = rdf.randomSplit(Array(0.8, 0.2))

  // digression: cast types to get Dataset
  rdf.createOrReplaceTempView("ratings")
  val rds: Dataset[Rating] = spark.sql(
    """select cast(userId as int) as userId, cast(movieId as int) as movieId,
      |cast(rating as float) as rating, cast(timestamp as long) as timestamp from ratings""".stripMargin).as[Rating]
  //rds.show()

  val customSchema = StructType(Array(
    StructField("userId", IntegerType, true),
    StructField("movieId", IntegerType, true),
    StructField("rating", FloatType, true),
    StructField("timestamp", LongType, true)))

  val ratings: Dataset[Rating] = spark.read.format("csv")
    .option("header", "true")
    .option("nullValue", "null")
    .schema(customSchema)
    .load(moviesDir + "ratings.csv")
    .as[Rating]
  ratings.show()

//    from source code CSVOptions
//    .option("delimiter", "\t")
//    .option("timestampFormat", "yyyy/MM/dd HH:mm:ss")
//    .option("inferSchema", "true")

  val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

  val als = new ALS()
    .setMaxIter(5)
    .setRegParam(0.01)
    .setUserCol("userId")
    .setItemCol("movieId")
    .setRatingCol("rating")
  val model = als.fit(training)

  // Evaluate the model by computing the RMSE on the test data
  // RMSE applicable because of explicit rating
  // Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
  model.setColdStartStrategy("drop")
  val predictions = model.setPredictionCol("prediction").transform(test)

  val evaluator = new RegressionEvaluator()
    .setMetricName("rmse")
    .setLabelCol("rating")
    .setPredictionCol("prediction")

  val rmse = evaluator.evaluate(predictions)
  println(s"Root-mean-square error = $rmse")

  // wrong!
  // https://stackoverflow.com/questions/37975715/rankingmetrics-in-spark-scala
  // TODO: e.g. Compare with top ten most relevant documents
  val testRatingCol: Array[Float] = test.select("rating").collect().map(_.getFloat(0))
  val testPredictionCol: Array[Float] = predictions.select("prediction").collect().map(_.getFloat(0))
  val metrics: RDD[(Array[Float], Array[Float])] = spark.createDataset(List((testPredictionCol, testRatingCol))).rdd
  val rm = new RankingMetrics[Float](metrics)
  println(s"Mean-average-precision = ${rm.meanAveragePrecision}")

  // Generate top 10 movie recommendations for each user
  val userRecs = model.recommendForAllUsers(10)
  // Generate top 10 user recommendations for each movie
  val movieRecs = model.recommendForAllItems(10)

  // Generate top 10 movie recommendations for a specified set of users
  val users = ratings.select(als.getUserCol).distinct().limit(3)
  val userSubsetRecs = model.recommendForUserSubset(users, 10)
  // Generate top 10 user recommendations for a specified set of movies
  val movies = ratings.select(als.getItemCol).distinct().limit(3)
  val movieSubSetRecs = model.recommendForItemSubset(movies, 10)

  //userRecs.show()
  //movieRecs.show()
  //userSubsetRecs.show()
  //movieSubSetRecs.show()

  spark.stop()

}
