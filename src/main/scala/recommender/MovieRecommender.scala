package recommender

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{collect_list, slice, sort_array, struct}

object MovieRecommender extends App {

  val spark = SparkSession
    .builder()
    .appName("Movie Recommender")
    .config("spark.master", "local")
    .getOrCreate()

  import spark.implicits._

  Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
  Logger.getLogger("org.spark-project").setLevel(Level.OFF)
  //Logger.getRootLogger.setLevel(Level.WARN)
  //spark.sparkContext.get .setLogLevel("ERROR")
  //java.util.logging.Logger.getGlobal().setLevel(java.util.logging.Level.SEVERE)

  val moviesDir = "/home/tatjana/code/spark/datasets/ml-latest/"

  val rdf: DataFrame = spark.read.format("csv").option("header", "true").load(moviesDir + "ratings.csv")
  //rdf.show()
  //Exception when fitting ALS: Column userId must be of type numeric but was actually of type string.

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
  val model: ALSModel = als.fit(training)

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
  // Root-mean-square error = 0.8425404178067802

  // wrong! compare recommendations, not predictions
  // https://stackoverflow.com/questions/37975715/rankingmetrics-in-spark-scala
//  val testRatingCol: Array[Float] = test.select("rating").collect().map(_.getFloat(0))
//  val testPredictionCol: Array[Float] = predictions.select("prediction").collect().map(_.getFloat(0))
//  val metrics: RDD[(Array[Float], Array[Float])] = spark.createDataset(List((testPredictionCol, testRatingCol))).rdd
//  val rm = new RankingMetrics[Float](metrics)
//  println(s"Mean-average-precision = ${rm.meanAveragePrecision}")
//  // Mean-average-precision = 6.491003557056679E-7
//  val k = 5
//  println(s"Precision at $k = ${rm.precisionAt(k)}")

  val nRecommendations = 10
  // Generate top 10 movie recommendations for each user
  val userRecs = model.recommendForAllUsers(nRecommendations)
  // Generate top 10 user recommendations for each movie
  val movieRecs = model.recommendForAllItems(nRecommendations)

  // recommend for all test users to calculate precision
//  val testUsers = test.select(als.getUserCol).distinct().limit(10)
//  model.setColdStartStrategy("drop")
//  val recommendations = model.recommendForUserSubset(testUsers, nRecommendations)
//  recommendations.show(10, truncate = false)
//  println("Reccommendations size " + recommendations.count())

  val moviesSchema = StructType(Array(
    StructField("movieId", IntegerType, true),
    StructField("title", StringType, true),
    StructField("genres", StringType, true)))

  val movies: Dataset[Movie] = spark.read.format("csv")
    .option("header", "true")
    .option("nullValue", "null")
    .schema(moviesSchema)
    .load(moviesDir + "movies.csv")
    .as[Movie]
  movies.show()

  quickRecommendationCheck(1)
  quickRecommendationCheck(2)
  quickRecommendationCheck(3)

  def quickRecommendationCheck(userID: Int): Unit = {

    val existingMovieIDs: Array[Int] = training.
      filter($"userId" === userID).
      select("movieId").as[Int].collect()

    println(s"Movies user ${userID} watched")
    movies.filter($"movieId" isin (existingMovieIDs:_*)).show(truncate = false)

    val user = training.filter($"userId" === userID)
    val userRecs: DataFrame = model.recommendForUserSubset(user, 5)

    val recommendedMovieIDs = userRecs.select("recommendations").as[Array[(Int, Float)]]
      .flatMap(ar => ar.map(_._1)).collect()

    println(s"Movies recommended to user ${userID}")
    movies.filter($"movieId" isin (recommendedMovieIDs:_*)).show(truncate = false)
  }

  spark.stop()

}
