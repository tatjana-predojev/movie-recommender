package recommender

import org.apache.spark.sql.types.{FloatType, IntegerType, LongType, TimestampType}

case class Rating(userId: Int,
                  movieId: Int,
                  rating: Float,
                  timestamp: Long)

case class RatingSpark(userId: IntegerType,
                       movieId: IntegerType,
                       rating: FloatType,
                       timestamp: LongType)
