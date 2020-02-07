package recommender

case class RankingHelper(userId: Int, actual: Array[(Float, Int)], recommendations: Array[(Int, Float)])
