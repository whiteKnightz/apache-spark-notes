from pyspark.ml.feature import Tokenizer

sentences_df = spark.createDataFrame([
    (1, "This is introduction to Spark MLlib"),
    (2, "MLlib includes libraries for classifications and regression"),
    (3, "It also contains supporting tools for pipelines")
],
    ["id", "sentence"])
sentences_df.show()
sent_token = Tokenizer(inputCol="sentence", outputCol="words")
sent_tokenized_df = sent_token.transform(sentences_df)
sent_tokenized_df.show()

from pyspark.ml.feature import HashingTF, IDF
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
sent_hfTF_df = hashingTF.transform(sent_tokenized_df)
sent_hfTF_df.take(1)

idf = IDF(inputCol="rawFeatures", outputCol="idf_features")
idfModel = idf.fit(sent_hfTF_df)
tfidf_df = idfModel.transform(sent_hfTF_df)
tfidf_df.take(1)