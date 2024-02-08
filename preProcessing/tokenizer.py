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