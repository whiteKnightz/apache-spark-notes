emp_df = spark.read.csv("employee.txt", header=True)
emp_df.schema
emp_df.printSchema()
emp_df.columns
emp_df.take(5)
emp_df.count()
sample_df = emp_df.sample(False, 0.1)
emp_mgrs_df = emp_df.filter("salary >= 100000")
emp_mgrs_df.count()
emp_mgrs_df.select("salary").show()
