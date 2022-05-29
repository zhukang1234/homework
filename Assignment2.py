# import the module which we use
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

#Creat the spark session
spark = SparkSession.builder.appName("COMP5349 A2 Data Loading Example").getOrCreate()

#Load the data and display the schema
test_data = "/Users/zhukang/Downloads/train_separate_questions.json"
test_init_df = spark.read.json(test_data)
test_init_df.printSchema()

#Check the schema of a data frame
test_data_df= test_init_df.select((explode("data").alias('data')))
test_data_df.printSchema()

#Process the data as needed
test_paragraph_df = test_data_df.select(explode("data.paragraphs").alias("paragraph"))
test_paragraph_df.printSchema()
test_paragraph_df=test_paragraph_df.select("paragraph.*")
test_paragraph_df.show(5)
test_paragraph_df.count()
test_context_qas_df=test_paragraph_df.withColumn('id',monotonically_increasing_id())
test_context_qas_df=test_context_qas_df.withColumn("length", length("context"))
test_context_qas_df.show(5)
#Because the file is too large, we need to split some data
test_context_qas_df1 = test_context_qas_df.select('qas','length',test_context_qas_df.id.between(0, 99).alias('id'))
test_context_qas_df2= test_context_qas_df.select('qas','length',test_context_qas_df.id.between(50,149).alias('id'))
test_context_qas_df3= test_context_qas_df.select('qas','length',test_context_qas_df.id.between(100,199).alias('id'))
test_context_qas_df4= test_context_qas_df.select('qas','length',test_context_qas_df.id.between(150,199).alias('id'))
test_context_qas_df5= test_context_qas_df.select('qas','length',test_context_qas_df.id.between(200,test_paragraph_df.count()).alias('id'))
#We use a list to store the split data for easy processing
test_context_qas_df=[test_context_qas_df1,test_context_qas_df2,test_context_qas_df3,test_context_qas_df4,test_context_qas_df5]
#Let's continue to expand the JSON file and show some results
for i in range(len(test_context_qas_df)):
    test_context_qas_df[i]=test_context_qas_df[i].select('length','qas')
    test_context_qas_df[i]=test_context_qas_df[i].select('length',explode("qas").alias("qas"))
    test_context_qas_df[i]=test_context_qas_df[i].select('length','qas.*')
    test_context_qas_df[i].count()
test_context_qas_df[0].show(5)
#Next, let's print out the final result
for i in range(len(test_context_qas_df)):
    test_context_qas_df[i]=test_context_qas_df[i].select(explode("answers").alias("answers"),'id','is_impossible','question','length')
    test_context_qas_df[i]=test_context_qas_df[i].select('answers.*','id','is_impossible','question','length')
test_context_qas_df[0].show(5)
test_context_qas_df[0].printSchema()
#Then, we move on to where the answer ends (answer_end)
test_fin_df=test_context_qas_df
for i in range(len(test_fin_df)):
    test_fin_df[i]=test_fin_df[i].withColumn('answer_start' ,when(col('is_impossible') == False, col('answer_start')) \
        .otherwise(0)) \
        .withColumn('answer_end',when(col('is_impossible') == False, col('answer_start')+length("text")) \
        .otherwise(0))
    test_fin_df[i]=test_fin_df[i] \
        .withColumn('answer_end',when((col('is_impossible') == False)&(col('answer_end')<col('length')),col('answer_end')) \
        .otherwise(col('length')))
    test_fin_df[i]=test_fin_df[i] \
        .withColumn('answer_end',when(col('answer_start') != 0,col('answer_start')) \
        .otherwise(0))
test_fin_df[0].show(5)
#Impossible_negative, positive, and Possible_negative were partitioned, and the results of the first segment were preliminaries.
test_fin_Impossible_negative=[]
test_fin_positive=[]
test_fin_Possible_negative=[]

for i in range(len(test_fin_df)):
    test_fin_Impossible_negative.append(0)
    test_fin_positive.append(0)
    test_fin_Possible_negative.append(0)

for i in range(len(test_fin_df)):
    test_fin_Impossible_negative[i]=test_fin_df[i].where(col('is_impossible') == True)
    test_fin_positive[i]=test_fin_df[i].where((col('is_impossible') == False)& (col('answer_end')<=col('length')))
    test_fin_Possible_negative[i]=test_fin_df[i].where((col('is_impossible') == False)& (col('answer_start')==0))

test_fin_Impossible_negative[0].show()
test_fin_positive[0].show()
test_fin_Possible_negative[0].show()
#Next, we process the data as required
for i in range(len(test_fin_df)):
    test_fin_Impossible_negative[i]=test_fin_Impossible_negative[i].withColumnRenamed('id','source').drop("text",'is_impossible','length')
    test_fin_positive[i]=test_fin_positive[i].withColumnRenamed('id','source').drop("text",'is_impossible','length')
    test_fin_Possible_negative[i]=test_fin_Possible_negative[i].withColumnRenamed('id','source').drop("text",'is_impossible','length')

Possible_negative=[]
Impossible_negative=[]
for i in range(len(test_fin_df)):
    Possible_negative.append(0)
    Impossible_negative.append(0)
for i in range(len(test_fin_df)):
    stemp1=test_fin_Impossible_negative[i].groupBy("question").count()
    stemp2=test_fin_Possible_negative[i].groupBy("question").count()
    stemp=stemp1.select('question','count').intersect(stemp2.select("question",'count'))
    Possible_negative[i] = test_fin_Possible_negative[i].join(stemp,on="question", how="inner")
    Impossible_negative[i]=test_fin_Impossible_negative[i].join(stemp,on="question", how="inner")
#stemp1.show(truncate=False)
#stemp2.show(truncate=False)
Possible_Negative=Possible_negative[0]
Impossible_Negative=Impossible_negative[0]
Positive=test_fin_positive[0]
for i in range(len(test_fin_df)-1):
    Impossible_Negative.union(Impossible_negative[i+1])
    Possible_Negative.union(Impossible_negative[i+1])
    Positive.union(test_fin_positive[i+1])
#Process the data in the format required by the teacher
Impossible_Negative=Impossible_Negative.drop('count')
Possible_Negative=Possible_Negative.drop('count')
Positive=Positive.drop('count')
#The output data
Impossible_Negative.write.json('/Users/zhukang/Downloads/new/Impossible_Negative.json')
Possible_Negative.write.json('/Users/zhukang/Downloads/new/Possible_Negative.json')
Positive.write.json('/Users/zhukang/Downloads/new/Positive.json')