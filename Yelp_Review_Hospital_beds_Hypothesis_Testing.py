import random
import numpy as np
import math
import hashlib
import csv
import sys
import json
import re
from datetime import datetime
from scipy import stats
from pyspark import SparkContext, SparkConf
zip_codes = ["89109", "89118", "15237", "44122", "44106"]
conf = SparkConf().setAppName("appName")
sc = SparkContext(conf=conf)
review_rdd = sc.textFile(sys.argv[1], 32)
review_rdd = review_rdd.map(json.loads)
business_rdd = sc.textFile(sys.argv[2], 32)
business_rdd = business_rdd.map(json.loads)
Hospital_rdd = sc.textFile(sys.argv[3], 32)
header = Hospital_rdd.first()
header_list = header.split(",")
header_dict = {}
i=0
for ele in header_list:
    header_dict[ele] = i
    i+=1
def csv_reader(x):
    temp_list=[]
    temp_list.append(x)
    input_line=list(csv.reader(temp_list,delimiter=','))
    input_line = input_line[0]
    return input_line

Hospital_rdd = Hospital_rdd.filter(lambda row : row != header).map(csv_reader) 
dictionary_rdd = sc.textFile(sys.argv[4], 32)
dictionary = dictionary_rdd.collect()
for i in range(len(dictionary)):
    dictionary[i]=dictionary[i][1:-1]
def more_30(x):
    val = x[header_dict["total_beds_7_day_avg"]]
    if len(val)>0 and float(x[header_dict["total_beds_7_day_avg"]])>30:
        return True
    return False

def cal_mean_bed_usage_pct(x):
    n=0
    s=0
    zip=''
    for ele in x[1]:
        zip=ele[0]
        n+=1
        s+=ele[1]
    return (x[0],zip,s/n)
def mean_zip(x):
    n=0
    s=0
    for ele in x[1]:
        s+=ele
        n+=1
    return (x[0],s/n)


checkpoint_3_1_1_rdd = Hospital_rdd.filter(more_30).map(lambda x: (x[header_dict["hospital_pk"]],x[header_dict["zip"]],x[header_dict["inpatient_beds_used_7_day_avg"]],x[header_dict["total_beds_7_day_avg"]]))
checkpoint_3_1_2_rdd = checkpoint_3_1_1_rdd.map(lambda x: (x[0],x[1],float(x[2]) if (x[2] not in ['','-999999']) else 0,x[3])).filter(lambda x: x[2]!=-999999 or x[3]!=-999999).map(lambda x: (x[0],[[x[1],float(x[2])/float(x[3])]])).reduceByKey(lambda x,y: x+y).map(cal_mean_bed_usage_pct)
checkpoint_3_1_3_rdd = checkpoint_3_1_2_rdd.map(lambda x: (x[1],[x[2]])).reduceByKey(lambda x,y:x+y).map(mean_zip)
checkpoint_3_1_3_rdd.persist()
print("**********************************************")
print("Step 3.1: Aggregate outcome data to zip codes: (Zip, value)")
print(checkpoint_3_1_3_rdd.filter(lambda x: x[0] in zip_codes).collect())
print("**********************************************")
# review_rdd.map(lambda x: (x["business_id"],x["text"]))
# def text_to_vector(x):
#     res=[]
#     zip_code = x[0]
#     text_list = x[1].split()
#     i=0
#     for v in dictionary:
#         d=0
#         if v[-1]=="*":
#             for t in text_list:
#                 if t.startswith(v[:-1]):
#                     d=1
#                     break
#         else:
#             if v in text_list:
#                 d=1
#         i+=1
#         res.append(((zip_code,v),d))
#     return res
def text_to_vector(x):
    res=[]
    zip_code = x[0]
    text_list = x[1]
    for v in dictionary:
        if re.search(v,text_list)!=None:
            res.append(((zip_code,v),[1]))
        else:
            res.append(((zip_code,v),[0]))
    return res

checkpoint_3_2_1_rdd = business_rdd.map(lambda x: (x["business_id"],x["postal_code"])).join(review_rdd.map(lambda x: (x["business_id"],x["text"])))
checkpoint_3_2_2_rdd = checkpoint_3_2_1_rdd.map(lambda x : (x[1][0],x[1][1])).filter(lambda x: len(x[1])>256)
checkpoint_3_2_3_rdd = checkpoint_3_2_2_rdd.flatMap(text_to_vector).reduceByKey(lambda x,y:x+y).map(lambda x: (x[0],sum(x[1])/len(x[1])))
checkpoint_3_2_3_rdd.persist()
checkpoint_3_2_rdd = checkpoint_3_2_3_rdd.map(lambda x: (x[0][0],[[x[0][1],x[1]]])).reduceByKey(lambda x,y:x+y)
print("**********************************************")
print("Step 3.2: Aggregate word usage per zip code:")
print(checkpoint_3_2_rdd.filter(lambda x: x[0] in zip_codes).map(lambda x: (x[0], [i for i in sorted(x[1],key=lambda z:z[1],reverse=True)][:5])).collect())
print("**********************************************")


def mean_centric(x):
    val=0
    n=len(x[1])
    for j in x[1]:
        val+=float(j[1])
    m = val/n
    for j in x[1]:
        j[1] = [j[1],j[1]-m]
    return x


def cosine_sim(x,y):
    res={}
    for i in x:
        if i[0] not in res:
            res[i[0]] = []
        res[i[0]].append((i[1],1))
    for j in y:
        if j[0] not in res:
            res[j[0]]=[]
        res[j[0]].append((j[1][1],2))
    n=0
    x1=0
    x2=0
    for key,val in res.items():
        if len(val)==2:
            n+=val[0][0]*val[1][0]
            x1+=val[0][0]*val[0][0]
            x2+=val[1][0]*val[1][0]
        else:
            if val[0][1]==1:
                x1+=val[0][0]*val[0][0]
            else:
                x2+=val[0][0]*val[0][0]
    return n/(math.sqrt(x1)*math.sqrt(x2)) if x1!=0 and x2!=0 else 0

def mean_center_list(x):
    val=0
    n=len(x)
    for j in x:
        val+=float(j[1])
    m=val/n
    for j in x:
        j[1]-=m
    return x
def keep_original(x):
    for j in x:
        j[1]=j[1][0]
    return x

hospital_zip_val = checkpoint_3_1_3_rdd.map(lambda x: [x[0],x[1]]).collect()
hospital_zip_val_mean_center = mean_center_list(hospital_zip_val)
word_zip_val_rdd = checkpoint_3_2_3_rdd.map(lambda x: (x[0][1],[[x[0][0],x[1]]])).reduceByKey(lambda x,y:x+y).map(mean_centric)
checkpoint_3_3_1_rdd = word_zip_val_rdd.map(lambda x: (x[0], x[1],cosine_sim(hospital_zip_val_mean_center,x[1]))).map(lambda x:(x[0],keep_original(x[1]),x[2]))
# print(checkpoint_3_3_1_rdd.take(1))

def get_p_val(v1,v2):
    v1=[i[1] for i in v1]
    v2=[i[1] for i in v2]
    return stats.ttest_ind(v1, v2)[1]

checkpoint_3_3_2_rdd = checkpoint_3_3_1_rdd.map(lambda x: (x[0],x[2],get_p_val(x[1],hospital_zip_val))).map(lambda x: (x[0],x[1],x[2],x[2]*len(dictionary)))

print("Step 3.3: Calculate correlations between word usage and mean_bed_usage_pct:")
print(" top 20 most positively correlated words: (word,Similarity,p_value,Bonferonni corrected p-value")
print(checkpoint_3_3_2_rdd.sortBy(lambda x: x[1],ascending=False).take(20))
print("**********************************************")
print(" top 20 most positively correlated words: (word,Similarity,p_value,Bonferonni corrected p-value")
print(checkpoint_3_3_2_rdd.sortBy(lambda x: x[1]).take(20))
print("**********************************************")