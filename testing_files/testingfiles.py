
# import matplotlib.pyplot as plt
import os

my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
# my_file = '/graph.png'
 
# print(my_path)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(range(100))

# fig.savefig(my_path+ my_file)

import json
  
# Data to be written

aList = [{'a':2, 'b':2}, {'c':3, 'd':4}]
jsonStr = json.dumps(aList)

dictionary ={
    "name" : "sathiyajith",
    "rollno" : 55,
    "cgpa" : 8.6,
    "phonenumber" : "9976770500",
    "cones":aList
}

# Serializing json 
json_object = json.dumps(dictionary, indent = 4)
  
# Writing to sample.json
with open(my_path+"/&sample.json", "w") as outfile:
    outfile.write(json_object)

# input("continue")
with open(my_path+"/&sample.json", "r") as file:
    jObj=json.load(file)
print(jObj['cones'][0]['a'])