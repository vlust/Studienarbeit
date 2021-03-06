from VL_trackGenerator import *
import random
# import matplotlib.pyplot as plt
import os
import json
import pandas as pd


# p0=[0,0]
# p1=[1,0]
# pn_1=[0,1]
# pn=[30,30]
# control_points = [p0, p1, pn_1, pn]

# p2=[0,0]
# p3=[0,1]
# pn_3=[1,0]
# pn=[30,30]

# p5=[60,20]
# p6=[-60,30]

# param1 = {
#   "turn_against_normal": False,
#   "radius": 30,
#   "circle_percent": 0.3
# }
# param11 = {
#   "turn_against_normal": False,
#   "radius": 30,
#   "circle_percent": 0.12
# }


# param2 = {
#   "false_element": 0,
# }

# param3 = {
#   "scale_in": 20,
#   "scale_out": 20,

# }
# param33 = {
#   "scale_in": 100,
#   "scale_out": 100,

# }

######################## DATA ############################

# data1,tangent_out1,normal_out1=TrackGenerator.add_constant_turn(p0, pn_3, p3,param1)
# data2, tangent_out2, normal_out2=TrackGenerator.add_straight(data1[-1], (tangent_out1), (normal_out1), param2)
#data1, tangent_out3, normal_out3=TrackGenerator.add_bezier([0,0], [30,30],[0,1],[1,0], param3)
# data1, tangent_out3, normal_out3=TrackGenerator.random_Bezier([0,0], [1,0])

# my_list = [TrackGenerator.random_Bezier, TrackGenerator.add_straight, TrackGenerator.add_constant_turn]
# data1, tangent_out3, normal_out3=random.choice(my_list)(p0, p1, pn_1)
# data2, tangent_out2, normal_out2=TrackGenerator.add_straight(data1[-1], (tangent_out3), (normal_out3), param2)

# data1, tangent_out1, normal_out1=TrackGenerator.add_constant_turn(p0, pn_3, p3,param1)
# data2, tangent_out2, normal_out2=TrackGenerator.add_straight(data1[-1], (tangent_out1), (normal_out1), param2)
# data3, tangent_out3, normal_out3=TrackGenerator.add_constant_turn(data2[-1], tangent_out2,  (normal_out1), param2)
#data3, tangent_out3, normal_out3=TrackGenerator.add_bezier(data2[-1], p5,(tangent_out2),(tangent_out2), param33)

# def cones_to_dict(cones):
#     dict_list=[]
#     keys=['x', 'y', 'color']
#     for cone in cones:
#         dict_list.append(dict(zip(keys, cone)))
#     return dict_list


# my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.

data, cones, elements, error = TrackGenerator.generate_random_local_track()
print(f"ERROR {error}")
# track=[]
# for point in data:
#     point_l=list(point)
#     point_l.extend('M')
#     track.append(point_l)

# df = pd.DataFrame(cones, columns =['x', 'y', 'color'])
# df.append(data[0:len(data):5])
# print(df)
# print(cones)
print(f"ELEMENTS {elements}")
TrackGenerator.visualize_all(data, cones)


# # Data to be written
# aList = cones_to_dict(cones)


# track_obj ={
#     "elements" : elements,
#     "key_to_csv" : 1,
#     "cones":aList
# }

# # Serializing json
# json_object = json.dumps(track_obj, indent = 4)

# # Writing to sample.json
# with open(my_path+"/&sample.json", "w") as outfile:
#     outfile.write(json_object)

# with open(my_path+"/&sample.json", "r") as file:
#     jObj=json.load(file)
# TrackGenerator.show_track(data)
