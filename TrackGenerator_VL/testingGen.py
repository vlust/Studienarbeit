from VL_trackGenerator import *
import random
p0=[0,0]
p1=[1,0]
pn_1=[0,1]
pn=[30,30]
control_points = [p0, p1, pn_1, pn]

p2=[0,0]
p3=[0,1]
pn_3=[1,0]
pn=[30,30]

p5=[60,20]
p6=[-60,30] 

param1 = {
  "turn_against_normal": False,
  "radius": 30,
  "circle_percent": 0.3
}
param11 = {
  "turn_against_normal": False,
  "radius": 30,
  "circle_percent": 0.12
}


param2 = {
  "length": 20,
}
 
param3 = {
  "scale_in": 20,
  "scale_out": 20,
  
}
param33 = { 
  "scale_in": 100,
  "scale_out": 100,
  
}

######################## DATA ############################

# data1,tangent_out1,normal_out1=TrackGenerator.add_constant_turn(p0, pn_3, p3,param1)
# data2, tangent_out2, normal_out2=TrackGenerator.add_straight(data1[-1], (tangent_out1), (normal_out1), param2)
data1, tangent_out3, normal_out3=TrackGenerator.add_bezier([0,0], [30,30],[0,1],[1,0], param3)
# data1, tangent_out3, normal_out3=TrackGenerator.random_Bezier([0,0], [1,0])

# my_list = [TrackGenerator.random_Bezier, TrackGenerator.add_straight, TrackGenerator.add_constant_turn]
# data1, tangent_out3, normal_out3=random.choice(my_list)(p0, p1, pn_1)
p1 = data1[256]
p2 = data1[256+1]

                        
p3 = data1[257]
p4 = data1[257+1]

print(TrackGenerator.doIntersect(p1, p2, p3, p4))

#print(TrackGenerator.check_if_overlap(data1))
#print(TrackGenerator.intersectsWithSelf(data1))


# data1, tangent_out1, normal_out1=TrackGenerator.add_constant_turn(p0, pn_3, p3,param1)
# data2, tangent_out2, normal_out2=TrackGenerator.add_straight(data1[-1], (tangent_out1), (normal_out1), param2)
# data3, tangent_out3, normal_out3=TrackGenerator.add_constant_turn(data2[-1], tangent_out2,  (normal_out1), param2)
#data3, tangent_out3, normal_out3=TrackGenerator.add_bezier(data2[-1], p5,(tangent_out2),(tangent_out2), param33)


# data1.extend(data2)
# data1.extend(data3)


TrackGenerator.visualize(data1)