from VL_trackGenerator import *
import random
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
#   "length": 20,
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



#print(TrackGenerator.check_if_overlap(data1))



# data1, tangent_out1, normal_out1=TrackGenerator.add_constant_turn(p0, pn_3, p3,param1)
# data2, tangent_out2, normal_out2=TrackGenerator.add_straight(data1[-1], (tangent_out1), (normal_out1), param2)
# data3, tangent_out3, normal_out3=TrackGenerator.add_constant_turn(data2[-1], tangent_out2,  (normal_out1), param2)
#data3, tangent_out3, normal_out3=TrackGenerator.add_bezier(data2[-1], p5,(tangent_out2),(tangent_out2), param33)
# print(len(data1))
# print(len(data2))
# print(len(data3))
# data1=data1[:-1]
# data1.extend(data2[:-1])
# data1.extend(data3[:-1])

def randomElement(point_in, tangent_in, normal_in, newElement=None):
  """
  Adds new random Track element (if newElement is TRUE then empty track element is not an option)
  """
  if newElement is None:
    newElement=True

  finished=False #last track element?
  if newElement:
    functions = [TrackGenerator.random_Bezier, TrackGenerator.add_straight, TrackGenerator.add_constant_turn]
    data_out, tangent_out, normal_out=random.choice(functions)(point_in, tangent_in, normal_in)
  else:
    functions = [TrackGenerator.random_Bezier, TrackGenerator.add_straight, TrackGenerator.add_constant_turn]
    data_out, tangent_out, normal_out=random.choice(functions)(point_in, tangent_in, normal_in)
    if data_out is None:
      finished=True
  return data_out, tangent_out, normal_out, finished

def generate_randomTrack():
  """
  generates random track with max amount of TrackGenerator.MAX_ELEMENTS track elements
  """
  #Starting position
  point_in = [0,0]
  tangent_in = [1,0]
  normal_in = [0,1]
  finished = False
  track_data = []
  cur_track_data= []
  elementCounter = 0
  failed = False
  
  while finished is False:
    if elementCounter == 4:
      break
    data_out, tangent_out, normal_out, finished = randomElement(point_in, tangent_in, normal_in)
    cur_track_data=track_data
    cur_track_data.extend(data_out[1:])
    
    if check_if_viable(cur_track_data):
      failed=False
      track_data=cur_track_data
      #prep for new data
      point_in = data_out[-1]
      tangent_in = tangent_out
      normal_in = normal_out
      elementCounter += 1
      continue
    else:
      failed=True
      continue
  return track_data


def check_if_viable(track_data):
  """
  checks if added track element is viable
  """
  return True

TrackGenerator.visualize(generate_randomTrack())

#data1=list(set(map(tuple, data1)))
# print(len(data1))
# print(TrackGenerator.intersectsWithSelf(data1))

# TrackGenerator.visualize(data1)