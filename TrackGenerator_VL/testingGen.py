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

# data1, tangent_out1, normal_out1=TrackGenerator.add_constant_turn(p0, pn_3, p3,param1)
# data2, tangent_out2, normal_out2=TrackGenerator.add_straight(data1[-1], (tangent_out1), (normal_out1), param2)
# data3, tangent_out3, normal_out3=TrackGenerator.add_constant_turn(data2[-1], tangent_out2,  (normal_out1), param2)
#data3, tangent_out3, normal_out3=TrackGenerator.add_bezier(data2[-1], p5,(tangent_out2),(tangent_out2), param33)


def randomElement(point_in, tangent_in, normal_in, newElement=None):
  """
  Adds new random Track element (if newElement is TRUE then empty track element is not an option)
  """
  if newElement is None:
    newElement=True

  track_element = 0

  finished=False #last track element?
  if newElement:
    
    #functions = [TrackGenerator.random_Bezier, TrackGenerator.add_straight, TrackGenerator.add_constant_turn, TrackGenerator.emptyElement]
    functions = [TrackGenerator.random_Bezier, TrackGenerator.add_straight, TrackGenerator.add_constant_turn]
    i = random.choice(range(len(functions)))
    data_out, tangent_out, normal_out=(functions)[i](point_in, tangent_in, normal_in)

  else:
    functions = [TrackGenerator.random_Bezier, TrackGenerator.add_straight, TrackGenerator.add_constant_turn]
    i = random.choice(range(len(functions)))
    data_out, tangent_out, normal_out=(functions)[i](point_in, tangent_in, normal_in)
    if data_out is None:
      finished=True
  track_element = i

  return data_out, tangent_out, normal_out, finished, track_element

def generate_randomTrack():
  """
  generates random track with max amount of TrackGenerator.MAX_ELEMENTS track elements
  """
  #Starting position
  point_in = [0,0]
  tangent_in = [1,0]
  normal_in = [0,1]
 
  #Trackdata
  track_data = []
  cur_track_data= []

  #track elements
  elementList=[999]
  elementCounter = 0

  # exit conditions
  error=False
  failedCounter = 0
  failedElement = False
  finished = False

  #loop for generating track elemnts
  while finished is False:
    if failedCounter == 10:
      error=True
      break #Generation failed due to to many tries
    if elementCounter == TrackGenerator.MAX_ELEMENTS:
      break #generation finished due to max number of elements
    print(f"counter {elementCounter}")
    cur_track_data=[]
    
    

    
    print(f"TD_len {len(track_data)}")

    
    
    cur_track_data=track_data.copy()

    if failedElement:
      data_out, tangent_out, normal_out, finished, elementType= randomElement(point_in, tangent_in, normal_in)
    else:
      data_out, tangent_out, normal_out, finished, elementType= randomElement(point_in, tangent_in, normal_in)
    print(f"data_out_len {len(data_out)}")
    cur_track_data.extend(data_out[1:])
    #print(f"TD_len {len(track_data)}")
    if check_if_viable(cur_track_data, elementType, elementList[-1]):
      print(f"TD_len {len(track_data)}")
      print("viable")
      failedElement=False
      track_data=cur_track_data
      elementList.append(elementType)
      #prep for new data
      point_in = data_out[-1]
      tangent_in = tangent_out
      normal_in = normal_out
      elementCounter += 1
      continue
    else:
      print("failed")
      print(f"TD_len {len(track_data)}")
      failedCounter += 1
      failedElement=True
      continue
  return track_data, error, elementList


def check_if_viable(toCheck_track_data, newElement, lastElement):
  """
  checks if added track element is viable
  """
  doubleStraight = newElement == lastElement == 1
  return not TrackGenerator.intersectsWithSelf(toCheck_track_data) and not doubleStraight


data, error, _=generate_randomTrack()
print(f"ERROR {error}")
#print(data)
TrackGenerator.visualize(data)
