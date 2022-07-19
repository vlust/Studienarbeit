import rospy
import rospkg
from std_msgs.msg import Header
from utilities.msg import PlanningResults, TrajectoryPoints

import numpy as np
import filtering
import planning
from velocity import Velocity
from global_optimization import optimizing_Handler
import time

class GlobalMotionPlanning():
    def __init__(self):
        rospack = rospkg.RosPack()
        self.toppath = rospack.get_path('utilities')

        self.publisher = rospy.Publisher('global_trajectory_points', TrajectoryPoints, queue_size=10)
        self.subscriber_slam = rospy.Subscriber('/planning_results', PlanningResults, self.callback)
        self.subscriber_velo = rospy.Subscriber('/velocity', float)
        #eigentlich auf Slam subscriben noch einfügen

    def callback(self, data):
        t0 = time.time()
        #Platzhalter Filtering

        

        #ende
        msg = TrajectoryPoints()
        x = []
        y = []
        for i in range(len(data.spline)):
            x.append(data.spline[i].x)
            y.append(data.spline[i].y)
        x = np.array(x)
        x = np.reshape(x, (-1,1))
        y = np.array(y)
        y = np.reshape(y, (-1,1))

        
        rospy.loginfo("[Global_Motion_Planning] Gloabl Process starting")
        optimizer = optimizing_Handler()
        left_cones, right_cones, orange_cones = optimizer.read_Data(data.spline)
        raceline, raceline_distance, raceline_curvature = optimizer.run(left_cones, right_cones, orange_cones)
        velocity = Velocity(g_x_max=1.5, g_y_max=1.5)
        optimal_velocity = velocity.run(raceline, raceline_curvature, raceline_distance, v_initial=self.subscriber_velo, plot=False)
        optimal_velocity = np.reshape(optimal_velocity, (len(raceline),1))
        global_trajectory = np.append(raceline, optimal_velocity, axis=1)
        msg.x = global_trajectory[:,0]
        msg.y = global_trajectory[:,1]
        msg.v = global_trajectory[:,2]

        self.publisher.publish(msg)
        rospy.loginfo("[Global Motion Planning] Finished Global Motion Planning Zyclus")

if __name__ == '__main__':
    rospy.init_node('global_motion_planning', anonymous=False)
    rospy.loginfo("[Global_Motion_Planning] Global Motion Planning Node started")
    GlobalMotionPlanning()
    rospy.spin()

