#!/usr/bin/env python
# coding: utf-8

import json
from os import access
import numpy as np
from numpy.linalg import inv
import rospy
from waterlinked_a50_ros_driver.msg import DVL
from geometry_msgs.msg import Pose, Twist
from bluerov_auto.msg import Acceleration, Depth
from mpl_toolkits import mplot3d
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

class KalmanFilter:
    
    def __init__(self,x, P, u):
        self.x = x
        self.u = u
        self.P = P
        
    def time_update(self, frameTimeDiff):
        # ------------------------------------- Time Update ("Predict") ------------------------------------------------
        millisec_to_sec = 1.0/1000.0
        dt = frameTimeDiff * millisec_to_sec

        # State Transition Matrix
        F = np.array([[1., dt, 0.5*(dt**2), 0., 0., 0., 0., 0., 0.],
                        [0., 1, dt, 0., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., dt, 0.5*(dt**2), 0., 0., 0.],
                        [0., 0., 0., 0., 1., dt, 0., 0., 0.],
                        [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 1., dt, 0.5*(dt**2)],
                        [0., 0., 0., 0., 0., 0., 0., 1., dt],
                        [0., 0., 0., 0., 0., 0., 0., 0., 1.]])

        # Control Matrix
        G = np.transpose(np.array([[0., 0., 0., 0., 0., 0., 0.5, 1, 0]]))

        # Gain Matrix
        # self.u from parameter

        # Process Noise Uncertainty
        Q = np.array([[0.25*(dt**4), 0.5*(dt**3), 0.5*(dt**2), 0., 0., 0., 0., 0., 0.],
                      [0.5*(dt**3), dt**2, dt, 0., 0., 0., 0., 0., 0.],
                      [0.5*(dt**2), dt, 1., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0.25*(dt**4), 0.5*(dt**3), 0.5*(dt**2), 0., 0., 0.],
                      [0., 0., 0., 0.5*(dt**3), dt**2, dt, 0., 0., 0.],
                      [0., 0., 0., 0.5*(dt**2), dt, 1., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0.25*(dt**4), 0.5*(dt**3), 0.5*(dt**2)],
                      [0., 0., 0., 0., 0., 0., 0.5*(dt**3), dt**2, dt],
                      [0., 0., 0., 0., 0., 0., 0.5*(dt**2), dt, 1.]])

        self.x = F@self.x + G@self.u
        self.P = F@self.P@np.transpose(F) + Q
    
    def measurement_update(self, z, frameTimeDiff, measurement_uncertainty):
        
        self.time_update(frameTimeDiff)
        
        millisec_to_sec = 1.0/1000.0
        dt = frameTimeDiff * millisec_to_sec
        
        # ---------------------------------- Measurement Update ("Correct") ---------------------------------------------
        # Observation Matrix
        H = np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 1., 0.]])

        # Identity Matrix
        I = np.eye(9)

        # Measurement Equation
        # z from parameter

        # Measurement Uncertainty
        R = np.eye(4)
        R *= dt*measurement_uncertainty

        # Kalman Gain
        K = self.P@np.transpose(H)@inv(H@self.P@np.transpose(H)+R)

        self.x = self.x + K@(z - H@x)
        self.P = (I - K@H)@self.P@np.transpose(I - K@H) + K@R@np.transpose(K)

        return self.x, self.P


def rosPublishers(pos, vel, acc, depth):
    pos_publisher = rospy.Publisher("/BlueRov/KalmanFilter/Pose", Pose, queue_size=10)
    vel_publisher = rospy.Publisher("/BlueRov/KalmanFilter/Velocity", Twist, queue_size=10)
    acc_publisher = rospy.Publisher("/BlueRov/KalmanFilter/Acceleration", Acceleration, queue_size=10)
    depth_publisher = rospy.Publisher("/BlueRov/KalmanFilter/Depth", Depth, queue_size=10)
    rate = rospy.Rate(1)

    # while not rospy.is_shutdown():
    # prepping pose data
    pose = Pose()
    pose.position.x = pos[0]
    pose.position.y = pos[1]
    pose.position.z = pos[2]

    # zero because no sensor to provide values
    pose.orientation.x = 0
    pose.orientation.y = 0
    pose.orientation.z = 0
    pose.orientation.w = 0


    # prepping velocity data
    velocities = Twist()
    velocities.linear.x = vel[0]
    velocities.linear.y = vel[1]
    velocities.linear.z = vel[2]

    # zero because sensor does not provide these values
    velocities.angular.x = 0
    velocities.angular.y = 0
    velocities.angular.z = 0
    

    # prepping acceleration data
    acceleration = Acceleration()
    acceleration.Acceleration.x = acc[0]
    acceleration.Acceleration.y = acc[1]
    acceleration.Acceleration.z = acc[2]

    # depth data
    rov_depth = Depth()
    rov_depth.Depth = depth

    # log infos
    # rospy.loginfo(velocities)
    # rospy.loginfo(pose)
    # rospy.loginfo(acceleration)
    # rospy.loginfo(rov_depth)

    # publish data
    vel_publisher.publish(velocities)
    pos_publisher.publish(pose)
    acc_publisher.publish(acceleration)
    depth_publisher.publish(rov_depth)

    # sleep for rest of cycle
    rate.sleep()


def callback(data, MyKalmanFilter):
    # get data from different fields
    velocity = data.velocity
    depth = data.altitude

    dis = 0
    for idx, transducer in enumerate(data.beams):
        dis += transducer.distance
    dis /= (idx+1)
    
    z = np.transpose(np.array([[data.velocity.x, data.velocity.y, dis, data.velocity.z]]))        # position measurement from DVL [vx, vy, alt, vz]
    frameTimeDiff = data.time                                         # time measurement from DVL
    measurement_uncertainty = data.fom                               # fom measurement from DVL
    
    pos, E_uncertainty = MyKalmanFilter.measurement_update(z, frameTimeDiff, measurement_uncertainty)
    pos = pos.flatten()
    # print(pos)
    
    position = np.take(pos, [0,3,6])
    velocity = np.take(pos, [1,4,7])
    acceleration = np.take(pos, [2,5,8])
    
    # call ROS publishers
    rosPublishers(position, velocity, acceleration, depth)
    # print("Position: {}\nVelocity: {}\nAcceleration: {}\nDepth: {}\n".format(position, velocity, acceleration, i["altitude"]))


def listener():
    rospy.init_node("KalmanFilterNode")

    rospy.Subscriber("/dvl/data", DVL, callback)

    rospy.spin()




# ------------------------------------------------- Static Initial Settings -------------------------------------------------------


if __name__ == '__main__':
    # Initial State Vector [0 ... 0]
    x = np.zeros((9, 1))

    # Estimate Uncertainty
    P = np.eye(9)*500

    # Gain Matrix
    # u = np.array([[-9.8]])
    u = np.array([[0]])

    f = open('out.json')
    data = json.load(f)
    # np.set_printoptions(precision=4, suppress=True)

    MyKalmanFilter = KalmanFilter(x, P, u)
    
    #ROS Nodes
    listener()


