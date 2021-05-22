#!/usr/bin/env python3
import numpy as np
from numpy.linalg import inv

# Kalman Filter Function
def KalmanFilter(x, frameTimeDiff, gainMatrix, P_scale = 500):
    
    # Step 1: Initial Estimation
    P = np.eye(9)
    P *= P_scale

    # Step 2: Time Update ("Predict")
    millisec_to_sec = 1.0/1000.0
    dt = frameTimeDiff * millisec_to_sec
    
    # Physics Mechanics Function
    F = np.array([[1., dt, 0.5*(dt**2), 0., 0., 0., 0., 0., 0.],
                    [0., 1, dt, 0., 0., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 1., dt, 0.5*(dt**2), 0., 0., 0.],
                    [0., 0., 0., 0., 1., dt, 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 1., dt, 0.5*(dt**2)],
                    [0., 0., 0., 0., 0., 0., 0., 1., dt],
                    [0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    
    
    G = np.transpose(np.array([[0., 0., 0., 0., 0., 0., 0.5, 1, 0]]))
    u = gainMatrix
    Q = np.array([[0.25*(dt**4), 0.5*(dt**3), 0.5*(dt**2), 0., 0., 0., 0., 0., 0.],
                  [0.5*(dt**3), dt**2, dt, 0., 0., 0., 0., 0., 0.],
                  [0.5*(dt**2), dt, 1., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0.25*(dt**4), 0.5*(dt**3), 0.5*(dt**2), 0., 0., 0.],
                  [0., 0., 0., 0.5*(dt**3), dt**2, dt, 0., 0., 0.],
                  [0., 0., 0., 0.5*(dt**2), dt, 1., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0.25*(dt**4), 0.5*(dt**3), 0.5*(dt**2)],
                  [0., 0., 0., 0., 0., 0., 0.5*(dt**3), dt**2, dt],
                  [0., 0., 0., 0., 0., 0., 0.5*(dt**2), dt, 1.]])

    x = F@x + G@u
    P = F@P@np.transpose(F) + Q

    # Step 3: Measurement Update ("Correct")
    H = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 1., 0., 0.]])
    I = np.eye(9)
    R = np.eye(3)
    R *= dt*2.707
    z = np.transpose(np.array([[0.007*frameTimeDiff, 0.017*frameTimeDiff, 0.006*frameTimeDiff]]))

    K = P@np.transpose(H)@inv(H@P@np.transpose(H)+R)
    x = x + K@(z - H@x)
    P = (I - K@H)@P@np.transpose(I - K@H) + K@R@np.transpose(K)

    return x


# Initial Setup
x = np.zeros((9, 1))
gainMatrix = np.array([[-9.8]])

print(x)
for i in range(10):
    x = KalmanFilter(x, frameTimeDiff, gainMatrix)
    print(x)

