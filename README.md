# Bluerov Automation Repo

Group members:
- Peter-Newman Messan
- Aron Jinga

This repo houses scripts used in the pursuit of automating the Bluerov 2 robot. The repo is organised as follows:


* src: houses KalmanFilter.py, sample output from DVL, and a script attempting to implement optical flow.
* msg: holds custom messages (Acceleration and Depth) created for the BlueROV
* launch: contains the launch file which attempts to spawn multiple nodes over multiple machines. (currently creates an ssh connection, but spawned node dies.)


