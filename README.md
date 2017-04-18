# Baxter Force/Velocity Control

This ROS package contains work-in-progress code for various force/velocity control tasks on Baxter.

The only really useful things here at the moment are in baxter_velocity_control.py. This file contains code for endpoint velocity control, as well as a closed loop endpoint pose controller built on top of the velocity control class. Running the file itself runs a demo which moves both of Baxter's arms to follow a single motion capture marker in real time.
