#!/usr/bin/env python
import rospy, math
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

class PurePursuit(object):
    """
    The class that handles pure pursuit.
    """
    def __init__(self, plan, lookahead_distance=0.4, kp=0.3):
        self.plan = plan
        self.lookahead_distance = lookahead_distance
        self.kp = kp

    def pursue(self, curr_x, curr_y, heading):
        # Choose next waypoint to pursue (lookahead)
        i = 0
        while math.sqrt(math.pow(self.plan[i][0]-curr_x, 2) + math.pow(self.plan[i][1]-curr_y,2)) < self.lookahead_distance:
            i += 1
            if (i > len(self.plan) - 1):
                i = 0
        marker = self.create_waypoint_marker(self.plan[i][0], self.plan[i][1], nearest_wp=True)

        goal_x = self.plan[i][0]
        goal_y = self.plan[i][1]
        eucl_d = math.sqrt(math.pow(goal_x - curr_x, 2) + math.pow(goal_y - curr_y, 2))

        lookahead_angle = math.atan2(goal_y - curr_y, goal_x - curr_x)
        del_y = eucl_d * math.sin(lookahead_angle - heading)

        curvature = 2.0*del_y/(math.pow(eucl_d, 2))
        steering_angle = self.kp * curvature
        while (steering_angle > np.pi/2) or (steering_angle < -np.pi/2):
            if steering_angle > np.pi/2:
                steering_angle -= np.pi
            elif steering_angle < -np.pi/2:
                steering_angle += np.pi

        # Prepare the drive command for pursuing that waypoint
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steering_angle)
        sa_deg = np.rad2deg(steering_angle)
        print("Steering Angle: {:.3f} [deg]".format(sa_deg))
        drive_msg.drive.speed = self.get_velocity(steering_angle)

        return drive_msg, marker

    def create_waypoint_marker(self, wp_x, wp_y, nearest_wp=False):
        """Given the index of the nearest waypoint, publishes the necessary Marker data to the 'wp_viz' topic for RViZ visualization"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = 2 # sphere
        marker.action = 0 # add the marker
        marker.pose.position.x = wp_x #self.plan[waypoint_idx][0]
        marker.pose.position.y = wp_y #self.plan[waypoint_idx][1]
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        
        # different color/size for the whole waypoint array and the nearest waypoint
        if nearest_wp:
            marker.scale.x *= 2
            marker.scale.y *= 2
            marker.scale.z *= 2
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        return marker

    def get_velocity(self, steering_angle):
        """ Given the desired steering angle, returns the appropriate velocity to publish to the car """
        slow = True
        if slow == True:
            return 0.5
        if abs(steering_angle) < np.deg2rad(10):
            velocity = 1.2 #2.6 #2.5
        elif abs(steering_angle) < np.deg2rad(20):
            velocity = 1.0 #2.2
        else:
            velocity = 0.8 #1.4
        return velocity