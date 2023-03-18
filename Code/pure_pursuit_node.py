#!/usr/bin/env python
import math, os, csv, tf, rospy
import numpy as np
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray

# Tunable parameters
LOOKAHEAD_DISTANCE = 1.6 # Optimal value: 1.6
KP = 0.3 # Optimal value: 0.3 (this is approximately the distance between the front and rear axles of the car)
MAX_SPEED = 2.7 # Optimal value: 2.7

# Modes
SLOW_MODE = False # Use this mode to drive slowly at constant speed, for testing
WAYPOINT_EDITING = False # Use this mode to drive slowly at constant speed, for testing

class PurePursuit(object):
    """ The class that handles pure pursuit """
    def __init__(self):
        self.trajectory_name = 'mp_assignment' # the file containing the waypoints to track
        self.plan = self.construct_path()
        
        drive_topic = '/vesc/ackermann_cmd_mux/input/navigation'
        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size=10)

        pf_odom_topic = '/pf/pose/odom'
        self.pf_odom_sub = rospy.Subscriber(pf_odom_topic, Odometry, self.odom_callback, queue_size=10)

        wp_viz_topic = '/wp_viz'
        self.nearest_wp_pub = rospy.Publisher(wp_viz_topic+'/nearest', Marker, queue_size = 10)
        self.all_wp_pub = rospy.Publisher(wp_viz_topic+'/all', MarkerArray, queue_size = 10)

    def odom_callback(self, odom_msg):
        """ Callback function for the subcriber of the localization-inferred odometry """
        # Show all waypoints
        markerArray = self.create_waypoint_marker_array()
        self.all_wp_pub.publish(markerArray)
        
        # Find nearest waypoint
        curr_x = odom_msg.pose.pose.position.x
        curr_y = odom_msg.pose.pose.position.y
        nearest_waypoint_idx = self.find_nearest_point(curr_x, curr_y)

        # Lookahead to another waypoint, which we'll pursue
        lookahead_waypoint_idx = nearest_waypoint_idx
        while math.sqrt(math.pow(self.plan[lookahead_waypoint_idx][0]-curr_x,2) + math.pow(self.plan[lookahead_waypoint_idx][1]-curr_y,2)) < LOOKAHEAD_DISTANCE:
            lookahead_waypoint_idx += 1
            if (lookahead_waypoint_idx > len(self.plan) - 1):
                lookahead_waypoint_idx = 0
        
        # Show the lookahead waypoint
        marker = self.create_waypoint_marker(lookahead_waypoint_idx, nearest_wp=True)
        self.nearest_wp_pub.publish(marker)

        # Calculate the current heading of the car
        heading = tf.transformations.euler_from_quaternion((odom_msg.pose.pose.orientation.x,
                                                        odom_msg.pose.pose.orientation.y,
                                                        odom_msg.pose.pose.orientation.z,
                                                        odom_msg.pose.pose.orientation.w))[2]

        # Calculate the Euclidean distance between the lookahead waypoint and the car
        goal_x = self.plan[lookahead_waypoint_idx][0]
        goal_y = self.plan[lookahead_waypoint_idx][1]
        eucl_d = math.sqrt(math.pow(goal_x - curr_x, 2) + math.pow(goal_y - curr_y, 2))

        # Calculate the change in y-coordinate between the car and the goal point
        lookahead_angle = math.atan2(goal_y - curr_y, goal_x - curr_x)
        del_y = eucl_d * math.sin(lookahead_angle - heading)

        # Calculate the curvature of the arc from the car to the waypoint, then get the steering angle from that value (using P control)
        curvature = 2.0*del_y/(math.pow(eucl_d, 2))
        steering_angle = KP * curvature

        # Wrap the steering angle to be between -90 and +90 degrees
        while (steering_angle > np.pi/2) or (steering_angle < -np.pi/2):
            if steering_angle > np.pi/2:
                steering_angle -= np.pi
            elif steering_angle < -np.pi/2:
                steering_angle += np.pi
        
        # Prepare the drive command for pursuing that waypoint
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steering_angle)
        print("Steering Angle: {:.3f} [deg]".format(np.rad2deg(steering_angle)))
        drive_msg.drive.speed = self.get_velocity(steering_angle)
        if not WAYPOINT_EDITING: # If this mode is on, the car will not move - this is good for refining the waypoints
            self.drive_pub.publish(drive_msg)

    def create_waypoint_marker(self, waypoint_idx, nearest_wp=False):
        """ Given the index of the nearest waypoint, publishes the necessary Marker data to the 'wp_viz' topic for RViZ visualization """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = 2 # sphere
        marker.action = 0 # add the marker
        marker.pose.position.x = self.plan[waypoint_idx][0]
        marker.pose.position.y = self.plan[waypoint_idx][1]
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

    def create_waypoint_marker_array(self):
        """ Creates a MarkerArray message for all waypoints, to be published to the 'wp_viz' topic for RViZ visualization """
        markerArray = MarkerArray()
        for i in range(len(self.plan)):
            marker = self.create_waypoint_marker(i)
            markerArray.markers.append(marker)
        id = 0
        for m in markerArray.markers:
            m.id = id
            id += 1
        return markerArray

    def get_velocity(self, steering_angle):
        """ Given the desired steering angle, returns the appropriate velocity to publish to the car """
        if SLOW_MODE:
            return 0.8 # [m/s], roughly the minimum speed that the car drives smoothly at
        velocity = max(MAX_SPEED - abs(np.rad2deg(steering_angle))/50, 0.8) # [m/s], velocity varies smoothly with steering angle
        print('Velocity: ' + str(velocity))
        return velocity

    def find_nearest_point(self, curr_x, curr_y):
        """ Given the current XY position of the car, returns the index of the nearest waypoint """
        ranges = []
        for index in range(0, len(self.plan)):
            eucl_x = math.pow(curr_x - self.plan[index][0], 2)
            eucl_y = math.pow(curr_y - self.plan[index][1], 2)
            eucl_d = math.sqrt(eucl_x + eucl_y)
            ranges.append(eucl_d)
        return (ranges.index(min(ranges)))

    def construct_path(self):
        """ Reads waypoint data from the .csv file, inserts it into an array called 'plan' """
        file_path = os.path.expanduser('~/f1tenth_ws/logs/{}.csv'.format(self.trajectory_name))
        plan = []
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            for waypoint in csv_reader:
                plan.append(waypoint)
        for index in range(0, len(plan)):
            for point in range(0, len(plan[index])):
                plan[index][point] = float(plan[index][point])
        return plan

def main():
    rospy.init_node('pure_pursuit_node')
    pp = PurePursuit()
    rospy.spin()

if __name__ == '__main__':
    print("Pure Pursuit Mode Initialized...")
    main()