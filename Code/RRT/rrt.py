#!/usr/bin/env python
import rospy, tf
import numpy as np

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import OccupancyGrid

from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

from RRTMotionPlanner import RRTMotionPlanner
from rrt_pp import PurePursuit

class RRT(object):
    def __init__(self):
        pf_topic = '/odom'
        scan_topic = '/scan'
        drive_topic = '/drive'
        wp_viz_topic = '/wp_viz'

        self.occ_grid = None

        self.occ_pub = rospy.Publisher('/occ_grid', OccupancyGrid, queue_size = 10)
        self.width = 25
        self.height = 25
        self.resolution = 0.1
        self.footprint = 0.3 # square size of the car footprint [m]
        self.car_pose = tf.TransformListener() # listener of transforms between the car_base_link and the world frame
        
        self.curr_pos = [0,0]
        self.curr_orientation = [0,0,0,1]

        rospy.Subscriber(scan_topic, LaserScan, self.scan_callback)

        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size=10)
        self.pf_odom_sub = rospy.Subscriber(pf_topic, Odometry, self.pf_callback, queue_size=10)

        self.nearest_wp_pub = rospy.Publisher(wp_viz_topic+'/nearest', Marker, queue_size = 10)
        self.goal_wp_pub = rospy.Publisher(wp_viz_topic+'/goal', Marker, queue_size = 10)
        self.plan_pub = rospy.Publisher(wp_viz_topic+'/plan', Marker, queue_size = 10)
    
    def scan_callback(self, scan_msg):
        """ LaserScan callback, the occupancy grid is updated here """
        # Create a local occupancy grid for the car, where the car's current location is at (x,y)=(0, WIDTH/2)
        grid = np.ndarray((self.width, self.height), buffer=np.zeros((self.width, self.height), dtype=np.int), dtype=np.int)
        grid.fill(int(-1))

        angle_min = scan_msg.angle_min
        angle_max = scan_msg.angle_max
        dist = list(scan_msg.ranges) # laser frame
        angles = np.linspace(angle_min, angle_max, num = len(dist)) # laser frame

        t = self.car_pose.getLatestCommonTime('/base_link', '/laser_model')
        lidar_position, _ = self.car_pose.lookupTransform('/base_link', '/laser_model', t)
        
        self.set_obstacle(grid, lidar_position, angles, dist)
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = 'map'
        map_msg.header.stamp = rospy.Time.now()
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.width
        map_msg.info.height = self.height
        map_msg.info.origin.position.x = self.curr_pos[0]
        map_msg.info.origin.position.y = self.curr_pos[1] - self.height // 2 * self.resolution
        map_msg.info.origin.orientation.x = self.curr_orientation[0]
        map_msg.info.origin.orientation.y = self.curr_orientation[1]
        map_msg.info.origin.orientation.z = self.curr_orientation[2]
        map_msg.info.origin.orientation.w = self.curr_orientation[3]
        map_msg.data = range(self.width*self.height)
        for i in range(self.width*self.height):
            map_msg.data[i] = grid.flat[i]  
        self.occ_grid = map_msg
        self.occ_pub.publish(map_msg)

    def set_obstacle(self, grid, lidar_position, angles, dist):
        for d in range(len(dist)):
            x = dist[d]*np.cos(angles[d]) # x coordinate in lidar frame
            y = dist[d]*np.sin(angles[d]) # y coordinate in lidar frame
            x -= lidar_position[0] # x coordinate in base link frame
            y -= lidar_position[1] # y coordinate in base link frame
            if abs(y) < (self.width*self.resolution / 2) and abs(x) < (self.height*self.resolution): # obstacle is within the grid
                obstacle = [y//self.resolution - self.height//2, x//self.resolution]
                grid[int(obstacle[0]), int(obstacle[1])] = int(100)
                
                if int(obstacle[0]+1) < self.height:
                    if grid[int(obstacle[0]+1), int(obstacle[1])] < int(1):
                        grid[int(obstacle[0]+1), int(obstacle[1])] = int(50)
                if int(obstacle[1]+1) < self.width:
                    if  grid[int(obstacle[0]), int(obstacle[1]+1)] < int(1):
                        grid[int(obstacle[0]), int(obstacle[1]+1)] = int(50)
                if obstacle[0]-1 > 0:
                    if  grid[int(obstacle[0]-1), int(obstacle[1])] < int(1):
                        grid[int(obstacle[0]-1), int(obstacle[1])] = int(50)
                if obstacle[1]-1 > 0:
                    if  grid[int(obstacle[0]), int(obstacle[1]-1)] < int(1):
                        grid[int(obstacle[0]), int(obstacle[1]-1)] = int(50)

                t = 0.1; i = 1
                while t*i <= dist[d]: # create free cells
                    x = (dist[d]-t*i)*np.cos(angles[d]) # x coordinate in lidar frame
                    y = (dist[d]-t*i)*np.sin(angles[d]) # y coordinate in lidar frame
                    x -= lidar_position[0] # x coordinate in base link frame
                    y -= lidar_position[1] # y coordinate in base link frame
                    possible_free_cell = [y//self.resolution - self.height//2, x//self.resolution]
                    if grid[int(possible_free_cell[0]), int(possible_free_cell[1])] < int(1):
                        grid[int(possible_free_cell[0]), int(possible_free_cell[1])] = int(0)
                    i += 1
            else: # obstacle beyond the grid
                t = 0.1; i = 1
                while t*i <= dist[d]: # create free cells
                    x = min(dist[d]-t*i - lidar_position[0], self.resolution*self.width/2)*np.cos(angles[d]) # x coordinate in lidar frame
                    y = min(dist[d]-t*i - lidar_position[1], self.resolution*self.width/2)*np.sin(angles[d]) # y coordinate in lidar frame
                    possible_free_cell = [y//self.resolution - self.height//2, x//self.resolution]
                    if grid[int(possible_free_cell[0]), int(possible_free_cell[1])] < int(1):
                        grid[int(possible_free_cell[0]), int(possible_free_cell[1])] = int(0)
                    i += 1
        return
    
    def pf_callback(self, pose_msg):
        """ The pose callback when subscribed to particle filter's inferred pose, where the main RRT loop happens """
        curr_x = pose_msg.pose.pose.position.x # global frame
        curr_y = pose_msg.pose.pose.position.y # global frame
        self.curr_pos = [curr_x, curr_y] # global frame
        self.curr_orientation = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        heading = tf.transformations.euler_from_quaternion(self.curr_orientation)[2]

        if self.occ_grid is None:
            return
        
        mp = RRTMotionPlanner(self.occ_grid)
        plan = mp.plan()

        plan_marker = self.create_plan_marker(plan)
        self.plan_pub.publish(plan_marker)
        
        PP = PurePursuit(plan)
        drive_msg, marker = PP.pursue(curr_x, curr_y, heading)
        
        self.nearest_wp_pub.publish(marker)
        self.drive_pub.publish(drive_msg)
    
    def create_plan_marker(self, plan):
        """ Given the position of a sample, publishes the necessary Marker data to the 'wp_viz' topic for RViZ visualization """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.action = 0 # add the marker
        marker.type = 5 # line_list

        for p in range(len(plan)-1):
            pt1 = Point()
            pt1.x = plan[p][0]
            pt1.y = plan[p][1]
            marker.points.append(pt1)

            pt2 = Point()
            pt2.x = plan[p+1][0]
            pt2.y = plan[p+1][1]
            marker.points.append(pt2)

        marker.scale.x = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0    
        return marker

def main():
    rospy.init_node('rrt')
    rrt = RRT()
    rospy.spin()

if __name__ == '__main__':
    main()