#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

# Tunable parameters
MAX_SPEED = 3.2 # Optimal value: 3.2

# Modes
SLOW_MODE = False # Use this mode to drive slowly at constant speed, for testing or mapping

class ReactiveFollowGap():
    """ The class that handles follow-the-gap """
    def __init__(self):
        self.threshold = 5. #check this
        self.min_gap = 3
        self.safety_radius = 0.8 #[m] #what is the car width?
        self.min_distance = 1
        self.gap_check_distance = 1.

        self.angle_min = None
        self.angle_max = None
        self.angle_increment = None
        self.angles = None
        
        lidarscan_topic = '/scan'
        self.scan_sub_ = rospy.Subscriber(lidarscan_topic, LaserScan, self.lidar_callback, queue_size=10)

        drive_topic = '/vesc/ackermann_cmd_mux/input/navigation'
        self.drive_pub_ = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size=10)

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        min_range = min(ranges)
        min_index = ranges.index(min_range)
        proc_ranges = np.array(ranges)
        proc_ranges[proc_ranges > self.threshold] = self.threshold
        for item in proc_ranges:
            if item == 'nan' or item == np.inf:
                item = self.threshold

        delta_theta = abs(np.arctan2(self.safety_radius/2, min_range))

        current_angle_sum = 0
        current_index = min_index
        while (current_angle_sum < delta_theta) and (current_index < len(self.angles)):
            proc_ranges[current_index] = 0
            current_index += 1
            current_angle_sum += self.angle_increment
        current_angle_sum = 0
        current_index = min_index
        while (current_angle_sum > delta_theta) and (current_index >= 0):
            proc_ranges[current_index] = 0
            current_index -= 1
            current_angle_sum -= self.angle_increment

        return proc_ranges

    def valid_range(self, proc_ranges, index):
        return (proc_ranges[index] > self.min_distance)

    def bigger_ranges(self, ranges1, ranges2):
        return ranges1 if (ranges1[1]-ranges1[0]) > (ranges2[1]-ranges2[0]) else ranges2

    def find_max_gap(self, proc_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges"""
        current_index = 0
        current_range = [0,0]
        max_range = [0,0]

        for current_index in range(len(proc_ranges)):
            if self.valid_range(proc_ranges, current_index):
                current_range[1] = current_index
            else:
                if current_range == self.bigger_ranges(max_range, current_range):
                    max_range = current_range
                current_range = [current_index,current_index]
                        
        length = max_range[1] - max_range[0]   
        if length > self.min_gap:
            return max_range
        return
    
    def find_best_point(self, start_i, end_i, ranges, method='furthest'):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        if method == 'furthest':
            # Find furthest point in max_gap -- closest to centre of max_gap

            #Initialize
            index_furthest_point = int((start_i+end_i)/2)
            current_index = int((start_i+end_i)/2)
            current_furthest_point = ranges[current_index]

            #First, increasing from middle
            while current_index <= end_i:
                if ranges[current_index] > current_furthest_point:
                    current_furthest_point = ranges[current_index]
                    index_furthest_point = current_index
                current_index += 1

            #Next Decreasing from the middle
            while current_index >= start_i:
                if ranges[current_index] > current_furthest_point:
                    current_furthest_point = ranges[current_index]
                    index_furthest_point = current_index
                current_index -= 1

            return index_furthest_point
        
        elif method == 'middle':
            middle_index = int((start_i+end_i)/2)
            return middle_index

    def get_velocity(self, steering_angle):
        """ Given the desired steering angle, returns the appropriate velocity to publish to the car """
        if SLOW_MODE:
            return 0.8 # [m/s], roughly the minimum speed that the car drives smoothly at
        velocity = max(MAX_SPEED - abs(np.rad2deg(steering_angle))/50, 0.8) # [m/s], velocity varies smoothly with steering angle
        print('Velocity: ' + str(velocity))
        return velocity

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message """
        if self.angle_min is None:
            self.angle_min = data.angle_min

        if self.angle_max is None:
            self.angle_max = data.angle_max

        if self.angle_increment is None:
            self.angle_increment = data.angle_increment

        if self.angles is None:
            self.angles = np.linspace(self.angle_min, self.angle_max, num = len(data.ranges))

        ranges = list(data.ranges)
        ranges.reverse()
        processed_ranges = self.preprocess_lidar(ranges)

        max_gap_indices = self.find_max_gap(processed_ranges)
        steering_angle = 0.

        if max_gap_indices is not None:
            best_idx = self.find_best_point(max_gap_indices[0], max_gap_indices[-1], processed_ranges, method='middle')
            steering_angle = -self.angles[best_idx]
        else:
            print("Getting NONE for max_gap")

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steering_angle)
        drive_msg.drive.speed = self.get_velocity(steering_angle)
        self.drive_pub_.publish(drive_msg)

    def cornering_left(self, ranges):
        #check if there is a drastic drop between range values at angles around -90
        angle_closer_left_idx = np.argmax(self.angles < np.deg2rad(-96))
        angle_further_left_idx = np.argmin(self.angles > np.deg2rad(-85))

        if ranges[angle_further_left_idx] - ranges[angle_closer_left_idx] > self.gap_check_distance:
            print("TURNING LEFT")
            return True
        else:
            return False

    def cornering_right(self, ranges):
        #check if there is a drastic drop between range values at angles around -90
        angle_closer_right_idx = np.argmin(self.angles < np.deg2rad(85))
        angle_further_right_idx = np.argmax(self.angles > np.deg2rad(96))

        if ranges[angle_further_right_idx] - ranges[angle_closer_right_idx] > self.gap_check_distance:
            print("TURNING RIGHT")
            return True
        else:
            return False

def main():
    rospy.init_node("reactive_node", anonymous=True)
    reactive_node = ReactiveFollowGap()
    rospy.spin()

if __name__ == '__main__':
    print("Follow-The-Gap Mode Initialized...")
    main()