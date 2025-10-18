#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from math import pi as PI
from typing import Tuple, List
import threading
import time

from geometry_msgs.msg import TwistStamped, Twist
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import UInt8MultiArray
from std_msgs.msg import Bool

import os
import sys
script_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_path)
from ros_interface import DataInterface

from generated import public_api_down_pb2
from generated import public_api_types_pb2
from generated import public_api_up_pb2

class ChassisInterface:
    def __init__(self) -> None:
        self.data_interface = DataInterface(node_name = "chassis_trans")
        self.__lock = threading.Lock()

        self.__api_initialized = False
        self.__is_set_vehicle_origin_position = False
        self.__vehicle_origin_position = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        self.__report_frequency_map = {
            50: public_api_types_pb2.ReportFrequency.Rf50Hz,
            100: public_api_types_pb2.ReportFrequency.Rf100Hz,
            250: public_api_types_pb2.ReportFrequency.Rf250Hz,
            500: public_api_types_pb2.ReportFrequency.Rf500Hz,
            1000: public_api_types_pb2.ReportFrequency.Rf1000Hz
        }

        self.__parking_stop_detail = public_api_types_pb2.ParkingStopDetail()
        self.__last_warning_time = time.perf_counter()

        self.__session_id = None
        self.__session_holder = None

        # Timeout monitoring
        self.__last_cmd_time = None  # Last control command timestamp
        self.__timeout_threshold = 0.3  # 300ms timeout threshold

        self.data_interface.set_parameter('frame_id', "base_link")
        self.data_interface.set_parameter('simple_mode', True)
        self.data_interface.set_parameter('report_freq', 100)
        self.__frame_id = self.data_interface.get_parameter('frame_id')
        self.__simple_mode = self.data_interface.get_parameter('simple_mode')
        self.__report_frequency = self.data_interface.get_parameter('report_freq')

        self.__ws_down_pub = self.data_interface.create_publisher(UInt8MultiArray, "ws_down")
        self.__motor_states_pub = self.data_interface.create_publisher(JointState, "motor_states")
        self.__real_vel_pub = self.data_interface.create_publisher(TwistStamped, "real_vel")
        self.__odom_pub = self.data_interface.create_publisher(Odometry, "odom")

        self.data_interface.create_subscriber(UInt8MultiArray, "ws_up", self.__ws_up_callback)
        if self.__simple_mode:
            self.data_interface.logi("simple mode enabled, you can only control the chassis with /cmd_vel")
            self.data_interface.create_subscriber(Twist, "cmd_vel", self.__cmd_vel_callback)
        else:
            self.data_interface.logi("simple mode disabled, you can only control the chassis with /joint_ctrl")
            self.data_interface.create_subscriber(JointState, "joint_ctrl", self.__joint_ctrl_callback)
        self.data_interface.create_subscriber(Bool, "clear_err", self.__clear_err_callback)
        self.data_interface.create_subscriber(Bool, "shutdown_request", self.__shutdown_request_callback)

        self.__timeout_timer = self.data_interface.create_timer(0.1, self.__timeout_check_callback)
        self.__session_timer = self.data_interface.create_timer(3.0, self.__session_check_callback)

    @property
    def report_freq(self):
        return self.__report_frequency
    
    def get_report_freq(self, key: int):
        if key not in self.__report_frequency_map:
            self.data_interface.logw(f"report frequency only can be set to 50Hz ,100Hz, 250Hz, 500Hz, 1000Hz, but got {key}")
            self.data_interface.logw(f"report frequency set to 100Hz by default")
            self.__report_frequency = 100
        return self.__report_frequency_map.get(key, public_api_types_pb2.ReportFrequency.Rf100Hz)

    # pub
    def pub_ws_down(self, data: List[int]):
        try:
            msg = UInt8MultiArray()
            msg.data = data
            self.__ws_down_pub.publish(msg)
        except Exception:
            pass

    def pub_motor_states(self, pos: List[float], vel: List[float], eff: List[float]):
        try:
            length = len(pos)
            msg = JointState()
            msg.header.stamp = self.data_interface.get_timestamp()
            msg.name = [f"joint{i}" for i in range(length)]
            msg.position = pos
            msg.velocity = vel
            msg.effort = eff
            self.__motor_states_pub.publish(msg)
        except Exception:
            pass

    def pub_real_vel(self, x: float, y: float, yaw: float):
        try:
            msg = TwistStamped()
            msg.header.stamp = self.data_interface.get_timestamp()
            msg.header.frame_id = self.__frame_id
            msg.twist.linear.x = x
            msg.twist.linear.y = y
            msg.twist.angular.z = yaw
            self.__real_vel_pub.publish(msg)
        except Exception:
            pass

    def pub_odom(self, pos_x: float, pos_y: float, pos_yaw: float, linear_x: float, linear_y: float, angular_z: float):
        try:
            out = Odometry()
            out.header.stamp = self.data_interface.get_timestamp()
            out.header.frame_id = "odom"
            out.child_frame_id = self.__frame_id
            out.pose.pose.position.x = pos_x
            out.pose.pose.position.y = pos_y
            out.pose.pose.position.z = 0.0
            qz = np.sin(pos_yaw / 2.0)
            qw = np.cos(pos_yaw / 2.0)
            out.pose.pose.orientation.x = 0.0
            out.pose.pose.orientation.y = 0.0
            out.pose.pose.orientation.z = qz
            out.pose.pose.orientation.w = qw
            out.twist.twist.linear.x = linear_x
            out.twist.twist.linear.y = linear_y
            out.twist.twist.angular.z = angular_z
            self.__odom_pub.publish(out)
        except Exception:
            pass

    # sub
    def __ws_up_callback(self, msg: UInt8MultiArray):
        api_up = public_api_up_pb2.APIUp()
        api_up.ParseFromString(bytes(msg.data))
        with self.__lock:
            self.__api_initialized = api_up.base_status.api_control_initialized
            if api_up.base_status.HasField("parking_stop_detail"):
                self.__parking_stop_detail = api_up.base_status.parking_stop_detail
            else:
                self.__parking_stop_detail = public_api_types_pb2.ParkingStopDetail()
            self.__session_id = api_up.session_id
            self.__session_holder = api_up.base_status.session_holder
        self.__check_parking_stop_detail()
        # parse data
        pp, vv, tt = self.prase_wheel_data(api_up)
        (spd_x, spd_y, spd_z), (pos_x, pos_y, pos_z) = self.parse_vehicle_data(api_up)
        acc, angular_velocity, quaternion = self.parse_imu_data(api_up)
        with self.__lock:
            if not self.__is_set_vehicle_origin_position:
                self.reset_vehicle_position(pos_x, pos_y, pos_z)
                self.__is_set_vehicle_origin_position = True
        (relative_x, relative_y, relative_yaw) = self.get_vehicle_position(pos_x, pos_y, pos_z)
        # publish data
        self.pub_motor_states(pp, vv, tt)
        self.pub_real_vel(spd_x, spd_y, spd_z)
        self.pub_odom(relative_x, relative_y, relative_yaw, spd_x, spd_y, spd_z)

    def __joint_ctrl_callback(self, msg: JointState):
        # Update last command time
        with self.__lock:
            self.__last_cmd_time = time.time()
            api_initialized = self.__api_initialized
        if not api_initialized:
            api_init = public_api_down_pb2.APIDown()
            api_init.base_command.api_control_initialize = True
            bin = api_init.SerializeToString()
            self.pub_ws_down(bin)
        length = len(msg.name)
        if length != len(msg.position) or length != len(msg.velocity) or length != len(msg.effort):
            self.data_interface.logw("JointState message format error.")
            return
        api_down = public_api_down_pb2.APIDown()
        for i in range(length):
            joint_cmd = api_down.base_command.motor_targets.targets.add()
            # joint_cmd.position = msg.position[i]
            joint_cmd.speed = msg.velocity[i]
            # joint_cmd.torque = msg.effort[i]
        bin = api_down.SerializeToString()
        self.pub_ws_down(list(bin))

    def __cmd_vel_callback(self, msg: Twist):
        # Update last command time
        with self.__lock:
            self.__last_cmd_time = time.time()
            api_initialized = self.__api_initialized
        if not api_initialized:
            api_init = public_api_down_pb2.APIDown(
                base_command = public_api_types_pb2.BaseCommand(api_control_initialize = True)
            )
            bin = api_init.SerializeToString()
            self.pub_ws_down(bin)
        api_down = public_api_down_pb2.APIDown(
            base_command = public_api_types_pb2.BaseCommand(
                simple_move_command = public_api_types_pb2.SimpleBaseMoveCommand(
                    xyz_speed = public_api_types_pb2.XyzSpeed(
                        speed_x = msg.linear.x,
                        speed_y = msg.linear.y,
                        speed_z = msg.angular.z
                    )
                )
            )
        )
        bin = api_down.SerializeToString()
        self.pub_ws_down(bin)
    
    def __clear_err_callback(self, msg: Bool):
        if msg.data:
            msg = public_api_down_pb2.APIDown()
            msg.base_command.clear_parking_stop = True
            bin = msg.SerializeToString()
            self.pub_ws_down(bin)

    def __shutdown_request_callback(self, msg: Bool):
        """Handle graceful shutdown request from upper layer"""
        if msg.data:
            self.data_interface.logi("Received shutdown request, stopping vehicle...")

            # First, send zero velocity command to stop the vehicle
            stop_cmd = public_api_down_pb2.APIDown(
                base_command = public_api_types_pb2.BaseCommand(
                    simple_move_command = public_api_types_pb2.SimpleBaseMoveCommand(
                        xyz_speed = public_api_types_pb2.XyzSpeed(
                            speed_x = 0.0,
                            speed_y = 0.0,
                            speed_z = 0.0
                        )
                    )
                )
            )
            # Send stop command multiple times to ensure it's received
            for _ in range(5):
                self.pub_ws_down(list(stop_cmd.SerializeToString()))
                time.sleep(0.02)  # 20ms interval

            # Then send disable command to hardware
            api_down = public_api_down_pb2.APIDown(
                base_command = public_api_types_pb2.BaseCommand(
                    api_control_initialize = False
                )
            )
            bin = api_down.SerializeToString()
            self.pub_ws_down(list(bin))

            # Reset internal state to allow reconnection
            with self.__lock:
                self.__last_cmd_time = None
                self.__api_initialized = False
                self.__is_set_vehicle_origin_position = False  # Reset origin on reconnection

            self.data_interface.logi("Vehicle stopped and control disabled, odometry origin will reset on reconnection")
            # Note: Node remains running and can accept new connections

    def __timeout_check_callback(self):
        need_disable = False

        with self.__lock:
            if self.__last_cmd_time is not None and self.__api_initialized:
                elapsed = time.time() - self.__last_cmd_time
                if elapsed > self.__timeout_threshold:
                    need_disable = True

        # Send disable command (but keep node running)
        if need_disable:
            self.data_interface.logw("Command timeout detected, stopping vehicle...")

            # First, send zero velocity command to stop the vehicle
            stop_cmd = public_api_down_pb2.APIDown(
                base_command = public_api_types_pb2.BaseCommand(
                    simple_move_command = public_api_types_pb2.SimpleBaseMoveCommand(
                        xyz_speed = public_api_types_pb2.XyzSpeed(
                            speed_x = 0.0,
                            speed_y = 0.0,
                            speed_z = 0.0
                        )
                    )
                )
            )
            # Send stop command multiple times to ensure it's received
            for _ in range(5):
                self.pub_ws_down(list(stop_cmd.SerializeToString()))
                time.sleep(0.02)  # 20ms interval

            # Then send disable command
            api_down = public_api_down_pb2.APIDown(
                base_command = public_api_types_pb2.BaseCommand(
                    api_control_initialize = False
                )
            )
            bin = api_down.SerializeToString()
            self.pub_ws_down(list(bin))

            # Reset state to allow reconnection
            with self.__lock:
                self.__last_cmd_time = None
                self.__api_initialized = False
                self.__is_set_vehicle_origin_position = False  # Reset origin on reconnection

            self.data_interface.logi("Vehicle stopped and control disabled due to timeout, odometry origin will reset on reconnection")

    def __check_parking_stop_detail(self):
        start_time = time.perf_counter()
        with self.__lock:
            parking_stop_detail = self.__parking_stop_detail
        if parking_stop_detail != public_api_types_pb2.ParkingStopDetail():
            if start_time - self.__last_warning_time > 1.0:
                self.data_interface.loge(f"emergency stop: {parking_stop_detail}")
                self.__last_warning_time = start_time

    def __session_check_callback(self):
        with self.__lock:
            session_id = self.__session_id
            session_holder = self.__session_holder
        if session_id != session_holder:
            self.data_interface.logw(f"You can not control this vehicle, because it is controlled by user : {session_holder}, your user id : {session_id}")

    def prase_wheel_data(self, api_up: public_api_up_pb2.APIUp) -> Tuple[list, list, list]:
        vv = []
        tt = []
        pp = []
        if api_up.base_status.IsInitialized():
            for motor_states in api_up.base_status.motor_status:
                torque = motor_states.torque
                speed = motor_states.speed
                position = (motor_states.position % motor_states.pulse_per_rotation) / motor_states.pulse_per_rotation * (2.0 * PI) - PI # radian
                tt.append(torque)
                vv.append(speed)
                pp.append(position)
        return pp, vv, tt
    
    def parse_vehicle_data(self, api_up: public_api_up_pb2.APIUp) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        (spd_x, spd_y, spd_z) = (0.0, 0.0, 0.0)
        (pos_x, pos_y, pos_z) = (0.0, 0.0, 0.0)
        if api_up.base_status.IsInitialized():
            spd_x = api_up.base_status.estimated_odometry.speed_x
            spd_y = api_up.base_status.estimated_odometry.speed_y
            spd_z = api_up.base_status.estimated_odometry.speed_z
            pos_x = api_up.base_status.estimated_odometry.pos_x
            pos_y = api_up.base_status.estimated_odometry.pos_y
            pos_z = api_up.base_status.estimated_odometry.pos_z
        return (spd_x, spd_y, spd_z), (pos_x, pos_y, pos_z)
    
    def parse_imu_data(self, api_up: public_api_up_pb2.APIUp) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float, float]]:
        acc = (0.0, 0.0, 0.0)
        angular_velocity = (0.0, 0.0, 0.0)
        quaternion = (0.0, 0.0, 0.0, 0.0)
        if api_up.imu_data.IsInitialized():
            acc = (api_up.imu_data.acceleration.ax, api_up.imu_data.acceleration.ay, api_up.imu_data.acceleration.az)
            angular_velocity = (api_up.imu_data.angular_velocity.wx, api_up.imu_data.angular_velocity.wy, api_up.imu_data.angular_velocity.wz)
            quaternion = (api_up.imu_data.quaternion.qx, api_up.imu_data.quaternion.qy, api_up.imu_data.quaternion.qz, api_up.imu_data.quaternion.qw)
        return acc, angular_velocity, quaternion
    
    def get_vehicle_position(self, x, y, yaw) -> Tuple[float, float, float]:
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        current_matrix = np.array([
            [cos_yaw, -sin_yaw, x],
            [sin_yaw,  cos_yaw, y],
            [0.0,      0.0,     1.0]
        ])
        # Calculate relative transformation: current * inverse(origin)
        with self.__lock:
            origin_inv = np.linalg.inv(self.__vehicle_origin_position)
        relative_matrix = origin_inv @ current_matrix
        # Extract position and orientation from relative matrix
        relative_x = relative_matrix[0, 2]
        relative_y = relative_matrix[1, 2]
        relative_yaw = np.arctan2(relative_matrix[1, 0], relative_matrix[0, 0])

        return (relative_x, relative_y, relative_yaw)
    
    def reset_vehicle_position(self, x, y, yaw):
        # Convert (x, y, yaw) to 2D transformation matrix
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        self.__vehicle_origin_position = np.array([
            [cos_yaw, -sin_yaw, x],
            [sin_yaw,  cos_yaw, y],
            [0.0,      0.0,     1.0]
        ])
    
def main():
    chassis = ChassisInterface()
    # set report frequency
    set_report_frequency_msg = public_api_down_pb2.APIDown()
    set_report_frequency_msg.set_report_frequency = chassis.get_report_freq(chassis.report_freq)
    bin = set_report_frequency_msg.SerializeToString()
    chassis.pub_ws_down(bin)
    chassis.data_interface.logi(f"Set report frequency to {chassis.report_freq}Hz.")
    try:
        while chassis.data_interface.ok():
            chassis.data_interface.sleep()
    except KeyboardInterrupt:
        chassis.data_interface.logi("Received Ctrl-C.")
    finally:
        chassis.data_interface.shutdown()

if __name__ == '__main__':
    main()