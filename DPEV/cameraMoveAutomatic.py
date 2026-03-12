# @Time : 2025/4/9 9:57
# @Author : LiuyuCui
# @Email : 22111140001@m.fudan.edu.cn
# @File : cameraMove.py

import threading
import time
import tkinter as tk
from imuDataReceive import *
import numpy as np
from gpiozero import Robot, Motor
from time import sleep

# 全局变量
accel_bias = np.zeros(3)  # 加速度计偏置
gyro_bias = np.zeros(3)  # 陀螺仪偏置
affine_matrix = None

car_control_params = {
    "continue_motion": True,  # 是否继续运动
    "speed": 0,  # 运动速度
    "direction": 0  # 运动方向
}  # 控制小车运动的参数

def bias_cal(calitime):
    global accel_bias, gyro_bias
    print("校准偏置，请确保相机静止...")
    start_time = time.time()
    sample_count = 0

    while time.time() - start_time < calitime:  # 校准持续10秒
        IMUdata = imuDataReceive()
        accel_bias += np.array([IMUdata[3], IMUdata[4], IMUdata[5]])
        gyro_bias += np.array([IMUdata[0], IMUdata[1], IMUdata[2]])
        sample_count += 1

    # 计算平均值
    accel_bias /= sample_count
    gyro_bias /= sample_count
    print("加速度计偏置校准完成:", accel_bias)
    print("陀螺仪偏置校准完成:", gyro_bias)

# 统计IMU数据的函数
def collect_imu_data():
    global car_control_params, accel_bias, gyro_bias, affine_matrix
    rotation = np.zeros(3)
    filter_size = 16
    filter_buffer = np.zeros((filter_size, 3))
    velocity = np.zeros(3)
    position = np.zeros(3)

    def euler_to_rotation_matrix(euler_angles):
        """
        将欧拉角（ZYX顺序）转换为旋转矩阵
        :param euler_angles: 一个包含三个欧拉角的列表或元组，顺序为 [yaw, pitch, roll]
        :return: 对应的旋转矩阵
        """
        yaw, pitch, roll = euler_angles

        # 计算每个角度的正余弦值
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cr = np.cos(roll)
        sr = np.sin(roll)

        # 构造旋转矩阵
        R_z = np.array([[cy, sy, 0],
                        [-sy, cy, 0],
                        [0, 0, 1]])

        R_y = np.array([[cp, 0, -sp],
                        [0, 1, 0],
                        [sp, 0, cp]])

        R_x = np.array([[1, 0, 0],
                        [0, cr, sr],
                        [0, -sr, cr]])

        # 组合旋转矩阵
        rotation_matrix = np.dot(R_x, np.dot(R_y, R_z))

        return rotation_matrix

    imu_data_initial = imuDataReceive()
    prev_time = imu_data_initial[6]
    while car_control_params["continue_motion"]:
        # 获取IMU数据
        imu_data = imuDataReceive()

        # 将IMU数据处理得到转换数据
        bias_rotation_matrix = euler_to_rotation_matrix(rotation)
        accel_bias_rotation = bias_rotation_matrix @ accel_bias  # 重力在传感器坐标系中的分量
        accel = np.array([imu_data[3], imu_data[4], imu_data[5]]) - accel_bias_rotation
        # 将绝对值小于0.05的加速度值视为0
        accel[np.abs(accel) < 0.1] = 0

        filter_buffer = np.roll(filter_buffer, -1, axis=0)
        filter_buffer[-1] = accel
        accel_filter = np.mean(filter_buffer, axis=0)

        # 更新位置和速度
        current_time = imu_data[6]
        dt = current_time - prev_time
        prev_time = current_time
        position += velocity * dt + 0.5 * accel_filter * dt ** 2
        velocity += accel_filter * dt

        gyro = np.array([imu_data[0], imu_data[1], imu_data[2]]) - gyro_bias
        rotation += gyro * dt  # 更新旋转角度

    rotation_deg = np.degrees(rotation)
    print("IMU data collection stopped.")
    # 打印累积的IMU数据
    print(f"位置: x={position[0]:.4f} m, y={position[1]:.4f} m, z={position[2]:.4f} m")
    print(f"速度: x={velocity[0]:.4f} m/s, y={velocity[1]:.4f} m/s, z={velocity[2]:.4f} m/s")
    print(f"旋转角度: x={rotation_deg[0]:.4f}°, y={rotation_deg[1]:.4f}°, z={rotation_deg[2]:.4f}°")
    print("--------------------------------------------------")
    affine_matrix = np.vstack((np.hstack((bias_rotation_matrix.T, position.reshape(-1, 1))), [0, 0, 0, 1]))


# GUI界面控制函数
def control_car_gui(step):
    global car_control_params
    robot = Robot(left=Motor(forward=22, backward=27, enable=18), right=Motor(forward=25, backward=24, enable=23))
    
    imu_data_thread = threading.Thread(target=collect_imu_data)
    imu_data_thread.start()

    if step == 1:
        robot.forward(0.3)
        sleep(1)
    elif step == 2:
        robot.left(0.3)
        sleep(1)
    elif step == 3:
        robot.forward(0.3)
        sleep(1)
    else:
        robot.stop()
    car_control_params["continue_motion"] = False
    
    imu_data_thread.join()
        



# 主函数
def cameraMoveAutomatic(step):
    global car_control_params, accel_bias, gyro_bias
    # 静止偏置
    bias_cal(3)

    # 启动GUI界面控制
    control_car_gui(step)

    # 返回转换矩阵
    return affine_matrix


if __name__ == "__main__":
    cameraMoveAutomatic()
