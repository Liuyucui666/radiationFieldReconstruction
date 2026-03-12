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
        dt = (current_time - prev_time)*0.000001
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
def control_car_gui():
    # 控制小车运动的函数
    global car_control_params
    robot = Robot(left=Motor(forward=22, backward=27, enable=18), right=Motor(forward=25, backward=24, enable=23))

    def control_car():
        global car_control_params
        while car_control_params["continue_motion"]:
            # 获取当前的控制参数
            speed = car_control_params["speed"]
            direction = car_control_params["direction"]

            if direction == 0:  # 向前
                robot.forward(speed)
                sleep(1)
                robot.stop()
                update_params(0, 0)
            elif direction == 180:  # 向后
                robot.backward(speed)
                sleep(1)
                robot.stop()
                update_params(0, 0)
            elif direction == 90:  # 左转30°
                robot.left(speed)
                sleep(0.5)
                robot.stop()
                update_params(0, 0)
            elif direction == 270:  # 右转30°
                robot.right(speed)
                sleep(0.5)
                robot.stop()
                update_params(0, 0)
            else:
                robot.stop()

            print(f"Controlling car: Speed={speed}, Direction={direction}")

            time.sleep(1.5)
        print("Car stopped.")


    def update_params(speed, direction):
        car_control_params["speed"] = speed
        car_control_params["direction"] = direction

    def start_motion():
        global car_control_thread, imu_data_thread
        car_control_params["continue_motion"] = True
        car_control_thread = threading.Thread(target=control_car)
        car_control_thread.start()
        imu_data_thread = threading.Thread(target=collect_imu_data)
        imu_data_thread.start()

    def stop_motion():
        global car_control_thread, imu_data_thread
        car_control_params["continue_motion"] = False
        car_control_thread.join()
        imu_data_thread.join()
        root.destroy()

    # 创建主窗口
    root = tk.Tk()
    root.title("Car Control GUI")

    # 创建按钮
    btn_forward = tk.Button(root, text="Forward", command=lambda: update_params(0.3, 0))
    btn_backward = tk.Button(root, text="Backward", command=lambda: update_params(0.3, 180))
    btn_left = tk.Button(root, text="Turn Left 30°", command=lambda: update_params(0.3, 90))
    btn_right = tk.Button(root, text="Turn Right 30°", command=lambda: update_params(0.3, 270))
    btn_stop = tk.Button(root, text="Stop", command=stop_motion)

    # 布局按钮
    btn_forward.grid(row=1, column=1, padx=10, pady=10)
    btn_backward.grid(row=3, column=1, padx=10, pady=10)
    btn_left.grid(row=2, column=0, padx=10, pady=10)
    btn_right.grid(row=2, column=2, padx=10, pady=10)
    btn_stop.grid(row=2, column=1, padx=10, pady=10)

    # 启动线程
    start_motion()

    # 启动GUI主循环
    root.mainloop()


# 主函数
def cameraMove():
    global car_control_params, accel_bias, gyro_bias
    # 静止偏置
    bias_cal(2)

    # 启动GUI界面控制
    control_car_gui()

    # 返回转换矩阵
    return affine_matrix


if __name__ == "__main__":
    cameraMove()
