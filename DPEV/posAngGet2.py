# @Time : 2025/3/4 16:54
# @Author : LiuyuCui
# @Email : 22111140001@m.fudan.edu.cn
# @File : posAngGet1.py

import pyrealsense2 as rs
import numpy as np
import time

def posAngGet2():
    # 初始化 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)  # 加速度计，250Hz
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # 陀螺仪，200Hz
    pipeline.start(config)

    # 初始化变量
    prev_time = time.time()
    velocity = np.zeros(3)
    position = np.zeros(3)
    rotation = np.zeros(3)
    accel_bias = np.zeros(3)  # 加速度计偏置
    gyro_bias = np.zeros(3)  # 陀螺仪偏置
    accel_filter = np.zeros(3)  # 滑动平均滤波器
    filter_size = 16
    filter_buffer = np.zeros((filter_size, 3))

    # 静止时校准偏置
    print("校准偏置，请确保相机静止...")
    start_time = time.time()
    sample_count = 0

    while time.time() - start_time < 10:  # 校准持续10秒
        frames = pipeline.wait_for_frames(100000)
        sample_count += 1
        for f in frames:
            data = f.as_motion_frame().get_motion_data()
            if f.profile.stream_type() == rs.stream.accel:
                accel_bias += np.array([data.x, data.y, data.z])
            elif f.profile.stream_type() == rs.stream.gyro:
                gyro_bias += np.array([data.x, data.y, data.z])

    # 计算平均值
    accel_bias /= sample_count
    gyro_bias /= sample_count
    print("加速度计偏置校准完成:", accel_bias)
    print("陀螺仪偏置校准完成:", gyro_bias)

    # 初始重力向量（使用静止时的偏置值）
    gravity_in_sensor_frame = accel_bias

    try:
        last_output_time = time.time()
        prev_time = time.time()

        while True:
            frames = pipeline.wait_for_frames()
            current_time = time.time()

            dt = current_time - prev_time
            prev_time = current_time

            for f in frames:
                if f.is_motion_frame():
                    data = f.as_motion_frame().get_motion_data()
                    if f.profile.stream_type() == rs.stream.accel:

                        # 根据实时姿态调整重力向量
                        rotation_matrix = euler_to_rotation_matrix(rotation)  # 当前旋转矩阵
                        gravity_in_sensor_frame = rotation_matrix @ accel_bias # 重力在传感器坐标系中的分量
                        accel = np.array([data.x, data.y, data.z]) - gravity_in_sensor_frame  # 补偿重力分量
                        # 将绝对值小于0.05的加速度值视为0
                        accel[np.abs(accel) < 0.1] = 0

                        # 滑动平均滤波
                        filter_buffer = np.roll(filter_buffer, -1, axis=0)
                        filter_buffer[-1] = accel
                        accel_filter = np.mean(filter_buffer, axis=0)

                        # 更新位置和速度
                        position += velocity * dt + 0.5 * accel_filter * dt ** 2
                        velocity += accel_filter * dt

                    elif f.profile.stream_type() == rs.stream.gyro:
                        gyro = np.array([data.x, data.y, data.z]) - gyro_bias
                        rotation += gyro * dt  # 更新旋转角度

            if current_time - last_output_time >= 10:
                rotation_deg = np.degrees(rotation)
                print(f"位置: x={position[0]:.4f} m, y={position[1]:.4f} m, z={position[2]:.4f} m")
                print(f"速度: x={velocity[0]:.4f} m/s, y={velocity[1]:.4f} m/s, z={velocity[2]:.4f} m/s")
                print(f"旋转角度: x={rotation_deg[0]:.4f}°, y={rotation_deg[1]:.4f}°, z={rotation_deg[2]:.4f}°")
                print("--------------------------------------------------")
                last_output_time = current_time
                affine_matrix = np.vstack((np.hstack((rotation_matrix.T, position.reshape(-1, 1))), [0, 0, 0, 1]))
                break

    except KeyboardInterrupt:
        print("程序已终止")
    finally:
        pipeline.stop()

    return affine_matrix

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

if __name__ == '__main__':
    posAngGet2()
