# @Time : 2025/3/5 17:12
# @Author : LiuyuCui
# @Email : 22111140001@m.fudan.edu.cn
# @File : clinkPointCalibration.py

import pyrealsense2 as rs
import numpy as np
import cv2

# 初始化 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 深度流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 彩色流
pipeline.start(config)

# 初始化鼠标点击位置和标志变量
click_position = None
output_result = False

# 新加入的内容：用于保存相机坐标和真实世界坐标的数组
camera_points = []  # 保存相机坐标
world_points = []   # 保存真实世界坐标

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global click_position, output_result
    if event == cv2.EVENT_LBUTTONDOWN:
        click_position = (x, y)
        print(f"Clicked at pixel: {click_position}")
        output_result = True  # 设置标志变量为 True，表示需要输出结果


def clinkPointCalibration():
    global click_position, output_result  # 声明为全局变量

    # 创建窗口并绑定鼠标回调
    cv2.namedWindow("RealSense")
    cv2.setMouseCallback("RealSense", mouse_callback)

    try:
        while True:
            # 获取帧
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 将帧转换为 NumPy 数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 如果有鼠标点击事件且需要输出结果
            if click_position and output_result:
                x, y = click_position
                # 检查点击的像素点是否在深度图像范围内
                if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
                    # 获取深度值
                    depth = depth_frame.get_distance(x, y)
                    if depth > 0:  # 检查深度值是否有效
                        # 将像素点转换为相机坐标系中的三维坐标
                        intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
                        depth_point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
                        print(f"3D point: {depth_point}")

                        # 新加入的内容：保存相机坐标
                        camera_points.append(depth_point)

                        # 提示用户输入真实世界坐标
                        user_input = input("Enter world coordinates (X Y Z): ")
                        world_x, world_y, world_z = map(float, user_input.split(','))
                        world_points.append([world_x, world_y, world_z])

                        # 检查是否已经收集了四组坐标
                        if len(camera_points) == 4:
                            # 使用OpenCV计算转换矩阵
                            camera_points_array = np.array(camera_points, dtype=np.float32)
                            world_points_array = np.array(world_points, dtype=np.float32)
                            # 使用 cv2.estimateAffine3D 计算刚体变换
                            ret, affine_matrix, inliers = cv2.estimateAffine3D(camera_points_array, world_points_array)

                            if ret:
                                print("采集结束\n")
                                # print("Affine Transformation Matrix:\n", affine_matrix)
                            else:
                                print("Failed to compute the transformation matrix.")
                            break  # 结束循环

                    else:
                        print("No depth data at this pixel.")
                else:
                    print("Clicked pixel is out of depth image range.")
                output_result = False  # 重置标志变量，避免重复输出

            # 显示图像
            cv2.imshow("RealSense", color_image)

            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    affine_matrix_homogeneous = np.vstack((affine_matrix, [0, 0, 0, 1]))
    print("Affine Transformation Matrix:\n", affine_matrix_homogeneous)
    return(affine_matrix_homogeneous)

if __name__ == '__main__':
    clinkPointCalibration()

