import cv2
import numpy as np
import pyrealsense2 as rs

def get_aligned_images():
    pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
    config = rs.config()  # 定义配置config
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流
    pipe_profile = pipeline.start(config)  # streaming流开始
    align = rs.align(rs.stream.color)

    frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐
 
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧
 
    #### 获取相机参数 ####
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
 
    img_color = np.asanyarray(aligned_color_frame.get_data())  # RGB图
    img_depth = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
 
    depth_colormap = cv2.applyColorMap \
        (cv2.convertScaleAbs(img_depth, alpha=0.008)
         , cv2.COLORMAP_JET)
 
    return color_intrin, depth_intrin, img_color, depth_colormap, aligned_depth_frame
 

def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    return dis, camera_coordinate



 
