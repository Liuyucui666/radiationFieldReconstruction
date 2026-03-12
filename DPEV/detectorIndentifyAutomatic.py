# @Time : 2025/3/6 15:07
# @Author : LiuyuCui
# @Email : 22111140001@m.fudan.edu.cn
# @File : detectorIndentify.py
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from camConfig import *
import time
from clinkPointCalibration import *
from cameraMoveAutomatic import *


def calculate_rotation_angle(top_left, top_right, bottom_left):
    horizontal_vector = np.array(top_right) - np.array(top_left)
    vertical_vector = np.array(bottom_left) - np.array(top_left)
    angle = np.arctan2(vertical_vector[1], horizontal_vector[0])
    return np.degrees(angle)

# 相机内参矩阵
def get_camera_matrix(color_intrin):
    fx, fy = color_intrin.fx, color_intrin.fy
    cx, cy = color_intrin.ppx, color_intrin.ppy
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)
    
    
# 检测卡片标记点
def detect_markers(flag, roi, img_color, aligned_depth_frame, x1, y1):
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    # 定义特征色的HSV阈值范围
    color_ranges = {
        "white": (np.array([0, 0, 180]), np.array([180, 50, 255])),
        "green": (np.array([25, 20, 20]), np.array([85, 255, 255])),
        "yellow": (np.array([10, 80, 80]), np.array([40, 255, 255])),
        "black": (np.array([0, 0, 0]), np.array([180, 150, 150]))
    }

    # 同时创建所有颜色的掩码，并执行开运算
    masks = {
        color: cv2.morphologyEx(
            cv2.inRange(roi_hsv, lower, upper),
            cv2.MORPH_OPEN,
            None  # 使用默认结构元素
        ) for color, (lower, upper) in color_ranges.items()
    }

    # 找到掩码中的轮廓
    def find_centroids(mask, color):
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pointlocation = []
        roi_area = mask.shape[0] * mask.shape[1]  # 计算ROI的面积
        roi_area_threshold = roi_area / 50  # 计算ROI面积的1/20作为阈
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) > 6:
                moments = cv2.moments(approx)
                area = cv2.contourArea(approx)  # 计算轮廓的面积
                if moments['m00'] != 0 and area > roi_area_threshold:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                    dis = aligned_depth_frame.get_distance(x1 + cx, y1 + cy)
                    pointlocation.append([x1 + cx, y1 + cy])
                    cv2.circle(img_color, (x1 + cx, y1 + cy), 5, color, -1)
        return pointlocation

    yellow_point = find_centroids(masks['yellow'], (0, 255, 0))
    black_point = find_centroids(masks['black'], (255, 255, 255))
    white_point = find_centroids(masks['white'], (0, 0, 0))
    green_point = find_centroids(masks['green'], (255, 255, 0))
    return yellow_point, black_point, white_point, green_point

# def point_calibration(feature_point, cornerpoint0, cornerpoint1, cornerpoint2):
#     top_left = feature_point
#
#     return [top_left, top_right, down_left, down_right]

def process_card(card_color, detection_model, color_intrin, depth_intrin, img_color, aligned_depth_frame, detected_info):
    result = get_sliced_prediction(
        img_color,
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    camera_matrix = get_camera_matrix(color_intrin)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    for index, obj in enumerate(result.object_prediction_list):
        x1_demo, y1_demo, x2_demo, y2_demo = map(int, obj.bbox.to_voc_bbox())
        x1 = max(int(x1_demo - (x2_demo - x1_demo) / 4), 0) # 扩大识别框
        x2 = int(x2_demo + (x2_demo - x1_demo) / 4)
        y1 = max(int(y1_demo - (y2_demo - y1_demo) / 4), 0)
        y2 = int(y2_demo + (y2_demo - y1_demo) / 4)
        x1, y1, x2, y2 = map(int, obj.bbox.to_voc_bbox())
        ux, uy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # 计算检测框距离与位置
        dis = aligned_depth_frame.get_distance(ux, uy)
        camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)
        camera_xyz = np.round(np.array(camera_xyz), 3) * 100  # 转为厘米

        # 根据四个点计算空间姿态
        roi = img_color[y1:y2, x1:x2]  # 感兴趣区域（扩大检测框内）
        top_left, top_right, down_left, down_right = detect_markers(card_color, roi, img_color, aligned_depth_frame, x1, y1)  # 检测点的三维坐标并在图上绘制
        object_points = np.array([[0, 0, 0],
                                   [1, 0, 0],
                                   [0, -1, 0],
                                   [1, -1, 0]], dtype=np.float32)
        # 检测是否识别出四个点
        if len(top_left) >= 1 and len(top_right) >= 1 and len(down_left) >= 1 and len(down_right) >= 1:
            image_points = np.array([top_left[0], top_right[0], down_left[0], down_right[0]], dtype=np.float32)
            point_flag = 'True'
        else:
            image_points = np.array([[0, 0],
                                   [1, 0],
                                   [0, -1],
                                   [1, -1]], dtype=np.float32)
            point_flag = 'False'

        # 根据四个点计算旋转矩阵
        _, rotation_vector, translation_vector = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        detected_info.append({
            'index': index + 1,
            'color': card_color,
            'position': camera_xyz,
            'distance': dis * 100,
            # 'rotation_angle': rotation_angle,
            'rotation_matrix': rotation_matrix,
            'translation_vector': translation_vector.flatten(),
            'point_flag': point_flag
        })

        color = (0, 255, 0) if card_color == 'blue' else (0, 0, 255)
        cv2.rectangle(img_color, (x1, y1), (x2, y2), color, 2)

        # 在检测框上标注 index
        text = f"ID: {index + 1}"  # 标注文本
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
        font_scale = 0.5  # 字体大小
        font_color = (255, 255, 255)  # 字体颜色（白色）
        font_thickness = 2  # 字体粗细
        text_x = x1
        text_y = y1 - 10  # 留出一些空间
        cv2.putText(img_color, text, (text_x, text_y), font, font_scale, font_color, font_thickness)


def main(blue_weights, red_weights):
    step = 1
    try:
        blue_detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=blue_weights,
            confidence_threshold=0.1,
            device='cuda'
        )
        red_detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=red_weights,
            confidence_threshold=0.1,
            device='cuda'
        )

        # 返回标定板到相机转换矩阵（需要转置）
        affine_matrix3 = clinkPointCalibration()
        # affine_matrix3 = [[-3.6186,1.9233,2.6959,-0.32744],[9.9873,-53.746,79.388,-59.674],[23.593,-109.42,156.08,-119.02],[0,0,0,1]]
        fileindex = 1

        while True:
            # 返回相机坐标系到初始相机坐标系的转换矩阵（需要转置）
            affine_matrix2 = cameraMoveAutomatic(step)
            # affine_matrix2 = [[-3.6186,1.9233,2.6959,-0.32744],[9.9873,-53.746,79.388,-59.674],[23.593,-109.42,156.08,-119.02],[0,0,0,1]]
            
            count_repeat_index = 0


            # 达到十秒后统计累加结果转换为变换矩阵，重复十次收集结果，将初始变换矩阵和累加矩阵叠加到
            while True:
                # time.sleep(10)    # 等待相机稳定
                color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()
                if img_color is None or img_depth is None:
                    continue

                images = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
                detected_info = []

                process_card("blue", blue_detection_model, color_intrin, depth_intrin, images, aligned_depth_frame, detected_info)
                process_card("red", red_detection_model, color_intrin, depth_intrin, images, aligned_depth_frame, detected_info)

                # 输出检测信息
                for info in detected_info:
                    if info['point_flag'] == 'False':
                        affine_matrix1 = np.vstack((np.hstack((info['rotation_matrix'].T, info['position'].reshape(-1, 1))), [0, 0, 0, 1]))
                        affine_matrix_final = np.dot(affine_matrix3, np.dot(affine_matrix2, affine_matrix1))
                        print(
                            f"目标 {info['index']} ({info['color']}): "
                            f"三维位置 ({info['position'][0]:.1f}cm, {info['position'][1]:.1f}cm, {info['position'][2]:.1f}cm)，"
                            f"距离 {info['distance']:.1f}cm，"
                            f"旋转矩阵: \n请手动勾画处理\n"
                        )
                        with open(f"/home/pi/Desktop/realsense/detectResult/output{fileindex:d}_{count_repeat_index:d}.txt", "a") as file:
                            # 格式化字符串
                            formatted_text = (
                                f"目标 {info['index']} ({info['color']}): "
                                f"三维位置 ({info['position'][0]:.1f}cm, {info['position'][1]:.1f}cm, {info['position'][2]:.1f}cm)，"
                                f"旋转矩阵:\n请手动勾画处理\n"
                            )
                            # 写入文件
                            file.write(formatted_text)
                    else:
                        affine_matrix1 = np.vstack((np.hstack((info['rotation_matrix'].T, info['position'].reshape(-1, 1))), [0, 0, 0, 1]))
                        affine_matrix_final = np.dot(affine_matrix3.T, np.dot(affine_matrix2.T, affine_matrix1.T))
                        print(
                            f"目标 {info['index']} ({info['color']}): "
                            f"三维位置 ({info['position'][0]:.1f}cm, {info['position'][1]:.1f}cm, {info['position'][2]:.1f}cm)，"
                            f"距离 {info['distance']:.1f}cm，"
                            f"旋转矩阵:\n{affine_matrix_final}\n"
                        )
                        # 打开一个文件用于写入
                        with open(f"/home/pi/Desktop/realsense/detectResult/output{fileindex:d_{count_repeat_index:d}}.txt", "a") as file:
                            # 格式化字符串
                            formatted_text = (
                                f"目标 {info['index']} ({info['color']}): "
                                f"三维位置 ({info['position'][0]:.1f}cm, {info['position'][1]:.1f}cm, {info['position'][2]:.1f}cm)，"
                                f"旋转矩阵:\n{affine_matrix_final}\n"
                            )
                            # 写入文件
                            file.write(formatted_text)

                if count_repeat_index >= 3:
                    break
                else:
                    count_repeat_index += 1
                    cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                    cv2.resizeWindow('detection', 640, 480)
                    images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"/home/pi/Desktop/realsense/detectResult/outputimg{fileindex:d}_{count_repeat_index:d}.png", images)
                    cv2.imshow('detection', images)
                    cv2.waitKey(1)
                    
            step += 1
            fileindex += 1
            affine_matrix3 = np.dot(affine_matrix2, affine_matrix3)
            
            if step >= 3:
                break



    finally:
        pipeline.stop()

# 入口
if __name__ == '__main__':
    blue_weights = r'/home/pi/Desktop/realsense/blue-best.pt'
    red_weights = r'/home/pi/Desktop/realsense/red-best.pt'
    main(blue_weights, red_weights)

