#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import serial  # 导入模块
import serial.tools.list_ports
import threading
import struct
from serial import EIGHTBITS, PARITY_NONE, STOPBITS_ONE

# 宏定义参数
PI = 3.1415926
FRAME_HEAD = str('fc')
FRAME_END = str('fd')
TYPE_IMU = str('40')
TYPE_AHRS = str('41')
TYPE_INSGPS = str('42')
TYPE_GEODETIC_POS = str('5c')
TYPE_GROUND = str('f0')
TYPE_SYS_STATE = str('50')
TYPE_BODY_ACCELERATION = str('62')
TYPE_ACCELERATION = str('61')
TYPE_MSG_BODY_VEL = str('60')
IMU_LEN = str('38')  # //56
AHRS_LEN = str('30')  # //48
INSGPS_LEN = str('48')  # //72
GEODETIC_POS_LEN = str('20')  # //32
SYS_STATE_LEN = str('64')  # // 100
BODY_ACCELERATION_LEN = str('10') #// 16
ACCELERATION_LEN = str('0c')  # 12
PI = 3.141592653589793
DEG_TO_RAD = 0.017453292519943295
isrun = True


# 获取命令行输入参数
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--debugs', type=bool, default=False, help='if debug info output in terminal ')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0', help='the models serial port receive data; example: '
                                                                 '    Windows: COM3'
                                                                 '    Linux: /dev/ttyUSB0')

    parser.add_argument('--bps', type=int, default=921600, help='the models baud rate set; default: 921600')
    parser.add_argument('--timeout', type=int, default=20, help='set the serial port timeout; default: 20')
    # parser.add_argument('--device_type', type=int, default=0, help='0: origin_data, 1: for single imu or ucar in ROS')

    receive_params = parser.parse_known_args()[0] if known else parser.parse_args()
    return receive_params


# 接收数据线程
def imuDataReceive():
    opt = parse_opt()
    # 尝试打开串口
    try:
        serial_ = serial.Serial(port=opt.port, baudrate=opt.bps, bytesize=EIGHTBITS, parity=PARITY_NONE,
                                stopbits=STOPBITS_ONE,
                                timeout=opt.timeout)
    except:
        print("error:  unable to open port .")
        exit(1)
    # 循环读取数据
    while serial_.isOpen():
        if not threading.main_thread().is_alive():
            print('done')
            break
        check_head = serial_.read().hex()
        # 校验帧头
        if check_head != FRAME_HEAD:
            continue
        head_type = serial_.read().hex()
        # 校验数据类型
        if (head_type != TYPE_IMU and head_type != TYPE_AHRS and head_type != TYPE_INSGPS and
                head_type != TYPE_GEODETIC_POS and head_type != 0x50 and head_type != TYPE_GROUND and 
                head_type != TYPE_SYS_STATE and  head_type!=TYPE_MSG_BODY_VEL and head_type!=TYPE_BODY_ACCELERATION and head_type!=TYPE_ACCELERATION):
            continue
        check_len = serial_.read().hex()
        # 校验数据类型的长度
        if head_type == TYPE_IMU and check_len != IMU_LEN:
            continue
        elif head_type == TYPE_AHRS and check_len != AHRS_LEN:
            continue
        check_sn = serial_.read().hex()
        head_crc8 = serial_.read().hex()
        crc16_H_s = serial_.read().hex()
        crc16_L_s = serial_.read().hex()

        # 读取并解析IMU数据
        if head_type == TYPE_IMU:
            data_s = serial_.read(int(IMU_LEN, 16))
            IMU_DATA = struct.unpack('12f ii',data_s[0:56])
            # print(IMU_DATA)
            # print("Gyroscope_X(rad/s): " + str(IMU_DATA[0]))
            # print("Gyroscope_Y(rad/s) : " + str(IMU_DATA[1]))
            # print("Gyroscope_Z(rad/s) : " + str(IMU_DATA[2]))
            # print("Accelerometer_X(m/s^2) : " + str(IMU_DATA[3]))
            # print("Accelerometer_Y(m/s^2) : " + str(IMU_DATA[4]))
            # print("Accelerometer_Z(m/s^2) : " + str(IMU_DATA[5]))
            # print("Timestamp(us) : " + str(IMU_DATA[12]))
            return list(IMU_DATA[0:6]) + [IMU_DATA[12]]
        # 读取并解析AHRS数据
        elif head_type == TYPE_AHRS:
            data_s = serial_.read(int(AHRS_LEN, 16))
            AHRS_DATA = struct.unpack('10f ii',data_s[0:48])


if __name__ == "__main__":
    while True:
        try:
            print(imuDataReceive())
        except(KeyboardInterrupt, SystemExit):
            break
