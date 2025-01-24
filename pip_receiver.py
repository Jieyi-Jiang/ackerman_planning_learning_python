import os
import mmap
import win32file
import win32pipe
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

if __name__ == "__main__":
    # 连接到命名管道
    pipe_name = r'\\.\pipe\position'

    hPipe = win32file.CreateFile(
        pipe_name,
        win32file.GENERIC_READ | win32file.GENERIC_WRITE,
        0,
        None,
        win32file.OPEN_EXISTING,
        0,
        None
    )

    # pipe_name = r'\\.\pipe\target_position'
    # hPipe2 = win32file.CreateFile(
    #     pipe_name,
    #     win32file.GENERIC_READ | win32file.GENERIC_WRITE,
    #     0,
    #     None,
    #     win32file.OPEN_EXISTING,
    #     0,
    #     None
    # )


    # 等待管道连接
    # win32pipe.ConnectNamedPipe(hPipe, None)

    # # 向服务端发送数据
    # data_to_send = "Hello from Python client"
    # win32file.WriteFile(hPipe, data_to_send.encode('utf-8'))

    # 从服务端读取响应
    # buffer = win32file.ReadFile(hPipe, 1024, None)
    # # response = buffer.decode('ascii')
    # print("Received from server:", buffer)

    trajectory_x = []
    trajectory_y = []
    fig = plt.figure()
    plt.figure(1)

    while True:
        buffer = win32file.ReadFile(hPipe, 1024, None)
        # response = buffer.decode('ascii')
        float_numbers = [0, 0, 0]
        for i in range(3):
            float_numbers[i] = struct.unpack('f', buffer[1][i*4 : 4+i*4])[0]
        if buffer == b'':
            break
        # print("Received from server:", buffer)
        print("Received float number:", end='')
        for number in float_numbers:
            print('{:.4f}'.format(number), end=' ')
        print()
        # 处理接收到的数据
        plt.cla()
        plt.gca().set_aspect('equal', adjustable='box')
        trajectory_x.append(float_numbers[0])
        trajectory_y.append(float_numbers[1])
        plt.plot(trajectory_x, trajectory_y, 'blue')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        # plt.pause(0.01)
        plt.pause(0.001)
    # 断开连接
    win32file.CloseHandle(hPipe)
    hPipe.Close()