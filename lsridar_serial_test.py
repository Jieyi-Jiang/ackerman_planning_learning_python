import serial
import time

ser = serial.Serial("COM4", 512000, timeout=1, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
time.sleep(0.5)
read_data = ser.readline() # 读取一行数据
print(read_data.decode()) # 打印读取的数据
ser.close() # 关闭串口连接