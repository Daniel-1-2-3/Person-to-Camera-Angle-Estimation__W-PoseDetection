#raspberry pi is the client, this code is also on raspberry pi, where picamera2 was installed successfully 
from picamera2 import Picamera2, Preview # type: ignore
import socket, io
import time
import cv2
import json 

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)
            
def connect_to_server():
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            host = '192.168.40.11'
            port = 10000
            client_socket.connect((host, port))
            client_socket.settimeout(2)
            print('Connected to server')
            
            picam2.start()
            
            return client_socket, picam2
        
        except socket.error as e:
            print(f'Connection failed: {e}')
            time.sleep(3) #wait 3 secs before retrying
            
def send_img(client_socket, picam2):
    while True:
        try:
            picam2.capture_file('frame.jpg')
            file = open('frame.jpg', 'rb')

            image_data = file.read(2048)
            while image_data:
                client_socket.sendall(image_data)
                image_data = file.read(2048)
            file.close()
            client_socket.sendall(b'<END>')#end marker

            try:
                ack = client_socket.recv(2048) #wait for msg from server acknowledging done processing
                if b'DONE PROCESSING' in ack:
                    continue
            except socket.timeout:
                print('Timeout')
            except Exception as e:
                print(f'Error occurred: {e}')
                break
            
        except (socket.error, Exception) as e:
            print(f'Error occurred while sending image: {e}')
            break #break from loop, wait for reconnect

while True:
    client_socket, picam2 = connect_to_server()
    send_img(client_socket, picam2)
    client_socket.close()
    picam2.stop()
    print('Connection lost, reconnecting...')
