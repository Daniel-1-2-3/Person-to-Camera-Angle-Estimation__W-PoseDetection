#laptop is the server
import socket
import pickle
import cv2
import numpy as np
import time
from lock_target import Target

#issue: no wating time after sending 'done processing' message. Server code goes to receiving mode, while client code still trying to receive the done processing msg
#solved by adding try/except blocks 
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '192.168.40.11'
port = 10000 #port can be any integer above 1024, ports below 1024 reserved for system services
server_socket.bind((host, port))
server_socket.listen(5) #listen for connection from client
print(f'Server listening on {host}:{port}')
client_socket, addr = server_socket.accept() #accept connection request from client
print(f'Connection established from {addr}')

def receive_img():
    searcher = Target()
    while True:
        try:
            data = b""
            image_chunk = client_socket.recv(2048)
            while image_chunk:
                data += image_chunk
                image_chunk = client_socket.recv(2048)
                if b'<END>' in image_chunk:
                    data += image_chunk[:-len(b'<END>')]
                    break

            #frame = cv2.imread('C:\\Daniel\\Python\\AutoAim\\frame.jpg')
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) #decode binary frame 
            if frame is not None:
                frame, angle_h, angle_v = searcher.find_target_coordinates(frame)
                cv2.imshow('Frame', frame)
        
            #send a msg to client to say done processing
            client_socket.sendall(b'DONE PROCESSING')
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        except (socket.timeout, socket.error) as e:
            print(f'Connection error: {e}')
            break
        
receive_img()
client_socket.close()       

