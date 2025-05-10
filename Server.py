import socket
import threading
import pickle
import struct  # 用于处理固定长度的数据（如数据长度）
import time
import asyncio
import websockets

clients = []
lock = threading.Lock()
calculation_count = 0

#---根据部署不同数量的客户端，调整全局模型参数聚合的计算方式，当前为FedAvg---
def calculate_average_values1(client1_data, client2_data):
    # 假设客户端发送的数据是一个字典，这里计算两个字典对应键的平均值
    average_data = {}
    for key in client1_data:
        if key in client2_data:
            average_data[key] = (client1_data[key] + client2_data[key]) / 2
    return average_data

def calculate_average_values(client1_data, client2_data, client3_data):

    average_data = {}
    all_keys = set(client1_data.keys()).union(client2_data.keys(), client3_data.keys())

    for key in all_keys:
        values = []
        if key in client1_data:
            values.append(client1_data[key])
        if key in client2_data:
            values.append(client2_data[key])
        if key in client3_data:
            values.append(client3_data[key])
            average_data[key] = (client1_data[key]+client2_data[key]+client3_data[key]) / 3
    return average_data

def handle_client(client_socket, client_address):
    print(f"Accepted connection from {client_address}")
    time.sleep(5)
    # 接收数据长度（4字节）
    data_length_bytes = client_socket.recv(4)
    if len(data_length_bytes) != 4:
        print("Error receiving data length from client.")
        client_socket.close()
        return
    data_length = struct.unpack('>I', data_length_bytes)[0]  # 转换为无符号整数
    # 根据接收到的数据长度接收实际的数据
    data = b''
    bytes_recd = 0
    while bytes_recd < data_length:
        packet = client_socket.recv(min(1024, data_length - bytes_recd))  # 缓冲区大小设为1024
        if not packet:
            print("Connection closed by client.")
            client_socket.close()
            return
        data += packet
        bytes_recd += len(packet)
    received_data = pickle.loads(data)
    # print(f"Received data from {client_address}: {received_data}")
    print(f"Received data from {client_address}")
    time.sleep(5)
    with lock:
        clients.append((client_socket, received_data))
        # 检查是否有三个客户端的数据
        if len(clients) == 3:
            global calculation_count
            calculation_count += 1
            client1_data, client2_data, client3_data = clients[0][1], clients[1][1], clients[2][1]
            average_data = calculate_average_values(client1_data, client2_data, client3_data)
            print(f"Calculation {calculation_count}: The average data is {average_data}")
            time.sleep(5)
            # 发送平均值数据回客户端
            for client_sock, _ in clients:
                response_data = pickle.dumps(average_data)
                response_length = len(response_data)
                client_sock.sendall(struct.pack('>I', response_length))
                client_sock.sendall(response_data)
                client_ip, _ = client_sock.getpeername()
                print(f"Data successfully sent to client at IP: {client_ip}")

            # 关闭客户端连接并清空clients列表
            for client_sock, _ in clients:
                client_sock.close()
            clients.clear()


def start_server(host='localhost', port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        # print(f"Server listening on {host}:{port}")
        print(f"Server listening on {host}:{port}")
        time.sleep(5)
        while True:
            client_socket, client_address = s.accept()
            client_thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
            client_thread.start()

if __name__ == "__main__":
    start_server()