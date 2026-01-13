import asyncio
from bleak import BleakClient
from bleak.exc import BleakError
from bleak import discover
import msgpack
import crcmod
import time
import sys
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime

# ADDRESS = "F5:B5:E7:39:61:46" # 002211213779
# ADDRESS = "F0:6E:C3:86:E5:4B"  # 002211933040 -> Device D3
ADDRESS = "C5:76:6E:78:2D:06"  # 002211330787 -> Device D6
# ADDRESS = "C9:29:18:69:20:6E" # 002211298891

RX_CHARACTERISTIC_UUID  = "00000001-0000-1000-8000-008025000000"
RXC_CHARACTERISTIC_UUID = "00000003-0000-1000-8000-008025000000"

TX_CHARACTERISTIC_UUID  = "00000002-0000-1000-8000-008025000000" 
TXC_CHARACTERISTIC_UUID = "00000004-0000-1000-8000-008025000000" 

PPG_GAIN = 10 ** 9 / 31.25

NUMBER_OF_SEQUENCE = 3
SAMPLE_TIME_1S_RAW_DATA = 768 * NUMBER_OF_SEQUENCE
SAMPLE_TIME_1S_PPG_DATA = 256 * NUMBER_OF_SEQUENCE

buffer = bytearray()

sample_count = []
ppg_sensor_data = []
ppg_sensor_data_store = [[] for _ in range(NUMBER_OF_SEQUENCE)]

# Initialize list to store last 300 samples
max_samples = 2560

fig, ax1 = plt.subplots(nrows=NUMBER_OF_SEQUENCE, sharex=True, figsize=(12, 8))

# step state machine
step = 0

# Initialize the CRC function with the CRC-CCITT (0xFFFF) polynomial
crc_ccitt_fn = crcmod.predefined.mkPredefinedCrcFun("crc-ccitt-false")

# Hàm update animation
def update_plot(frame):
    global buffer
    global ax1

    # Update plot data
    if (len(ppg_sensor_data) == SAMPLE_TIME_1S_PPG_DATA):
        buff_for_each_sq = []

        for i in range(NUMBER_OF_SEQUENCE):
            extracted_buffer = ppg_sensor_data[i::NUMBER_OF_SEQUENCE]  # Slicing to extract data for each channel
            buff_for_each_sq.append(extracted_buffer)

        for i in range(NUMBER_OF_SEQUENCE):
            ppg_sensor_data_store[i].extend(buff_for_each_sq[i])

        channels = ['RED', 'GREEN', 'IR', 'AMBIENT']
        colors = ['r', 'g', 'k', 'b']
        for i in range(NUMBER_OF_SEQUENCE):
            ax1[i].clear()
            ax1[i].plot(sample_count[-max_samples:], ppg_sensor_data_store[i][-max_samples:], color=colors[i], label='PPG_{}'.format(channels[i]))
            ax1[i].set_title('{}'.format(channels[i]))
            ax1[i].set_ylabel('Amplitude ')
            ax1[i].legend(loc='upper right')

        ppg_sensor_data.clear()


def close_plot(event):
    plt.close()  # Đóng cửa sổ
    sys.exit()   # Thoát chương trình


def create_plot():
    global buffer
    global fig, ax1
    print("create_plot")
    sys.stdout.flush()
    # Create the animation
    ani = FuncAnimation(fig, update_plot, interval=0)
    fig.canvas.mpl_connect('close_event', close_plot)

    # Start the plot
    plt.show()


async def connect_ble():
    async with BleakClient(ADDRESS) as client:
        print("Connect to device: " + ADDRESS)
        sys.stdout.flush()
        if client.is_connected:

            async def send_response(characteristic, response_data):
                await client.write_gatt_char(RXC_CHARACTERISTIC_UUID, b'\x20', response=False)
                                
                MTU_SIZE = 20 
                offset = 0
                while offset < len(response_data):
                    chunk = response_data[offset:offset + MTU_SIZE]
                    await client.write_gatt_char(characteristic, chunk, response=False)
                    offset += MTU_SIZE

            def callbackTX(sender, data):
                global buffer
                global step

                # Append the received data to the buffer
                buffer += data
                # If the buffer has reached the maximum size, parse the data
                if step == 0 and len(buffer) >= 63:                      
                    msgpack_data = msgpack.unpackb(buffer[2:63])

                    EXPECTED_HANDSHAKE = [
                        49152,
                        0,
                        1000,
                        #"002211298891",
                        #"002211933040",
                        "002211330787",
                        #"002211851663",
                        # "002211213779",
                        #"002211399433",
                        #"002211489564",
                        # "002211298891",
                        "ABPM",
                        "1.0.0.0",
                        "1.0.0.0",
                        "1.0.0.0",
                        "0x00000001"
                    ]

                    # Check if the received message is a handshake
                    if msgpack_data == EXPECTED_HANDSHAKE:
                        # begin state machine
                        if step == 0:
                            ### Step 1: Send a response handshake message
                            response_handshake = [
                                49152,
                                57344,
                                int(time.time()),
                                0,
                                "BLE"
                            ]     

                            response_data = msgpack.packb(response_handshake)      
                            response_data = b'\x01' + bytes([len(response_data)])  + response_data   

                            # Add the CRC-CCITT (0xFFFF) checksum to the response data
                            checksum = crc_ccitt_fn(response_data)

                            print("\nHANDSHAKE...")
                            sys.stdout.flush()
                            asyncio.create_task(send_response(RX_CHARACTERISTIC_UUID,  response_data + checksum.to_bytes(2, "little")))
                            print("\nHANDSHAKE OK")
                            
                            step = 1
                    else:
                        # Print out received data
                        print(f"Received data: {msgpack_data}")
                        sys.stdout.flush()

                    # Reset the buffer
                    buffer = bytearray()
                
                # Stream data
                if (step == 1):
                    if len(buffer) == SAMPLE_TIME_1S_RAW_DATA:
                        print(datetime.datetime.now(), len(buffer))
                        convert = lambda byte: int.from_bytes(byte, byteorder='big', signed=False)
                        signal = [convert(buffer[k:k + 3]) for k in range(0, len(buffer), 3)]
                        signal = np.asarray(signal) / PPG_GAIN

                        ppg_sensor_data.extend(signal)

                        for j in range(0, 256):
                            sample_count.append(len(sample_count) + 1)

                        buffer.clear()

            def callbackTXC(sender, data):
                integer_value = int.from_bytes(data[1:3], byteorder='big')
                #print(' '.join(format(byte, '02x') for byte in data))
                print("SQI: " + str(integer_value))

            # Start notifications
            await client.start_notify(TX_CHARACTERISTIC_UUID, callbackTX) 
            await client.start_notify(TXC_CHARACTERISTIC_UUID, callbackTXC)              

            # Wait for the animation to finish
            while True:
                if not plt.get_fignums():  # Check if the plot window is closed
                    break                
                await asyncio.sleep(0.1)

            # Stop notifications
            await client.stop_notify(TX_CHARACTERISTIC_UUID)

            # Disconnect from the device
            await client.disconnect()
            print("Diconnect BLE")
            sys.stdout.flush() 

def start_ble_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(connect_ble())
    loop.close()

# Tạo một luồng riêng biệt để kết nối BLE
ble_thread = threading.Thread(target=start_ble_thread)
ble_thread.start()

create_plot()