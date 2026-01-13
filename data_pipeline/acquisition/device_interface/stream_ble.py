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
from utils import butter_bandpass_filter
from handle import HandlePPG

from define import NUM_CH, PPG_GAIN, CHANNELS_COLOR, CHANNELS_LABEL, SAMPLE_TIME_1S_RAW_DATA, SAMPLE_TIME_1S_PPG_DATA, \
    PORT, BAUD_RATE, FS_128, DIR_SAVE, SUBJECT_ID

ADDRESS = "F5:B5:E7:39:61:46"  # 002211213779
# ADDRESS = "F0:6E:C3:86:E5:4B"  # 002211933040 -> Device D3
# ADDRESS = "C5:76:6E:78:2D:06"  # 002211330787 -> Device D6
# ADDRESS = "C9:29:18:69:20:6E" # 002211298891

RX_CHARACTERISTIC_UUID = "00000001-0000-1000-8000-008025000000"
RXC_CHARACTERISTIC_UUID = "00000003-0000-1000-8000-008025000000"

TX_CHARACTERISTIC_UUID = "00000002-0000-1000-8000-008025000000"
TXC_CHARACTERISTIC_UUID = "00000004-0000-1000-8000-008025000000"

STREAM_CHARACTERISTIC_UUID = "0000000b-0000-1000-8000-008025000000"

NUMBER_OF_SEQUENCE = 3

buffer = bytearray()
bufferBegin = bytearray()
flagConnect = 0
timeStart = 0
timeEnd = 0
total_time = 0

sample_count = []
ppg_sensor_data = []
ppg_sensor_data_store = [[] for _ in range(NUMBER_OF_SEQUENCE)]

# Initialize list to store last 300 samples
max_samples = 2560

# step state machine
step = 0

# Initialize the CRC function with the CRC-CCITT (0xFFFF) polynomial
crc_ccitt_fn = crcmod.predefined.mkPredefinedCrcFun("crc-ccitt-false")


# Hàm update animation
def update_plot(frame):
    global ax1

    # Update plot data
    if (len(ppg_sensor_data) == (SAMPLE_TIME_1S_PPG_DATA)):
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
            ax1[i].plot(sample_count[-max_samples:], ppg_sensor_data_store[i][-max_samples:], color=colors[i],
                        label='PPG_{}'.format(channels[i]))
            ax1[i].set_title('{}'.format(channels[i]))
            ax1[i].set_ylabel('Amplitude ')
            ax1[i].legend(loc='upper right')

        ppg_sensor_data.clear()


def close_plot(event):
    plt.close()  # Đóng cửa sổ
    sys.exit()  # Thoát chương trình


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


def disconnected_callback(client):
    global flagConnect
    print("Disconnected from device")
    flagConnect = 2


async def connect_ble():
    async with BleakClient(ADDRESS) as client:
        global flagConnect
        client.set_disconnected_callback(disconnected_callback)
        print("Connect to device: " + ADDRESS)
        sys.stdout.flush()
        if client.is_connected:
            flagConnect = 1

            async def send_response(characteristic, response_data):
                await client.write_gatt_char(RXC_CHARACTERISTIC_UUID, b'\x20', response=False)
                MTU_SIZE = 20
                offset = 0
                while offset < len(response_data):
                    chunk = response_data[offset:offset + MTU_SIZE]
                    await client.write_gatt_char(characteristic, chunk, response=False)
                    offset += MTU_SIZE

            def callbackTX(sender, data):
                global bufferBegin

                # Append the received data to the buffer
                bufferBegin += data
                if len(bufferBegin) >= 63:
                    msgpack_data = msgpack.unpackb(bufferBegin[2:63])

                    EXPECTED_HANDSHAKE = [
                        49152,
                        0,
                        1000,
                        # "002211298891",
                        # "002211933040",
                        # "002211330787",
                        # "002211851663",
                        "002211213779",
                        # "002211399433",
                        # "002211489564",
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
                        ### Step 1: Send a response handshake message
                        response_handshake = [
                            49152,
                            57344,
                            int(time.time()),
                            0,
                            "BLE"
                        ]

                        response_data = msgpack.packb(response_handshake)
                        response_data = b'\x01' + bytes([len(response_data)]) + response_data

                        # Add the CRC-CCITT (0xFFFF) checksum to the response data
                        checksum = crc_ccitt_fn(response_data)

                        print("\nHANDSHAKE...")
                        sys.stdout.flush()
                        asyncio.create_task(
                            send_response(RX_CHARACTERISTIC_UUID, response_data + checksum.to_bytes(2, "little")))
                        print("\nHANDSHAKE OK")
                    else:
                        # Print out received data
                        print(f"Received data: {msgpack_data}")
                        sys.stdout.flush()

                    # Reset the buffer
                    buffer = bytearray()

            def callbackSTREAM(sender, data):
                global buffer
                global timeEnd, timeStart, total_time

                # Append the received data to the buffer
                buffer += data
                timeEnd = time.time()
                if buffer[0] != 98 or buffer[1] != 101 or buffer[2] != 103 or buffer[3] != 105 or buffer[4] != 110:
                    print("Miss data callbackSTREAM")
                    sys.stdout.flush()

                    # Reset the buffer
                    buffer = bytearray()
                elif len(buffer) == (SAMPLE_TIME_1S_RAW_DATA + 5):
                    convert = lambda byte: int.from_bytes(byte, byteorder='big', signed=False)
                    signal = [convert(buffer[k:k + 3]) for k in range(5, len(buffer), 3)]
                    ppg_sensor_data.extend(signal)

                    # process time
                    current_time = round(timeEnd - timeStart, 2)
                    length = len(buffer)
                    timeStart = timeEnd
                    total_time += 1

                    buffer.clear()

                    ppg_npy = np.asarray(ppg_sensor_data)
                    ppg_npy = ppg_npy.reshape((-1, NUM_CH))
                    print(f"{current_time} - {length} -------- {total_time} --- {len(ppg_npy)}")

                    np.save('test.npy', ppg_npy)

                    for j in range(0, 256):
                        sample_count.append(len(sample_count) + 1)

            # Start notifications
            await client.start_notify(TX_CHARACTERISTIC_UUID, callbackTX)
            await client.start_notify(STREAM_CHARACTERISTIC_UUID, callbackSTREAM)

            # Wait for the animation to finish
            while client.is_connected:
                # if not plt.get_fignums():  # Check if the plot window is closed
                #     break
                await asyncio.sleep(1)

            # Stop notifications
            await client.stop_notify(TX_CHARACTERISTIC_UUID)
            await client.stop_notify(STREAM_CHARACTERISTIC_UUID)

            # Disconnect from the device
            await client.disconnect()
            flagConnect = 2
            print("Diconnect BLE")
            sys.stdout.flush()


def start_ble_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(connect_ble())
    loop.close()


def plot_sig_realtime_thread():
    plt.ion()
    fig = plt.figure(figsize=(15, 8))
    ax = []
    for i in range(6):
        ax.append(fig.add_subplot(3, 2, i + 1))

    def plot_sig_realtime(s_id, ppg, seconds):
        ppg = ppg / PPG_GAIN
        time = - 256 * 10
        ppg_raw = -ppg

        ppg = -butter_bandpass_filter(ppg.T, 0.5, 10, 256, 2).T

        ppg_green = ppg[:, 1]
        hd = HandlePPG(fs=FS_128)

        try:
            # _, percent = hd.calculate_sqi_real_time_green(ppg_green, thr_sqi=20)
            _, percent = hd.calculate_sqi_green(ppg_green)
            mm_ss = str(datetime.timedelta(seconds=seconds))
            fig.suptitle('Subject ID: {}\nDuration time: {}\nPass SQI: {}%'.format(s_id, mm_ss, percent))

        except Exception as error:
            print('{} - Error'.format(error))

        for i_raw in range(0, 6, 2):
            ch = i_raw // 2
            ax[i_raw].cla()
            ax[i_raw].plot(ppg_raw[time:, ch], color=CHANNELS_COLOR[ch], label=CHANNELS_LABEL[ch])
            ax[i_raw].legend(loc='upper right')

            i = i_raw + 1
            ax[i].cla()
            ax[i].plot(ppg[time:, ch], color=CHANNELS_COLOR[ch], label=CHANNELS_LABEL[ch])
            ax[i].legend(loc='upper right')
        ax[0].set_title('RAW')
        ax[1].set_title('FILTER')

        fig.canvas.draw()
        fig.canvas.flush_events()

    while 1:
        try:
            ppg = np.load('test.npy')
            seconds = len(ppg) // 256
            # print(seconds)
            plot_sig_realtime('123', ppg, seconds)
            time.sleep(0.1)
        except Exception as e:
            continue


# Tạo một luồng riêng biệt để kết nối BLE
ble_thread = threading.Thread(target=start_ble_thread)
plot_thread = threading.Thread(target=plot_sig_realtime_thread)
ble_thread.start()
plot_thread.start()
