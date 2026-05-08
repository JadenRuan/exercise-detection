import asyncio
from bleak import BleakScanner, BleakClient

DEVICE_NAME  = "ESP32_IMU"
CHAR_UUID    = "12345678-1234-1234-1234-123456789abd"
DEVICE_ADDRESS = "68:B6:B3:3E:11:22"

def on_notify(_handle, data: bytearray):
    print(data.decode("utf-8", errors="replace"))

async def main():
    print("Connecting to ESP32_IMU ...")
    async with BleakClient(DEVICE_ADDRESS) as client:
        print("Connected!")
        await client.start_notify(CHAR_UUID, on_notify)
        print("Streaming — Ctrl-C to stop")
        await asyncio.sleep(3600)

asyncio.run(main())