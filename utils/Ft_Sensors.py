import minimalmodbus
import struct
import time
import torch

class DR304ForceSensor:
    def __init__(self, port='COM3', baudrate=115200, slave_address=1):
        self.instrument = minimalmodbus.Instrument(port, slave_address)
        self.instrument.serial.baudrate = baudrate
        self.instrument.serial.bytesize = 8
        self.instrument.serial.parity   = minimalmodbus.serial.PARITY_NONE
        self.instrument.serial.stopbits = 1
        self.instrument.serial.timeout  = 0.1
        self.instrument.mode = minimalmodbus.MODE_RTU

        self.clear_all_channels()

    def clear_all_channels(self):
        self.instrument.write_registers(0x0A20, [0x0000, 0x0007])
        print("[INFO] All channels zered.")

    def read_force_data(self):
        raw = self.instrument.read_registers(0x0A00, 12, functioncode=3)
        data = []
        for i in range(0, len(raw), 2):
            hi, lo = raw[i], raw[i+1]
            combined = (hi << 16) | lo
            signed = struct.unpack('>i', combined.to_bytes(4, 'big'))[0]
            # data.append(signed / 1000.0)
            data.append(signed/100) # N
        return data

    def get_contact(self, threshold=1e-1):
        if(torch.norm(torch.tensor(self.read_force_data(), dtype=torch.float32), dim=-1)) > threshold:
            # print(torch.norm(torch.tensor(self.read_force_data(), dtype=torch.float32)[:3], dim=-1))
            return True
        else:
            return False


    def calculate_normal_force(self):
        force = self.read_force_data()
        normal = torch.norm(torch.tensor(force, dtype=torch.float32)[:3], dim=-1)

        return normal, force / (normal + 1e-8)


if __name__ == "__main__":
    sensor = DR304ForceSensor(port="COM4")
    while True:
        a = time.time()
        values = sensor.read_force_data()
        b = time.time()

        print(sensor.get_contact())
        # print(dict(zip(["Fx", "Fy", "Fz", "Mx", "My", "Mz"], values)))
        # print(torch.norm(torch.tensor(sensor.read_force_data(), dtype=torch.float32), dim=-1))
        # time.sleep(0.005)
