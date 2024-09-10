import numpy as np

import ctypes as ct
import threading
import time

from ..acquisition_base import BaseAcquisition

try:
    import pyllt as llt
except:
    pass


class Buffer:
    def __init__(self, resolution, data_width):
        """
        Class for storing the temporary profile buffer, with callback function.
        """
        self.profile_buffer = (ct.c_ubyte*(resolution * data_width))()
        self.event = threading.Event()
        self.event.set()


    def __call__(self, data, size, user_data):
        if user_data == 1:
            ct.memmove(self.profile_buffer, data, size)

            self.event.set()



class Scanner:
    """
    This is a class for setting up a Micro Epsilon laser profile sensor.

    To use this class, you need to download the windows SDK from:
    https://www.micro-epsilon.com/2d-3d-measurement/laser-profile-scanners/software/download/
    
    Installation instructions:

    - when downloading, select scanCONTROL Windows SDK (C/C++, C#, Python, VB.NET)
    
    - in the download files locate the pyllt folder

    - Next we build and install the library with setuptools. Refer to 
    https://setuptools.pypa.io/en/latest/userguide/quickstart.html for more details.
    
    1. Create a new pyllt folder and add the copied pyllt folder. You should have something like:
    pyllt
        pyllt
            __init__.py
            llt_datatypes.py
            LLT.dll
            pyllt.py
    
    2. next add the following files to the upper pyllt folder:
        -README.md
        -setup.py
        -LICENCE
        -MANIFEST.in
    
    3. in setup.py copy the following code:

    from setuptools import setup

    if __name__ == "__main__":
        setup(
            name="pyllt",
            version="4.1.1",
            install_requires=[],
            include_package_data=True
        )

    4. add this line to MANIFEST.in
    include pyllt/LLT.dll

    5. add a valid licence to LICENCE
    - eg. The MIT License (MIT)

    6. cd to pyllt folder and run:
    -> python -m build
    Note that you need to install build before.

    7. install the package (look for the created dist folder):
    -> pip install dist/pyllt-4.1.1-py3-none-any.whl

    Pyllt should now be installed inside your venv and can be imported like a
    normal package.
    """
    _default_channel_names = {
        "X": [float, ct.c_double],
        "Z": [float, ct.c_double],
        "I": [int, ct.c_ushort],
        "T": [int, ct.c_ushort],
        "W": [int, ct.c_ushort],
        "M0": [int, ct.c_uint],
        "M1": [int, ct.c_uint]
    }

    def __init__(self, exposure_time: int = 100,
                 idle_time: int = 700, resolution_id: int = 0):
        """
        Args:
            acquisition_name (str, optional): Name of the class. Dafaults to 
                None.
            exposure_time (int): Shutter open time, defaults to 100
            idle_time (int): Shutter close time, defaults to 700
            resolution (int): The scanner has multiple available resolutions, 
                eg. ltt25xx has a list of posible values [640, 320, 160, 0]. 
                Resolution must be an int 0-3. Defaut value is 0.
            channel_names list[str]: a list of channel names, the available 
                names are as following ["X", "Z", "I", "T", "W", "M0", "M1"],
                where each channel represents a datatype transmited from the
                scanner.
        """

        try:
            llt
        except:
            raise Exception("Pyllt library not found. Please install it" \
                            "before using this class.")
        
        self.exposure_time = exposure_time
        self.idle_time = idle_time
        self.resolution_id = resolution_id

        self.hLLT = llt.create_llt_device(llt.TInterfaceType.INTF_TYPE_ETHERNET)
        self._initialize_scanner()
        self.is_on = True
        self._connect_scanner()
        self._setup_laser()
        self.update_event = threading.Event()
        self._termination_flag = False
        self.llt = llt
        self.data_buffer = []


    def _initialize_scanner(self):
        """
        seting up interface
        """
        available_interfaces = (ct.c_uint*6)()

        ret = llt.get_device_interfaces_fast(self.hLLT, available_interfaces, 
                                             len(available_interfaces))
        if ret < 1:
            raise ValueError("Error getting interfaces : " + str(ret))

        ret = llt.set_device_interface(self.hLLT, available_interfaces[0], 0)
        if ret < 1:
            raise ValueError("Error setting device interface: " + str(ret))


    def _connect_scanner(self):
        """
        Connect to the scanner
        """
        ret = llt.connect(self.hLLT)

        if ret < 1: 
            raise Exception(f"Failed to connect to scanner:  {ret}")
        
        # get scanner type
        self.scanner_type = ct.c_int(0)

        ret = llt.get_llt_type(self.hLLT, ct.byref(self.scanner_type))

        if ret < 1:
            raise Exception(f"Error getting scanner type: {ret} ")


    def disconnect_scanner(self):
        """
        Disconnect the scanner
        """
        self.turn_off_laser()

        #TODO check if already disconnected?
        ret = llt.disconnect(self.hLLT)
        if ret < 1:
            raise Exception(f"Error while disconnect: {ret}")


    def turn_on_laser(self):
        """
        Method to turn on the laser.
        """
        if not self.is_on:
            ret = llt.set_feature(self.hLLT, llt.FEATURE_FUNCTION_LASER, 2)

            if ret < 1:
                raise Exception(f"Failed to turn on the laser {ret}")

            self.is_on = True
        
        else:
            print("Laser is already turned on.")


    def turn_off_laser(self):
        """
        Method to turn off the laser.
        """
        if self.is_on:
            ret = llt.set_feature(self.hLLT, llt.FEATURE_FUNCTION_LASER, 0)

            if ret < 1:
                raise Exception(f"Failed to turn off the laser {ret}")
            
            self.is_on = False

        else:
            print("Laser is already turned off")


    def _setup_laser(self):
        """
        setup the scanner parameters

        The sample rate is defined by exposure and idle time by the 
        following equation:

        f = 10^5/(t_Exposure + t_Idle) Hz

        Times can be set with steps of 10 us (us -> micro seconds). So to
        achieve an exposure time of 1 ms, exposure_time must be set as 100,
        so -> 100 * 10 us = 1000 us.
        """
        available_resolutions = (ct.c_uint*4)()

        # set resolution
        if isinstance(self.resolution_id, int) and (
            self.resolution_id >= 0 and self.resolution_id <= 3):    
            ret = llt.get_resolutions(
                self.hLLT, available_resolutions, 4
                )
            if ret < 1:
                raise ValueError(f"Error getting resolutions: {ret}")
        else:
            raise Exception(f"The passed value {self.resolution_id} for" \
                            " self.resolution is not valid." \
                            "\nself.resolution must be an int 0-3!")

        self.resolution = available_resolutions[self.resolution_id]

        ret = llt.set_resolution(self.hLLT, self.resolution)
        if ret < 1:
            raise ValueError(f"Error setting resolution: {ret}")

        # set sample rate
        if (isinstance(self.exposure_time, int) and 
                isinstance(self.idle_time, int)):
            ret = llt.set_feature(
                self.hLLT, llt.FEATURE_FUNCTION_EXPOSURE_TIME, 
                self.exposure_time
                )
            if ret < 1:
                raise ValueError(f"Error setting resolution: {ret}")
            
            ret = llt.set_feature(self.hLLT, llt.FEATURE_FUNCTION_IDLE_TIME, 
                                  self.idle_time)
            if ret < 1:
                raise ValueError(f"Error setting resolution: {ret}")
        else:
            raise Exception(f"Passed values for exposure and idle time are" \
                        f" not ints: ({self.exposure_time}, {self.idle_time})")

        #TODO to ne velja v primer laserjeve serije 30xx -> preverjaj verzijo in 
        # korektno izracunaj sample_rate
        # dobimo v hz
        self.sample_rate = 100_000/(self.exposure_time+self.idle_time)
        # print(f"f = {self.sample_rate:.1f} hz")


    def profile_setup(self, channel_names):
        #TODO ustrezno nastavi glede na zahtevane channele
        self.start_data = 0
        self.data_width = 16
        # _profile_buffer = (ct.c_ubyte*(self.resolution * self.data_width))()
        # c55
        # self.scanner_buffer = Buffer(self.resolution, self.data_width)
        self._pointers = {}
        self._buffer_arrays = {}
        # moramo si shranit, ker rabimo v _read_data_loop-u
        self._channel_names = channel_names

        for name, data_type in self._default_channel_names.items():
            if name in channel_names:
                self._buffer_arrays[name] = np.empty(self.resolution, 
                                                     dtype=data_type[0])
                self._pointers[name] = self._buffer_arrays[name].ctypes.data_as(
                    ct.POINTER(data_type[1])
                )
            else:
                self._pointers[name] = ct.POINTER(data_type[1])()

        # Set partial profile as profile config
        ret = llt.set_profile_config(
            self.hLLT, llt.TProfileConfig.PARTIAL_PROFILE
            )
        if ret < 1:
            raise Exception(f"Error setting profile config: {ret}" )
        
        self._partial_profile_struct = llt.TPartialProfile(
            0, self.start_data, self.resolution, self.data_width
            )

        # set partial profile
        ret = llt.set_partial_profile(
            self.hLLT, ct.byref(self._partial_profile_struct)
            )
        if ret < 1:
            raise Exception(f"Error setting partial profile: {ret}")
        

    def _stop_transfer(self) -> None:
        """
        Stop profile transfer
        
        Returns None.
        """
        ret = llt.transfer_profiles(self.hLLT, 
                            llt.TTransferProfileType.NORMAL_TRANSFER, 0)
        if ret < 1:
            raise Exception(f"Error stopping transfer profiles: {ret}")



class MELaserScanner(BaseAcquisition):
    """
    Acquisition class for LDAQ
    """
    def __init__(self, acquisition_name: str = None, device: Scanner = None,
                 channel_names: list = None) -> None:
        """
        Args:
            acquisition_name (str, optional): Name of the class. Dafaults to 
                None.
            device Scanner: instance of Scanner class.
            channel_names list[str]: a list of channel names, the available 
                names are as following ["X", "Z", "I", "T", "W", "M0", "M1"],
                where each channel represents a datatype transmited from the
                scanner.
        """
        
        super().__init__()
 
        if acquisition_name is None:
            self.acquisition_name = "Micro Epsilon"
        else:
            self.acquisition_name = acquisition_name
        
        self.device = device
            
        if channel_names is None:
            self._channel_names = ["X", "Z", "I", "T", "W", "M0", "M1"]
        else:
            #TODO check for correct channel names?
            self._channel_names = channel_names

        self._channel_names_init = [
                f"{name}_{i}" for name in self._channel_names for i in range(
                self.device.resolution)
            ]

        # self._channel_names_init = ["all"]
        # self._spacer = self.device.resolution * len(self._channel_names)
        # self._channel_names_video_init = self._channel_names
        # self._channel_shapes_video_init = [[640,1] for i in range(len(self._channel_names))]

        self.channel_idx = [i for i in range(len(self._channel_names))]

        self.sample_rate = self.device.sample_rate#s*self._spacer
        self._timeout_time = 1/self.device.sample_rate*2
        self.device.profile_setup(self._channel_names)

        self.set_trigger(1e20, 0, duration=1.0)

        # self.set_data_source()


    def terminate_data_source(self) -> None:
        """
        Properly closes acquisition source after the measurement.
        
        Returns None.
        """
        self.device._termination_flag = True

        self.acq_thread.join()

        # self.device._stop_transfer()


    def get_sample_rate(self):
        """
        Returns sample rate of acquisition class.
        This function is also useful to compute sample_rate estimation if no
        sample rate is given.
        
        Returns self.sample_rate
        """
        return self.sample_rate


    def set_data_source(self) -> None:
        """
        setup the data source and start the profile transfer.
        """
        self.clear_buffer()

        self.acq_thread = threading.Thread(
            target = _read_data_loop, args = [self.device]
        )
        self.acq_thread.start()

        super().set_data_source()


    def read_data(self) -> np.ndarray:
        """
        Read data from the scanner.
        """
        while True:
            
            # vrne False, ce se izvede timeout
            _flag = self.device.update_event.wait(self._timeout_time)

            data = np.array(
                self.device.data_buffer
                ).T
            print(data.shape)
            if len(data.shape) < 1:
                data.reshape(-1, 1)
                print("Only 1")
            if not _flag:
                data = np.zeros(self.device.resolution*len(self._channel_names)
                                ).reshape(-1,1)
                print("Added an empty profile, proceed with caution")
            
            self.clear_buffer()
            break

        self.device.update_event.clear()

        return data
    
    
    def clear_buffer(self) -> None:
        """
        clear the device buffer list
        """

        self.device.data_buffer = []



def _data_generator(device: Scanner, buffer: Buffer):
    """
    Generator for data acquisition

    device: instance of the Scanner class
    buffer: instance of the Buffer class
    """

    while True:
        buffer.event.wait()

        fret = llt.convert_part_profile_2_values(
            device.hLLT, buffer.profile_buffer,
            ct.byref(device._partial_profile_struct), device.scanner_type, 0,
            1, device._pointers["W"], device._pointers["I"], 
            device._pointers["T"], device._pointers["X"], 
            device._pointers["Z"], device._pointers["M0"],
            device._pointers["M1"]
            )

        if fret & llt.CONVERT_X == 0 or fret & llt.CONVERT_Z == 0:
            raise ValueError("Error converting data: " + str(fret))

        buffer.event.clear()

        yield


def _read_data_loop(device: Scanner):
    """
    Function to run the acquisition generator in a loop
    """
    buffer = Buffer(device.resolution, device.data_width)
    get_profile_cb = llt.buffer_cb_func(buffer)

    ret = llt.register_callback(
        device.hLLT, llt.TCallbackType.C_DECL, get_profile_cb, 1
        )
    get_profile_cb = llt.buffer_cb_func(buffer)

    ret = llt.register_callback(
        device.hLLT, llt.TCallbackType.C_DECL, get_profile_cb, 1
        )
    if ret < 1:
        raise Exception(f"Error setting callback: {ret}")
    
    time.sleep(0.1)
    ret = llt.transfer_profiles(device.hLLT, 
                                llt.TTransferProfileType.NORMAL_TRANSFER, 1)
    if ret < 1:
        raise Exception(f"Error starting transfer profiles: {ret}")

    time.sleep(0.1)

    # device.update_event = threading.Event()
    gen_fun = _data_generator(device, buffer)

    i = 0
    while True:
        if device._termination_flag:
            device._stop_transfer()
            time.sleep(0.1)
            gen_fun.close()
            break
        
        next(gen_fun)

        data = np.array(
                [device._buffer_arrays[name] for name in device._channel_names]
            ).T.flatten()
        
        device.data_buffer.append(data)
        # print(i, device._buffer_arrays["Z"][100])
        device.update_event.set()

        i += 1

