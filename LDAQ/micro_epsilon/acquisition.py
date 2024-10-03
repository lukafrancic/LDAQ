import numpy as np

import ctypes as ct
import threading
import time

from ..acquisition_base import BaseAcquisition

try:
    import pyllt as llt
except:
    pass


# datatypes for setting up proper buffer arrays
# using appropriate data types to reduce memory storage
# first list value to set numpy type, second for the C array for the scanner
_default_channel_names = {
        "X": [np.float64, ct.c_double],
        "Z": [np.float64, ct.c_double],
        "I": [np.int16, ct.c_ushort],
        "T": [np.int16, ct.c_ushort],
        "W": [np.int16, ct.c_ushort],
        "M0": [np.int32, ct.c_uint],
        "M1": [np.int32, ct.c_uint]
    }



class ProfileBuffer:
    """
    Temporary profile buffer with a custom callback.
    """
    def __init__(self, resolution: int, data_width: int):
        """
        Class for storing the temporary profile buffer, with callback function.
        
        args:
            resolutin: int
            data_width: int
        """
        self.profile_buffer = (ct.c_ubyte*(resolution * data_width))()
        self.event = threading.Event()
        self.event.set()


    def __call__(self, data, size, user_data):
        if user_data == 1:
            ct.memmove(self.profile_buffer, data, size)

            self.event.set()



class Flag:
    """
    Class for storing the flag value
    """
    def __init__(self, val = True):
        self.value = val



class Buffer:
    """
    Class that encapsulates all the transmited data from the scanner.

    Available methods:
    self.clear_buffer()
    self.add_to_buffer()

    Available atributes:
    self.data -> dict that holds all the data
    """
    def __init__(self, channel_names: dict, resolution: int, 
                 sample_rate: float):
        """
        Class for storing the data from the scanner.

        Args:
            channel_names: used channel names
            resolution: the used resolution
            sample_rate: the used sample rate, used to estimate the buffer size
            profile_buffer: dict that contains the current profile data with
                numpy arrays as items and channel names as keys
        """
        self._channel_names = channel_names
        self._resolution = resolution
        # counter to track the number of transmited profiles, it is used to
        # update the buffer size
        self._profile_count = 0
        self.data = {}
        # dict za shranjevanje pointerjev do C buffer array-ev
        self._pointers = {}
        # dict, kjer se shranijo prazni array-i, v teh array-ih se shranujejo
        # novo zajeti profili
        self._profile_buffer = {}
        # define the initial buffer size for a 1s long measurement
        self._buffer_size = int(1/sample_rate)+1


    def clear_buffer(self):
        """
        Clears the buffer. It is also used to setup the buffer.

        Caution! This will delete all the measurement data.
        """
        for name, data_type in _default_channel_names.items():
            if name in self._channel_names:
                self._profile_buffer[name] = np.empty(self._resolution, 
                                                     dtype=data_type[0])
                self._pointers[name] = self._profile_buffer[name].ctypes.data_as(
                    ct.POINTER(data_type[1])
                )
                # buffer za 1s zajema
                self.data[name] = np.zeros(
                    (self._resolution, self._buffer_size), dtype=data_type[0])
                
            else:
                self._pointers[name] = ct.POINTER(data_type[1])()

        self._current_size = self._buffer_size


    def _extend_buffer(self) -> None:
        """
        Increase the buffer size for another 1s of measurement.
        """
        #TODO ne vem, ce bo to dejansko dovolj hitro...
        for name in self._channel_names:
            self.data[name] = np.vstack((
                self.data[name], np.zeros(
                    (self._resolution, self._buffer_size),
                    dtype = _default_channel_names[name][0])
            ))

        self._current_size += self._buffer_size


    def add_to_buffer(self) -> None:
        """
        Append new data to the buffer.
        """
        #TODO to naceloma dela samo za X, Z, M0, M1 podatke, za I, W, T je
        # bila tezava pri pretvarjanju na RPI-ju, glej resitev v "scan" 
        # funkciji -> resitev je, da gres na C array preko pointerja in
        # iteriras cez njega ter zapisujes podatke
        for name in self._channel_names:
            self.data[name][
                self._profile_count] += self._profile_buffer[name]

        self._profile_count += 1

        # damo si 10 profilov manj, ce se extendanje izvaja predolgo
        if self._profile_count > self._current_size - 10:
            self._extend_buffer()



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

    def __init__(self, channel_names: dict = None, exposure_time: int = 100,
                 idle_time: int = 700, resolution_id: int = 0):
        """
        Args:
            channel_names list[str]: a list of channel names, the available 
                names are as following ["X", "Z", "I", "T", "W", "M0", "M1"],
                where each channel represents a datatype transmited from the
                scanner. If None all channels are used.
            exposure_time (int): Shutter open time, defaults to 100
            idle_time (int): Shutter close time, defaults to 700
            resolution_id (int): The scanner has multiple available resolutions, 
                eg. ltt25xx has a list of posible values [640, 320, 160, 0]. 
                Resolution must be an int 0-3. Defaut value is 0.
        """

        try:
            llt
        except:
            raise Exception("Pyllt library not found. Please install it" \
                            "before using this class.")
        
        self.exposure_time = exposure_time
        self.idle_time = idle_time
        self.resolution_id = resolution_id
        # check the key values
        if channel_names is None:
            self.channel_names = list(_default_channel_names.keys())
        else:
            for key in channel_names.keys():
                if key not in _default_channel_names.keys():
                    raise ValueError(f"Invalid channel name {key}")
            self.channel_names = channel_names
        # referenca za nastavljanje skenerja -> nisem testiral ce dejansko dela
        self.llt = llt
        # hLLT je v sami C implementaciji struct s podatki o skenerju, tukaj pa
        # je dejansko zgolj int, posledicno bi lahko povsod samo pisal 1
        self.hLLT = llt.create_llt_device(llt.TInterfaceType.INTF_TYPE_ETHERNET)
        self._initialize_scanner()
        self._connect_scanner()
        self._setup_laser()
        # flag ce je vklopljen laser, naceloma je vklopljen pri priklopitvi na
        # elektriko
        self.is_on = True
        # when set to True, it stops the acq. custom thread
        # using a simple class so it gets passed by reference
        # when started the value gets set to False, must be True to start the
        # measurement
        self._termination_flag = Flag(value = True)


    def _initialize_scanner(self):
        """
        seting up interface for later connection.
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
        Connects the PC to the scanner.
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


    def profile_setup(self):
        #TODO ustrezno nastavi glede na zahtevane channele
        # pri teh nastavitvah vedno prenasamo vse podatke -> lahko se optimira
        # kolicina prenesenih podatkov
        self.start_data = 0
        self.data_width = 16

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


    def get_data(self):
        """
        Getter method to get the measurement
        """
        if self._termination_flag.value:
            return self._data_buffer.data
        
        else:
            print("Measurement has not ended yet.")


    def start_measurement(self):
        """
        Method to start the measurement.

        Important note, when a new measurement gets started a new data buffer
        is created.
        """

        if self._termination_flag.value:
            # all data gets stored in the custom buffer class
            self._data_buffer = Buffer(self.channel_names, self.resolution, 
                                    self.sample_rate)
            
            self._termination_flag.value = False
            self._data_buffer.clear_buffer()

            # ni najbolj lepo...
            self.acq_thread = threading.Thread(
                target = _read_data_loop, args = [
                    self.resolution, self.data_width,
                    self.hLLT, self._termination_flag,
                    self._partial_profile_struct, self.scanner_type,
                    self._data_buffer, self.update_event
                ]
            )
            self.acq_thread.start()

        else:
            print("A measurement seems to be running already.")


    def stop_measurement(self):
        """
        Method to stop the measurement.
        """
        self._termination_flag.value = True

        self.acq_thread.join()



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

        # self._channel_names_init = [
        #         f"{name}_{i}" for name in self._channel_names for i in range(
        #         self.device.resolution)
        #     ]

        # self._channel_names_init = ["all"]
        # self._spacer = self.device.resolution * len(self._channel_names)
        self._channel_names_video_init = self._channel_names
        self._channel_shapes_video_init = [[640,1] for i in range(len(self._channel_names))]

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
            
        # vrne False, ce se izvede timeout
        _flag = self.device.update_event.wait(self._timeout_time)

        #TODO nov problem LDAQ-a -> podatki so zlozeni v dict, kjer ima posamezen
        # array svoj tip podatka
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
        

        self.device.update_event.clear()

        return data
    
    
    def clear_buffer(self) -> None:
        """
        clear the device buffer list
        """

        self.device.clear_buffer()



def _data_generator(buffer: ProfileBuffer, hLLT: int,
                    partial_profile_struct, scanner_type: int,
                    pointers: dict):
    """
    Generator for data acquisition

    device: instance of the Scanner class
    buffer: instance of the ProfileBuffer class
    """

    while True:
        buffer.event.wait()

        fret = llt.convert_part_profile_2_values(
            hLLT, buffer.profile_buffer,
            ct.byref(partial_profile_struct), scanner_type, 0,
            1, pointers["W"], pointers["I"], 
            pointers["T"], pointers["X"], 
            pointers["Z"], pointers["M0"],
            pointers["M1"]
            )

        if fret & llt.CONVERT_X == 0 or fret & llt.CONVERT_Z == 0:
            raise ValueError("Error converting data: " + str(fret))

        buffer.event.clear()

        #TODO smiselno bi bilo se shraniti profile count oz. cas zajema profila
        # in iz tega razbrati ali dejansko zajemamo vse profile

        yield


def _read_data_loop(resolution: int, data_width: int, hLLT: int,
                    termination_flag: Flag, partial_profile_struct,
                    scanner_type: int, buffer: Buffer):
    """
    Function to run the acquisition generator in a loop
    """
    profile_buffer = ProfileBuffer(resolution, data_width)
    get_profile_cb = llt.buffer_cb_func(profile_buffer)

    ret = llt.register_callback(
        hLLT, llt.TCallbackType.C_DECL, get_profile_cb, 1
        )

    if ret < 1:
        raise Exception(f"Error setting callback: {ret}")
    
    time.sleep(0.1)
    ret = llt.transfer_profiles(hLLT, 
                                llt.TTransferProfileType.NORMAL_TRANSFER, 1)
    if ret < 1:
        raise Exception(f"Error starting transfer profiles: {ret}")

    time.sleep(0.1)

    # device.update_event = threading.Event()
    gen_fun = _data_generator(buffer, hLLT, partial_profile_struct,
                              scanner_type, buffer._pointers)

    while True:
        if termination_flag.value:
            # device._stop_transfer()
            ret = llt.transfer_profiles(hLLT, 
                    llt.TTransferProfileType.NORMAL_TRANSFER, 0)
            if ret < 1:
                raise Exception(f"Error stopping transfer profiles: {ret}")
            time.sleep(0.1)
            gen_fun.close()
            break

        next(gen_fun)

        buffer.add_to_buffer()

