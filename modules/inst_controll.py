import pyvisa as visa
import os
import numpy as np
import time
import halo


class Osci(object):
    def __init__(self, ip='10.202.33.47', datatype='b'):
        '''Initializes the Osci'''
        self.OPEN_CMD = f"TCPIP0::{ip}::INSTR"
        self.ip = ip  # ip of the Osci
        self.rm = visa.ResourceManager()
        self.visa_if = self.rm.open_resource(self.OPEN_CMD)
        self.visa_if.timeout = 1000000  # set a high timeout, nessesary!
        self.datatype = datatype  # no more in use
        time.sleep(.1)

    def query(self, command):
        '''Easy way to query'''
        return self.visa_if.query(command)

    def query_binary_values(self, command, datatype, container=np.array):
        '''Easy way to query binary values, no more in use'''
        return self.visa_if.query_binary_values(command, datatype=datatype, is_big_endian=False, container=container)

    def write(self, command):
        '''Easy way to write'''
        return self.visa_if.write(command)

    def read(self, command):
        '''Easy way to read'''
        return self.visa_if.read(command)

    def messung(self, Number_of_SEQuence=100, Measurement_time=75e-9, \
                samplerate=6.25 * 10 ** 12, chanels=['CH1', 'CH2', 'CH3', 'CH4']):
        '''Makes the nessesary configuration for measuring, and returns an 2D array of rawdata, and several parameters'''
        self.write('HOR:MODE:SAMPLERate {}'.format(
            samplerate))  # write the samplerate, problem is osci only uses specific values
        time.sleep(.1)  # sleep needed, otherwise sometimes one get the old samplerate
        samplerate = float(self.query('HOR:MODE:SAMPLERate?'))  # to get correct samplerate
        points = int(Measurement_time * samplerate)  # calculate the number of points
        # self.write('CH1:DESKEW {}'.format(vertical_delay))
        self.write('ACQ:SEQuence:NUMSEQuence 1')  # Sequenes do not work, therefore use numver of sequence = 1
        self.write('ACQuire:MODe SAMPLE')
        self.write('DISplay:WAVEform OFF')  # for faster Measurement
        self.write('HOR:MODE:RECO {}'.format(points))
        self.write('DATA:STOP {}'.format(points))
        self.write('DATa:WIDth 1')  # number of byte per point
        self.write('ACQ:STATE STOP')
        self.write('HOR:FAST:STATE ON')  # FastFrame , Osci uses this mode instead of Sequences
        self.write('HORizontal:FASTframe:COUNt {}'.format(Number_of_SEQuence))
        Number_of_SEQuence = int(self.query("HORizontal:FASTframe:COUNt?"))
        chanel = ''
        for ch in chanels:
            chanel += ch + ','
        chanel = chanel[:-1]
        self.write('DATa:SOURce {}'.format(chanel))
        self.write('CLEAR')  # Delete old points
        self.write('ACQuire:STOPAfter SEQuence')
        self.write('ACQ:STATE ON')
        spinner = halo.Halo(text='Recording waveforms...', spinner='dots')  # just for fun
        spinner.start()  # start the loading symbold
        before = time.time()
        while float(self.query('ACQ:SEQuence:CURrent?').strip()) < 1:  # waitung untll measurement is finished
            time.sleep(Number_of_SEQuence / 10000)
        self.write('ACQ:STATE STOP')  # stop measure
        spinner.stop()
        spinner = halo.Halo(text='Transfering data', spinner='dots')  # just for fun
        spinner.start()  # start the loading symbold
        self.write('CURVe?')  # preperation to get the values
        y_values = self.visa_if.read_raw()  # get raw data
        spinner.stop()  # stop the loading symbol
        print('Measurement duration: ', time.time() - before)
        time.sleep(1)
        YMU = []
        YOFF = []
        for ch in chanels:
            self.write('DATa:SOURce {}'.format(ch))
            YMU.append(float(self.query('WFMO:YMU?').strip()))  # get the propotionality factor
            YOFF.append(float(self.query('WFMOutpre:YZEro?').strip()))  # get the offset
        y_values = np.reshape(np.frombuffer((y_values), dtype=np.int8), (len(chanels), Number_of_SEQuence,
                                                                         int(len(y_values) / Number_of_SEQuence / len(
                                                                             chanels))))  # reshape and convert binary values to usable values
        self.write('DISplay:WAVEform On')
        return y_values[:, :,
               len(y_values[0, 0, :]) - points - 1:len(y_values[0, 0, :]) - 1], Measurement_time, YOFF, YMU, samplerate

# def __del__(self):#Does not work at the moment
# '''Deconstructor, alwalys call at the end, otherwise there might be bugs!'''
# self.write('CLEAR')# Somehow if active there is sometimes a timeout, I don't know why
# time.sleep(1)
# self.visa_if.close()
# self.rm.close()
# print('Close')


class Funk_Gen(object):  # no more used
    def __init__(self, ip: str = '10.202.33.99'):
        self.rm = visa.ResourceManager()
        self.DG = self.rm.open_resource("TCPIP0::{}::INSTR".format(ip))

    def query(self, command):
        '''Easy way to query'''
        return self.DG.query(command)

    def write(self, command):
        '''Easy way to write'''
        return self.DG.write(command)

    def read(self, command):
        '''Easy way to read'''
        return self.DG.read(command)

    def on(self, channel=1):
        if channel == 1:
            self.write('OUTP ON')
        elif channel == 2:
            self.write('OUTP:CH2 ON')

    def off(self, channel=1):
        if channel == 1:
            self.write('OUTP OFF')
        elif channel == 2:
            self.write('OUTP:CH2 OFF')

    def sinus(self, freq=2e3, ampl=2e-3, off=0.5, channel=1):
        if (channel == 1):
            self.write('APPLy:SINusoid {0},{1},{2}'.format(freq, ampl, off))
        elif channel == 2:
            self.write('APPLy:SINusoid:CH2 {0},{1},{2}'.format(freq, ampl, off))
        else:
            raise ArributeError('chanel has to be either 1 or 2')

    def pulse(self, freq=2e4, ampl=20e-3, off=0.0095, width='MINimum', channel=1):
        if (channel == 1):
            self.write('APPLy:PULSe {0},{1},{2}'.format(freq, ampl, off))
            self.write('PULSe:WIDTh {}'.format(width))
        elif (channel == 2):
            self.write('APPLy:PULSe:CH2 {0},{1},{2}'.format(freq, ampl, off))
            self.write('PULSe:WIDTh:CH2 {}'.format(width))
        else:
            raise ArributeError('chanel has to be either 1 or 2')


class SHR(object):
    '''High voltage source iseg SHR '''

    def __init__(self, volt=1000, chanel=0, ramp=320):
        self.chanel = chanel
        self.rm = visa.ResourceManager()
        self.inst = self.rm.open_resource('ASRL/dev/ttyACM0::INSTR')
        self.volt = volt
        self.ramp = ramp
        self.voltage(volt=self.volt, chanel=self.chanel)

    def query(self, command):
        '''Easy way to query'''
        return self.inst.query(command)

    def write(self, command):
        '''Easy way to write'''
        return self.inst.write(command)

    def read(self, command):
        '''Easy way to read'''
        return self.inst.read(command)

    def voltage(self, volt, chanel=0):
        self.volt = volt
        return self.write(':VOLTage {0},(@{1})'.format(self.volt, chanel))

    def output_on(self, chanel=0):
        self.write(':VOLTage ON,(@{0})'.format(chanel))

    def output_off(self, chanel=0):
        self.write(':VOLTage OFF,(@{0})'.format(chanel))

# def __del__(self):
#	self.write(':EVENT CLEAR,(@0)')
