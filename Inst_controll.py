import pyvisa as visa
import os
import numpy as np
import matplotlib.pyplot as plt
import time



class Osci(object):

	def __init__(self, ip = '192.168.2.58', datatype='b'):
		'''Initializes the Osci'''
		self.OPEN_CMD = "TCPIP0::{}::INSTR"
		os.system('sudo ifconfig enp4s5 192.168.2.51')
		self.ip = ip
		self.rm = visa.ResourceManager('@py')
		self.visa_if = self.rm.open_resource(self.OPEN_CMD.format(self.ip))
		self.visa_if.timeout=10000
		self.datatype = datatype
		time.sleep(.1)

	def query(self, command):
		'''Easy way to query'''
		return self.visa_if.query(command)

	def query_binary_values(self, command, datatype, container=np.array):
		'''Easy way to query binary values'''
		return self.visa_if.query_binary_values(command, datatype=datatype, is_big_endian=False, container=container)

	def write(self, command):
		'''Easy way to write'''
		return self.visa_if.write(command)

	def read(self, command):
		'''Easy way to read'''
		return self.visa_if.read(command)

	def messung(self,Number_of_SEQuence = 500, Measurement_time = 1*10**-8,\
		samplerate =  3.125*10**10, Max_Ampl = 1, vertical_delay = 125E-9,chanel = 'CH1'):
		self.write('HOR:MODE:SAMPLERate {}'.format(samplerate))
		time.sleep(.1)
		samplerate = float(self.query('HOR:MODE:SAMPLERate?'))
		points = int(Measurement_time*samplerate)
		self.write('CH1:DESKEW {}'.format(vertical_delay))
		self.write('DISPLAY:WAVEV1:CH1:VERT:SCAL {}'.format(Max_Ampl/10))
		self.write('ACQ:SEQuence:NUMSEQuence 1')
		time.sleep(.1)
		self.write('ACQuire:MODe SAMPLE')
		self.write('DISplay:WAVEform OFF')
		self.write('HOR:MODE:RECO {}'.format(points))
		self.write('DATA:STOP {}'.format(points))
		self.write('DATa:WIDth 1')
		self.write('ACQ:STATE STOP')
		self.write('HOR:FAST:STATE ON')
		self.write('HORizontal:FASTframe:COUNt {}'.format(Number_of_SEQuence))
		self.write('DATa:SOURce {}'.format(chanel))
		self.write('CLEAR')
		self.write('ACQuire:STOPAfter SEQuence')
		self.write('ACQ:STATE ON')
		before = time.time()
		while float(self.query('ACQ:SEQuence:CURrent?').strip())<1:
			time.sleep(.1)
		self.write('ACQ:STATE STOP')
		self.write('CURVe?')
		y_values = self.visa_if.read_raw()
		print('Measurement duration: ',  time.time()-before)
		time.sleep(1)
		YOFF = float(self.query('WFMOutpre:YOFf?').strip())
		YMU = float(self.query('WFMO:YMU?').strip())
		y_values = np.reshape(np.frombuffer((y_values), dtype=np.int8),(Number_of_SEQuence,int(len(y_values)/Number_of_SEQuence)))
		self.write('DISplay:WAVEform On')
		print(len(y_values[0,:]))
		print(points)
		print(len(y_values[0,:])-points)
		return y_values[:,len(y_values[0,:])-points-1:len(y_values[0,:])-1], samplerate, Measurement_time, YOFF, YMU


	def __del__(self):
		'''Deconstructor, alwalys call at the end, otherwise there might be bugs!'''
		#self.write('CLEAR')# Somehow if active there is sometimes a timeout, I don't know why
		#time.sleep(1)
		self.visa_if.close()
		self.rm.close()
		print('Close')

class Funk_Gen(object):#no more used
    def __init__(self):
        self.rm = visa.ResourceManager('@py')
        self.DG = self.rm.open_resource('USB0::6833::1602::DG1ZA220900524::0::INSTR')

    def query(self, command):
        '''Easy way to query'''
        return self.DG.query(command)

    def write(self, command):
        '''Easy way to write'''
        return self.DG.write(command)

    def read(self, command):
        '''Easy way to read'''
        return self.DG.read(command)

    def sinus(self,freq = 2*10**3, ampl = 20*10**-3, off=0.5):
        self.write('APPLy:SINusoid {0},{1},{2}'.format(freq,ampl,off))

    def pulse(self,freq = 2*10**3, ampl = 20*10**-3, off=0.0095, width = 'MINimum'):
        self.write('APPLy:PULSe {0},{1},{2}'.format(freq,ampl,off))
        self.write('PULSe:WIDTh {}'.format(width))

    def __del__(self):
        pass

class SHR(object):
	def __init__(self, volt = 1000, ramp = 220):
		self.rm = visa.ResourceManager('@py')
		self.inst = self.rm.open_resource('ASRL/dev/ttyACM0::INSTR')
		self.volt = volt
		self.ramp = ramp

	def query(self, command):
		'''Easy way to query'''
		return self.inst.query(command)

	def write(self, command):
		'''Easy way to write'''
		return self.inst.write(command)

	def read(self, command):
		'''Easy way to read'''
		return self.inst.read(command)

	def voltage(self, volt, chanel = 0):
		self.volt = volt
		return self.write('VOLTage {0},(@{0})'.format(volt,chanel))

	def output_on(self, chanel = 0):
		self.write(':VOLTage ON,(@{0})'.format(chanel))

	def output_off(self, chanel = 0):
		self.write(':VOLTage OFF,(@{0})'.format(chanel))

	def __del__(self):
		self.write(':CONF:EVent CLEAR')