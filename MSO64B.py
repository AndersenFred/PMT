import pyvisa as visa
import os
import numpy as np
import matplotlib.pyplot as plt
import time



class Osci(object):

	def __init__(self, ip, datatype='b'):
		'''Initializes the Osci'''
		self.OPEN_CMD = "TCPIP0::{}::INSTR"
		os.system('sudo ifconfig enp4s5 192.168.2.51')
		self.ip = ip
		self.rm = visa.ResourceManager('@py')
		self.visa_if = self.rm.open_resource(self.OPEN_CMD.format(self.ip))
		self.visa_if.timeout=1000
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

	def messung(self,Number_of_SEQuence = 500, Measurement_time = 3*10**-7, samplerate =  3.125*10**10, Max_Ampl = 1, vertical_delay = -23E-9):
		points = int(Measurement_time*samplerate)
		self.write('CH1:DESKEW {}'.format(vertical_delay))
		self.write('DISPLAY:WAVEV1:CH1:VERT:SCAL {}'.format(Max_Ampl/10))
		time.sleep(.1)
		self.write('ACQuire:MODe SAMPLE')
		self.write('DAT:STOP {}'.format(points))
		self.write('HOR:MODE:SAMPLERate {}'.format(samplerate))
		self.write('HOR:MODE:RECO {}'.format(points))
		self.write('DATa:WIDth 1')
		self.write('ACQ:STATE STOP')
		self.write('ACQ:SEQuence:NUMSEQuence {}'.format(Number_of_SEQuence))
		self.write('CLEAR')
		self.write('ACQuire:STOPAfter SEQuence')
		print(self.query('ACQuire:STOPAfter?'))
		self.write('ACQ:STATE ON')
		before = time.time()
		while float(self.query('ACQ:SEQuence:CURrent?').strip())<Number_of_SEQuence:
			time.sleep(.1)
		self.write('ACQ:STATE STOP')
		print(self.query('ACQ?'))
		y_values=self.query_binary_values('CURV?', datatype = self.datatype)

		duration=time.time()-before
		print('Measurement duration: ',  duration)
		time.sleep(1)
		YOFF = float(self.query('WFMOutpre:YOFf?').strip())
		YMU = float(self.query('WFMO:YMU?').strip())
		return y_values, samplerate, Measurement_time, YOFF, YMU


	def __del__(self):
		'''Deconstructor, alwalys call at the end, otherwise there might be bugs!'''
		self.write('CLEAR')
		time.sleep(1)
		self.visa_if.close()
		self.rm.close()
		print('Close')

if __name__=='__main__':
	osci =  Osci('192.168.2.58')
	#osci.write('ACQuire:MODe AVERAGe;:ACQuire:NUMAVg 100')
	#time.sleep(2)
	#osci.write('CH1:DESK 1')
	time.sleep(2)
	del osci
