import pyvisa as visa
import os
import numpy as np
#Mal testen mit python 2
import matplotlib.pyplot as plt
import time
OPEN_CMD = "TCPIP0::{}::INSTR"


class Osci(object):

	def __init__(self, ip, points = 125000000/2, datatype='h'):
		'''Initializes the Osci'''
		self.ip = ip
		self.rm = visa.ResourceManager('@py')
		self.visa_if = self.rm.open_resource(OPEN_CMD.format(self.ip))
		self.visa_if.timeout=100
		self.points= int(points)
		self.datatype = datatype

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

	def messung(self,Number_of_SEQuence = 1000,AVERAGe_Number = 100):
		'''Returns the Waveform shown on the Display time in s and Voltage in  '''
		self.write('ACQuire:MODe AVERAGe;:ACQuire:NUMAVg {}'.format(AVERAGe_Number))
		self.write('DAT:STOP {}'.format(self.points))
		self.write('ACQ:STATE STOP')
		self.write('ACQ:SEQuence:NUMSEQuence {}'.format(Number_of_SEQuence))
		self.write('CLEAR')
		self.write('ACQ:STATE 1')
		#print(self.query('ACQuire:NUMAVg?'))
		time.sleep(1)
		before = time.time()
		y_values=self.query_binary_values('CURV?', datatype=self.datatype,container=np.array)
		duration=time.time()-before
		print('Measurement duration: ',  duration)
		XDIV = float(self.query('HORizontal:MODE:SCA?'))
		#print(XDIV)
		YMU = float(self.query('WFMO:YMU?').strip())
		x_values = np.linspace((-5)*XDIV,XDIV*5,len(y_values))
		return (x_values, y_values*YMU)

	def plot(self, Name = 'Messung'):
		fig, ax = plt.subplots(figsize=(10,5))
		x, y = self.messung()
		ax.plot(x,y)
		plt.show()
		#plt.savefig('{}.pdf'.format(Name))

if __name__=='__main__':
	os.system('sudo ifconfig enp4s5 192.168.2.51')
	osci =  Osci('192.168.2.58')
	#osci.write('ACQuire:MODe AVERAGe;:ACQuire:NUMAVg 100')
	#time.sleep(2)
	osci.plot()
