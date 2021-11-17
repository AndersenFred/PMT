#Works for MSO64B deveoped for PMT measurements
import pyvisa as visa
import os
import numpy as np
import matplotlib.pyplot as plt
import time
OPEN_CMD = "TCPIP0::{}::INSTR"


class Osci(object):

	def __init__(self, ip, points = 12500000/2, datatype='h'):
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
	

	def messung(self,Number_of_SEQuence = 1,AVERAGe_Number = 100):
		'''Returns the Waveform shown on the Display time in s and Voltage in  '''
		#self.write('ACQuire:MODe AVERAGe;:ACQuire:NUMAVg {}'.format(AVERAGe_Number))
		self.write('DAT:STOP {}'.format(self.points))
		self.write('HOR:MODE:RECO {}'.format(self.points))
		self.write('DATa:WIDth 2')
		self.write('ACQ:STATE STOP')
		self.write('ACQ:SEQuence:NUMSEQuence {}'.format(Number_of_SEQuence))
		self.write('ACQ:STOPAfter SEQ')
		print(self.query('ACQ:STOPA?'))
		self.write('CLEAR')
		self.write('ACQ:STATE 1')
		while float(self.query('ACQ:NUMAC?').strip())<1:
			time.sleep(.1)
		print(self.query('ACQuire:NUMAVg?'))
		time.sleep(1)
		before = time.time()
		y_values=self.query_binary_values('CURV?', datatype=self.datatype,container=np.array)
		duration=time.time()-before
		print('Measurement duration: ',  duration)
		XDIV = float(self.query('HORizontal:MODE:SCA?'))
		#print(XDIV)

		YOFF = float(self.query('WFMOutpre:YOFf?'))
		YMU = float(self.query('WFMO:YMU?').strip())
		print(YMU)
		print(YOFF)
		x_values = np.linspace((-5)*XDIV,XDIV*5,len(y_values))
		print((y_values-YOFF)*YMU)
		return (x_values, (y_values-YOFF)*YMU)

	def plot(self, Name = 'Messung.npy'):
		#fig, ax = plt.subplots(figsize=(10,5))
		x, y = self.messung()
		np.save(Name,(np.array([x,y]).T))
		#plt.savefig('{}.pdf'.format(Name))

	def __del__(self):
		'''Deconstructor. alwalys call at the end, otherwise there might be bugs!'''
		time.sleep(.1)
		self.write('CLEAR')
		self.visa_if.close()
		self.rm.close()
		print('Close')
		
	@staticmethod
	def save_waveforms_to_file(filepath, data_array, hor_interval, vert_gain, comment=None):
		with h5py.File(filepath, "w") as file:
			file.attrs[u'vertical_gain'] = vert_gain
			file.attrs[u'horizontal_interval'] = hor_interval
			if comment is not None:
				file.attrs[u'comment'] = comment
			file.create_dataset('waveforms', data=data_array)
		
	@staticmethod	
	def read_waveforms_from_file(filepath):
		retval = dict()
		with h5py.File(filepath, "r") as file:
			retval['vertical_gain'] = float(file.attrs[u'vertical_gain'])
			retval['horizontal_interval'] = float(file.attrs[u'horizontal_interval'])
			retval['comment'] = str(file.attrs[u'comment'])
			retval['data'] = np.asarray(file['waveforms'])
		return retval
	
if __name__=='__main__':
	os.system('sudo ifconfig enp4s5 192.168.2.51')
	osci =  Osci('192.168.2.58')
	#osci.write('ACQuire:MODe AVERAGe;:ACQuire:NUMAVg 100')
	#time.sleep(2)
	osci.write('DISPLAY:WAVEV1:CH1:VERT:SCAL 0.1')
	time.sleep(2)
	#osci.plot()
	del osci
