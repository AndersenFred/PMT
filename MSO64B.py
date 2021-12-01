import pyvisa as visa
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import h5py



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

	@staticmethod
	def save_to_file(filepath, data, samplerate, Measurement_time, y_off, YMULT):
		with h5py.File('{}.h5'.format(filepath),'w') as file:
			file.attrs[u'samplerate'] = samplerate
			file.attrs[u'measurement_time'] = Measurement_time
			file.attrs[u'y_off'] = y_off
			file.attrs[u'YMULT'] = YMULT
			file.create_dataset('waveforms', data = data)

	def messung(self,Number_of_SEQuence = 500, Measurement_time = 3*10**-7, samplerate = 10**7, Max_Ampl = 1, vertical_delay = -23E-9):
		'''Returns the Waveform shown on the Display time in s and Voltage in V '''
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
		self.write('ACQ:STOPAfter SEQ')
		self.write('CLEAR')
		self.write('ACQ:STATE 1')
		before = time.time()
		while float(self.query('ACQ:NUMAC?').strip())<1:
			time.sleep(.1)
			#print('y')
		time.sleep(1)
		y_values=self.query_binary_values('CURV?', datatype = self.datatype)
		duration=time.time()-before
		print('Measurement duration: ',  duration)
		#XDIV = float(self.query('HORizontal:MODE:SCA?'))
		time.sleep(1)
		YOFF = float(self.query('WFMOutpre:YOFf?').strip())
		YMU = float(self.query('WFMO:YMU?').strip())
		#x_values = np.linspace((-5)*XDIV,XDIV*5,len(y_values))*10**6
		return y_values, samplerate, Measurement_time, YOFF, YMU

	def plot(self, Name = 'Messung.npy'):
		fig, ax = plt.subplots(figsize=(10,5))
		x, y = self.messung()
		plt.xlabel(r"time in $\mu$s")
		plt.ylabel("Amplitude in V")
		ax.plot(x,y, label = 'Chanel 1')
		plt.show()
		#np.save(Name,(np.array([x,y]).T))
		#plt.savefig('{}.pdf'.format(Name))

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
	#osci.write('DISPLAY:WAVEV1:CH1:VERT:SCAL 0.1')
	#osci.plot()
	time.sleep(2)
	del osci
