#Works for Rigol DG1022 developed for PMT measurements
#Is now implemented in Inst_control
import pyvisa as visa
import numpy as np
import time

class Funk_Gen(object):
    def __init__(self):
        self.rm = visa.ResourceManager('@py')
        self.DG = self.rm.open_resource('USB0::6833::1602::DG1ZA220900524::0::INSTR')

    def query(self, command):
        '''Easy way to query'''
	try:
        	return self.DG.query(command)
	except Exception:
		print('Timeout Error: The command {} does probably  not exist'.format(command))
		return

    def write(self, command):
        '''Easy way to write'''
	try:
        	return self.DG.write(command)
	except Exception:
		print('Timeout Error: The command {} does probably  not exist'.format(command))
		return
    def read(self, command):
        '''Easy way to read'''
	try:
        	return self.DG.read(command)
	except Exception:
		pprint('Timeout Error: The command {} does probably  not exist'.format(command))
		return
	
    def sinus(self, freq=1000, ampl=1, off=0):
        self.write('APPLy:SINusoid {0},{1},{2}'.format(freq,ampl,off))

    def pulse(self, freq=1000, ampl=1, off=0, width = 'MINimum'):
	try:
		freq = float(freq)
		ampl = float(ampl)
		off = float(off)
	except ValueError:	
		print('not a Valid input')
	if width == "MINimum" or width == "MAXimum" or width == "MAX" or width == "MIN"":	
        	self.write('APPLy:PULSe {0},{1},{2}'.format(freq,ampl,off))
        	self.write('PULSe:WIDTh {}'.format(width))
		return
	else:
		try:
			width = float(width)
		except ValueError:
			print('{} is not a valid input'.format(width))
    #def __del__(self):
        #time.sleep(1)
        #self.write('SYSTem:LOCal')

if __name__=='__main__':
	Gen =  Funk_Gen()
	Gen.pulse()
	#del Gen

