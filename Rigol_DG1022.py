#Works for Rigol DG1022 developed for PMT measurements
import pyvisa as visa
import numpy as np
import time

class Funk_Gen(object):
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

    def sinus(self, freq=1000, ampl=1, off=0):
        self.write('APPLy:SINusoid {0},{1},{2}'.format(freq,ampl,off))

    def pulse(self, freq=1000, ampl=1, off=0, width = 'MINimum'):
        self.write('APPLy:PULSe {0},{1},{2}'.format(freq,ampl,off))
        self.write('PULSe:WIDTh {}'.format(width))
    #def __del__(self):
        #time.sleep(1)
        #self.write('SYSTem:LOCal')

if __name__=='__main__':
	Gen =  Funk_Gen()
	Gen.pulse()
	#del Gen

