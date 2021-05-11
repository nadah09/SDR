import scipy
import scipy.signal
import matplotlib.pyplot as plt
from math import e, sin, pi, cos
import math
import dtsp
import numpy as np
from lib6003.audio import wav_read, wav_write

class AudioExtractor:
	def __init__(self, fc, fs, hardwareFreq, fileName):
		self.fc = fc
		self.fs = fs
		self.fsnew = 256000 #Hz
		self.fh = hardwareFreq
		self.fileName = fileName
		self.signal = self.loadData()
		self.M = int(self.fs/self.fsnew)

	def process(self):
		"""
		Does entire filtering process to extract mono audio signal
		"""
		self.modulateToBase()
		self.downsample()
		self.FMDemodulation()

	def loadData(self):
		"""
		Load raw data from file
		"""
		signal = scipy.fromfile(open(self.fileName + '.raw'), dtype=scipy.complex64)
		signal = np.array(signal)
		return signal

	def modulateToBase(self):
		"""
		Part 2a: Modulate to baseband
		"""
		errorFreq = self.fc - self.fh
		w0 = errorFreq*2*pi/self.fs
		basebandShift = np.array([e**(-1j*w0*i) for i in range(len(self.signal))])  #shift to modulate down
		shiftedToBase = self.signal*basebandShift #multiply signal with shift
		self.signal = shiftedToBase

	def downsample(self):
		"""
		Part 2b: Downsample by fs/fsNew
		"""
		#LPF Parameters
		transitionBandWidth = 0.06
		wp = 1/self.M
		ws = wp+transitionBandWidth
		dpass = 0.0575
		dstop = 0.0033

		LPF = self.createLPF(wp, ws, dpass, dstop)
		self.convolve(LPF)
		self.plotFFT("Signal after Anti-Aliasing for M = 8")
		self.decimate(self.M)

	def createLPF(self, wp, ws, dpass, dstop):
		"""
		Returns low pass filter given the wp, ws, dpass, and dstop parameters
		"""
		numtaps, bands, amps, weights = dtsp.remezord([wp/2.0, ws/2.0], [1, 0], [dpass,dstop], Hz=1.0)
		bands *= 2.0    # above function outputs frequencies normalized from 0.0 to 0.5
		b = scipy.signal.remez(numtaps, bands, amps, weights, Hz=2.0)
		return b

	def decimate(self, M):
		"""
		Returns a signal of every Mth sample of the signal
		"""
		self.signal = np.array([self.signal[i] for i in range(0, len(self.signal), M)])
		self.N = len(self.signal)

	def convolve(self, filterList):
		"""
		Filters signal using convolution
		"""
		self.signal = scipy.signal.convolve(self.signal, filterList)

	def FMDemodulation(self):
		"""
		Part 3
		"""
		self.frequencyDiscriminator()
		self.deemphasisFilter()

		#LPF Parameters
		CT_PB = 15000 #Hz
		CT_SB = 18000 #Hz
		DT_PB = CT_PB/self.fsnew*2
		DT_SB = CT_SB/self.fsnew*2
		dpass = 0.0575
		dstop = 0.0033

		LPF = self.createLPF(DT_PB, DT_SB, dpass, dstop)
		self.convolve(LPF)
		self.decimate(4) #Final decimation to 64 kHz

	def frequencyDiscriminator(self):
		self.limiter()
		self.DTDifferentiator()
		self.toImag()

	def limiter(self):
		"""
		Normalizes all signal sample magnitudes to either 1 or -1
		"""
		self.signal = np.array([self.signal[i]/abs(self.signal[i]) for i in range(len(self.signal))])

	def DTDifferentiator(self):
		"""
		Returns limited signal after DT differentiation multiplied by a 
		shifted conjugated version of the limited signal
		"""
		M_filter = 15
		diff = self.generateDifferentiator(M_filter)
		shiftedConj = self.shiftedConj(M_filter)
		shiftedConj = np.concatenate((shiftedConj, np.array([0 for i in range(M_filter)]))) #extend shifted version with zeros for convolution
		self.convolve(diff)
		self.signal = self.signal*shiftedConj

	def generateDifferentiator(self, M_filter): 
		"""
		Generates DT differentiator filter of length M_filter and windowed 
		with a Kaiser window having alpha = M_filter+1 and beta = 2.4
		"""
		beta = 2.4 
		hdiff_truncated = np.array([cos(pi*(n-M_filter/2))/(n-M_filter/2) - sin(pi*(n-M_filter/2))/(pi*(n-M_filter/2)**2) if n != M_filter/2 else 0 for n in range(M_filter+1)])
		kaiser = scipy.signal.kaiser(M_filter+1, beta)
		windowed = hdiff_truncated*kaiser
		return windowed

	def shiftedConj(self, M_filter):
		"""
		Creates shifted conjugate of signal by expanding the signal by 2, 
		interpolating, shifting by M, and downsampling and taking the conjugate 
		which results in a shift by M/2
		"""
		L = 2
		signalToExpand = self.signal.copy()
		signalExpanded = self.expand(signalToExpand, L)
		for i in range(1, len(signalExpanded)-1):
			if i%2 == 1:
				signalExpanded[i] = 1/2*(signalExpanded[i-1]+signalExpanded[i+1])
		shiftedConj = np.array([0 for i in range(int(M_filter))] + [signalExpanded[i-int(M_filter)].conjugate() for i in range(int(M_filter), len(signalExpanded))])
		return np.array([shiftedConj[i] for i in range(0, len(shiftedConj), L)])

	def expand(self, filterList, L):
		"""
		Returns signal expanded by a factor of L with zeros between samples
		"""
		expanded = []
		for i in filterList:
			expanded.append(i)
			expanded.append(0)
		return expanded

	def toImag(self):
		"""
		Takes imaginary part of signal
		"""
		self.signal = np.array([self.signal[i].imag for i in range(len(self.signal))])

	def deemphasisFilter(self):
		"""
		Returns signal after deemphasis filter
		"""
		tau = 7.5e-5 #seconds
		num = [1] #H = 1/(1+s*tau)
		den = [tau, 1]

		filtz = scipy.signal.dlti(*scipy.signal.bilinear(num, den, self.fsnew))

		a = filtz.num[0]
		b = filtz.num[1]
		c = filtz.den[0]
		d = filtz.den[1]
		result = [a*self.signal[0]/c] #start new signal with a/c*signal[0]
		for n in range(1, len(self.signal)):
			result.append(1/c*(a*self.signal[n] + b*self.signal[n-1] - d*result[n-1]))
		self.signal = np.array(result)

	def plotFFT(self, title):
		"""
		Plots magnitude response of signal
		"""
		w, H = scipy.signal.freqz(self.signal)
		plt.plot(w, np.abs(H))
		plt.title(title)
		plt.xlabel("Radian Frequency ($\omega$)")
		plt.ylabel("Amplitude")
		plt.show()

#Variables
fs = 2048000 #Hz
fc = 8.89e7 #Hz
hardwareFreq = 88810400 #Hz 88.810400 MHz
fileName = "gqrx_20201022_225449_88810400_2048000_fc"
fsNew = 256000 #Hz

#Variables
fs = 2048000 #Hz
fc = 1.007e8 #Hz
hardwareFreq = 1.003e+8 #Hz 88.810400 MHz
fileName = "data1"
fsNew = 256000 #Hz

#Variables
fs = 2048000 #Hz
fc = 9.45e7 #Hz
hardwareFreq = 9.47e7 #Hz 88.810400 MHz
fileName = "data2"
fsNew = 256000 #Hz



test = AudioExtractor(fc, fs, hardwareFreq, fileName)
#test.deemphasisFilter()
#original, fsOriginal = wav_read('gqrx_20201022_225450_88900000.wav')

test.process()

#test.plotFFT()
wav_write(np.array(test.signal)/12, 64000, "data2result.wav")







