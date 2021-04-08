#!/bin/usr/env python3

class Config :
	def __init__(self, params) :
		self.A = int(params[0])
		self.Z = int(params[1])
		self.V0 = float(params[2])
		self.VS = float(params[3])
		self.a0 = float(params[4])
		self.aS = float(params[5])
		self.R0 = float(params[6])
		self.RS = float(params[7])
		self.l = int(params[8])
		self.j = float(params[9])
		self.Eb = float(params[10])

class ConfigFile : 
	def __init__(self, filename) :
		self.filename = filename
		self.outputfile = None
		self.size = 0
		self.validFlag = False
		self.configs = []

		self.ProcessFile()

	def __del__(self) :
		self.outputfile.close()

	def ProcessFile(self) :
		file = open(self.filename, "r")
		assert file.closed == False

		initial = file.readline()
		values = initial.split()
		self.outputfile = open(values[1], "w")
		assert self.outputfile.closed == False
		header = "{:12}".format("A")+"{:12}".format("Z")+"{:12}".format("Qp(Mev)")+"{:12}".format("L")+"{:12}".format("J")+"{:12}".format("Type")+"{:12}".format("Width(MeV)")+"{:12}".format("t1/2(s)")+"\n"
		self.outputfile.write(header)
		file.readline()
		for line in file:
			entries = line.split()
			self.configs.append(Config(entries))
			self.size += 1
		file.close()
		self.validFlag = True

	def IsValid(self) :
		return self.validFlag

	def GetConfiguration(self, index) :
		assert self.validFlag == True
		assert index < self.size

		return self.configs[index]

	def WriteResults(self, config, width, halflife, kind) :
		assert self.validFlag == True

		line = "{:<12d}".format(config.A)+"{:<12d}".format(config.Z)+"{:<11.4f}".format(config.Eb)+" "+"{:<12d}".format(config.l)+"{:<11.1f}".format(config.j)+" "+"{:12}".format(kind)+"{:<11.5e}".format(width)+" "+"{:<11.5e}".format(halflife)+"\n"

		self.outputfile.write(line)   
