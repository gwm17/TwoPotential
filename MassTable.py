#!/usr/bin/env python3


class MassTable:
	"""
	Class representing the AMDC Mass data. Takes in a file mass.txt which is a stripped
	down version of the AMDC Mass Evaluation table. You can then query the class for the mass
	of different nuclides based on A, Z. Can also give the symbol for pretty print out statements
	"""
	def __init__(self):
		file = open("data/mass.txt","r")
		self.mtable = {}
		u2mev = 931.4940954
		me = 0.000548579909 #MeV
		self.etable = {}

		file.readline()
		file.readline()

		for line in file:
			entries = line.split()
			n = entries[0]
			z = entries[1]
			a = entries[2]
			element = entries[3]
			massBig = float(entries[4])
			massSmall = float(entries[5])

			key = '('+z+','+a+')'
			value = (massBig+massSmall*1e-6)*u2mev - float(z)*me
			self.mtable[key] = value
			self.etable[key] = element
		file.close()

	def GetMass(self, z, a):
		key = '('+str(z)+','+str(a)+')'
		if key in self.mtable:
			return self.mtable[key]
		else:
			return 0

	def GetSymbol(self, z, a):
		key = '('+str(z)+','+str(a)+')'
		if key in self.etable:
			return str(a)+self.etable[key]
		else:
			return 'none'


"""
NOTE: I make a global instance of this table. Anytime this file is imported, the GLOBAL
instance of MassTable should be used, not a local instance. This will help you only parse 
the mass file once.
"""
Masses = MassTable()
