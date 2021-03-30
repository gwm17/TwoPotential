#!/usr/bin/env python3

import numpy as np 
from scipy.misc import derivative

HBARC = 197.3269804 #MeV*fm
HBAR = 6.582119569e-16*1e-6 #MeV*s
ALPHA = 1.0/137.0 #Fine structure const

class NuclearPotential :
	"""
	Class which represents a nuclear potential modeled by a Wood-Saxon potential combined with
	a Coulomb potential, as well as the centrifugal barrier. Holds all relevant parameters. Contains methods 
	for determining the location of the min and the max of the potential as well as generating data for making plots.
	NOTE: Potentials are natively handled in units of 1/fm^2. Class contains methods to convert back into MeV.
	Native units of MeV, fm, s
	"""
	def __init__(self, A, Z1, Z2, V0, VS, a0, aS, R0, RS, m, l, j, rB=0, VB=0) :
		self.A = A
		self.Z1 = Z1
		self.Z2 = Z2
		self.V0 = V0 #units of MeV
		self.VS = VS #units of MeV
		self.a0 = a0
		self.aS = aS
		self.R0 = R0 #units of fm
		self.RS = RS #units of fm
		self.m = m
		self.l = l
		self.j = j
		self.rB = rB
		self.VB = VB #units of 1/fm^2

	def f0(self, r) :
		return 1.0/(1.0 + np.exp((r-self.R0)/self.a0))

	def fS(self, r) :
		return 1.0/(1.0 + np.exp((r-self.RS)/self.aS))

	def WoodSaxson(self, r) :
		spin_orbit = 0
		if self.j == self.l+0.5 :
			spin_orbit = self.l
		else :
			spin_orbit = -(self.l + 1)

		return -2.0*self.m/(HBARC**2.0)*(self.V0*self.f0(r) + self.VS*spin_orbit*derivative(self.fS, r)*2.0/r)

	def Coulomb(self, r) :
		if r <= self.R0 :
			return ALPHA*2.0*self.m*self.Z1*self.Z2/(2.0*HBARC*self.R0)*(3.0 - r**2.0/(self.R0**2.0))
		else :
			return ALPHA*2.0*self.m*self.Z1*self.Z2/(HBARC*r)

	def Centripital(self, r) :
		if self.l == 0 :
			return 0.0
		elif r == 0:
			return (self.l**2.0 + self.l)/((r+1e-14)**2.0)
		else :
			return (self.l**2.0 + self.l)/(r**2.0)

	def PotentialV(self, r) :
		return (self.WoodSaxson(r) + self.Coulomb(r) + self.Centripital(r))

	def PotentialV_MeV(self, r) :
		return HBARC**2.0/(2.0*self.m)*(self.WoodSaxson(r) + self.Coulomb(r) + self.Centripital(r))

	def PotentialU(self, r):
		if r<self.rB :
			return self.PotentialV(r)
		else :
			return self.VB

	def PotentialWTilde(self, r) :
		if r<self.rB :
			return self.VB
		else :
			return self.PotentialV(r)

	def FindMaximumHeight(self, steps, rmax) :
		R = self.R0/2.0
		r_range = np.linspace(R*1.5, rmax, steps)
		curMax = 0.0
		curV = 0.0
		rMax = 0.0
		for r in r_range :
			curV = self.PotentialV(r)
			if curV > curMax :
				curMax = curV
				rMax = r
		return rMax, curMax

	def FindMinimumHeight(self, steps, rmax) :
		r_range = np.linspace(1e-14, rmax, steps)
		curMin = 1e20
		curV = 0.0
		rMin = 0.0
		for r in r_range :
			curV = self.PotentialV(r)
			if curV < curMin :
				curMin = curV
				rMin = r
		return rMin, curMin

	def Convert2MeV(self, V):
		return HBARC**2.0/(2.0*self.m)*V

	def GeneratePotential(self, r_range, Potential) :
		V_values = np.zeros(r_range.size)
		for i in np.arange(0, r_range.size):
			V_values[i] = HBARC**2.0/(2.0*self.m)*Potential(r_range[i])
		return V_values