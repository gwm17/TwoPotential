#!/usr/bin/env python3

import numpy as np
from scipy import integrate
from scipy import optimize
from scipy.misc import derivative
from MassTable import Masses
from NuclearPotential import *

def ReducedMass(m1, m2) :
	return m1*m2/(m1+m2)

class WKBPotential:

	def __init__(self, A, Z1, Z2, V0, VS, a0, aS, R0, RS, m, l, j, Eb) :
		self.Eb = Eb
		self.m = m
		self.Potential = NuclearPotential(A, Z1, Z2, V0, VS, a0, aS, R0, RS, m, l, j)
		self.r0 = 0.0
		self.r1 = 0.0
		self.r2 = 0.0
		self.FindTurningPoints()

	def CalculateK(self, r) :
		#return np.sqrt(2.0*self.m*abs(self.Eb - self.PotentialV(r)))/HBARC
		return np.sqrt(2.0*self.m*abs(self.Eb - self.Potential.PotentialV_MeV(r)))/HBARC

	def RootEquation(self, r) :
		#return self.Eb - self.PotentialV(r)
		return self.Eb - self.Potential.PotentialV_MeV(r)

	def FindTurningPoints(self) :
		rBarrier, VBarrier = self.Potential.FindMaximumHeight(15000, self.Potential.R0*3.0)
		VBarrier = self.Potential.Convert2MeV(VBarrier)
		print("rBarrier: ", rBarrier, " VBarrier: ",VBarrier)
		rWell, VWell = self.Potential.FindMinimumHeight(1500, self.Potential.R0*1.5)
		VWell = self.Potential.Convert2MeV(VWell)
		print("rWell: ", rWell, " VWell: ",VWell)
		if self.Potential.l!=0 :
			self.r0 = optimize.brentq(self.RootEquation, 1e-14, rWell)
		self.r1 = optimize.brentq(self.RootEquation, rWell, rBarrier)
		self.r2 = optimize.brentq(self.RootEquation, rBarrier, 1.0*self.Potential.R0**3.0)
		print("r0: ", self.r0, " r1: ", self.r1, " r2: ",self.r2)


	def CalculatePrefactor(self, r) :
		value, err = integrate.quad(self.CalculateK, self.r0, r)
		return 1.0/self.CalculateK(r)*(np.cos(value - np.pi/4.0))**2.0

	def CalculateDecayWidth(self) :
		exponent, err_ex = integrate.quad(self.CalculateK, self.r1, self.r2)
		print("exponent: ",exponent," error: ",err_ex)
		exponent *= -2.0
		prefactor, err_pre = integrate.quad(self.CalculatePrefactor, self.r0, self.r1)
		print("prefactor: ",prefactor," error: ",err_pre)
		prefactor = (prefactor)**(-1.0)
		gamma = prefactor*HBARC**2.0/(4.0*self.m)*np.exp(exponent)
		halflife = HBAR/gamma*np.log(2)
		return gamma, halflife

def BecchettiDepth(A, Z, E) :
	return 54.0 - 0.32*E + 0.4*Z/A**(1./3.)+24*(A - 2.0*Z)/A

def main():
	print("--------------------------------")
	print("------GWM & JCE WKB Solver------")
	print("--------------------------------")

	#"""147Tm Parameters
	rWS = 1.17
	rSO = 1.01
	Ap = 1
	Zp = 1
	AT = 146
	ZT = 68
	Eb = 1.132
	V0 = BecchettiDepth(AT, ZT, Eb)
	VS = -6.2
	a0 = 0.75
	aS = 0.75
	R0 = rWS*AT**(1./3.)
	RS = rSO*AT**(1./3.)
	l_mom = 2
	j_mom = 2.5
	#"""
	"""13N Parameters
	Ap = 1
	Zp = 1
	AT = 12
	ZT = 6
	V0 = 51.8011
	VS = -34.148
	a0 = 0.644174
	aS = 0.644174
	R0 = 2.96268
	RS = 2.82043
	l_mom = 0
	j_mom = 0.5
	Eb = 0.421
	"""
	mT = Masses.GetMass(ZT, AT)
	mproton = Masses.GetMass(1,1)
	redMass = ReducedMass(mT, mproton)

	print("Generating problem and calculating classical turning points...")
	my_wkb = WKBPotential(AT, ZT, Zp, V0, VS, a0, aS, R0, RS, redMass, l_mom, j_mom, Eb)
	print("Finished.")
	print("--------------------------------")

	print("Performing integration and calculating decay width (this may take some time)...")
	gamma, halflife = my_wkb.CalculateDecayWidth()
	print("Finished.")
	print("--------------------------------")

	print("Calculated width (MeV): ", gamma)
	print("Calculated half-life (s): ", halflife)
	print("--------------------------------")

if __name__ == '__main__':
	main()