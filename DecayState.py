#!/usr/bin/env python3

import scipy as sp
from scipy.misc import derivative
from scipy import optimize
from scipy import interpolate
import numpy as np
import mpmath as mp
from NuclearPotential import *

class DecayState:
	"""
	Class representing the bound state of the two potential problem. Solves for depth of the Wood-Saxon well
	for a given bound state, as well as the wavefunction of the bound state.
	Takes in parameters describing the system (height of the barrier, barrier radius, Coulomb params) and
	an initial guess for the depth of the Wood-Saxon well, as well as bound state energy. Uses a shooting method with 
	Numerov method for solving the Schrodinger equation.
	"""

	def __init__(self, potential, Eb) :
		self.Potential = potential
		self.Eb = Eb
		self.eta = ALPHA*np.sqrt(self.Potential.m/(2.0*self.Eb))*self.Potential.Z1*self.Potential.Z2
		self.k = np.sqrt(2.0*self.Potential.m*self.Eb)/HBARC
		self.rBig = self.Potential.rB*1.5
		VB_MeV = self.Potential.Convert2MeV(self.Potential.VB)
		self.alpha = np.sqrt(2.0*self.Potential.m*(VB_MeV-self.Eb))/HBARC
		self.delta_c = 0
		self.ChiRB = 0
		self.nsteps = 0 
		self.rmax = 0

	def InteriorWavefunction(self, r) :
		return self.ChiRB*np.sinh(self.alpha*r)/np.sinh(self.alpha*self.Potential.rB)

	def InteriorDerivative(self, r):
		return self.ChiRB*self.alpha*np.cosh(self.alpha*r)/np.sinh(self.alpha*self.Potential.rB)

	def RegCoulomb(self, r) :
		return mp.coulombf(self.Potential.l, self.eta, self.k*r)

	def IrregCoulomb(self, r) :
		return mp.coulombg(self.Potential.l, self.eta, self.k*r)

	def ExteriorWavefunction(self, r) :
		return np.cos(self.delta_c)*mp.coulombf(self.Potential.l, self.eta, self.k*r) - np.sin(self.delta_c)*mp.coulombg(self.Potential.l, self.eta, self.k*r)

	#solve for the phase shift
	def SolveDeltaC(self, logder) :
		f = self.RegCoulomb(self.Potential.rB)
		g = self.IrregCoulomb(self.Potential.rB)
		fprime = mp.diff(self.RegCoulomb, self.Potential.rB)
		gprime = mp.diff(self.IrregCoulomb, self.Potential.rB)
		rho = 1.0/self.k*logder
		value1 = (fprime - rho*f)
		value2 = (g*rho - gprime)
		print("f: ", f, " g: ",g," fprime: ",fprime," gprime: ",gprime," rho: ",rho," value: ",value1/value2)
		delta = mp.atan2(value1, value2)
		if delta < 0 :
			delta += 2.0*np.pi
		return delta

	def NumerovSolver(self, nsteps, rmax) :
		r_range, dr = np.linspace(self.Potential.rB, rmax, num=nsteps, retstep=True)
		r_range = np.append(r_range, r_range[nsteps-1] + dr)
		u = np.zeros(nsteps)
		fac = dr**2.0/12.0
		#initial boundary conditions
		u[0] = self.ChiRB
		u[1] = self.ChiRB+self.InteriorDerivative(self.ChiRB)
		u_pastBoundary = 0
		k2_b = self.k**2.0
		for i in np.arange(2, nsteps+1) :
			k2_3 = k2_b - self.PotentialFunction(r_range[i], self.A, self.Z1, self.Z2, self.V0, self.m, self.l, self.j, self.VB, self.rB)
			k2_2 = k2_b - self.PotentialFunction(r_range[i-1], self.A, self.Z1, self.Z2, self.V0, self.m, self.l, self.j, self.VB, self.rB)
			k2_1 = k2_b - self.PotentialFunction(r_range[i-2], self.A, self.Z1, self.Z2, self.V0, self.m, self.l, self.j, self.VB, self.rB)
			a = 1.0 + fac*k2_3
			b = 2.0*(1.0 - 5.0*fac*k2_2)
			c = 1.0 + fac*k2_1
			if i == nsteps :
				u_pastBoundary = (b*u[i-1] - c*u[i-2])/a
			else :	
				u[i] = (b*u[i-1] - c*u[i-2])/a

		logDerivAtBoundary = (u_pastBoundary - u[nsteps-2])/(2*dr*u[nsteps-1])
		return logDerivAtBoundary, u

	def CalculateBoundaries(self, guess) :
		self.ChiRB = guess
		logder, u_mid = self.NumerovSolver(self.nsteps, self.rBig)

		print("logder: ", logder)
		print("delta c: ", self.SolveDeltaC(logder))
		self.delta_c = float(self.SolveDeltaC(logder))

		ext_rBig = self.ExteriorWavefunction(self.rBig)

		return u_mid[self.nsteps-1]-ext_rBig

	def FindBoundaryValue(self, steps, start_guess) :
		minVal = start_guess-2.0*start_guess
		maxVal = start_guess+2.0*start_guess
		self.nsteps = steps
		value = optimize.brentq(self.CalculateBoundaries, minVal, maxVal)

		return value

	def CalculateWavefunction(self, nsteps, rMax) :
		u_in = np.zeros(nsteps)
		r_in = np.linspace(0,self.Potential.rB,nsteps)
		for i in np.arange(0, nsteps):
			u_in[i] = self.InteriorWavefunction(r_in[i])

		logder, u_mid = self.NumerovSolver(nsteps, self.rBig)
		fMiddle = interpolate.interp1d(np.linspace(self.rB, self.rBig,nsteps), u_mid, kind='cubic')

		r = np.linspace(1e-14,rMax,nsteps)
		u = np.zeros(nsteps)

		for i in np.arange(0, nsteps):
			if r[i] < self.Potential.rB :
				u[i] = self.InteriorWavefunction(r[i])
			elif r[i] < self.rBig :
				u[i] = fMiddle(r[i])
			else:
				u[i] = self.ExteriorWavefunction(r[i])

		return u

	#Aberg TPA1 approximation, no intermediate region
	def TPA1(self, steps, rmax) :
		fprime = derivative(self.RegCoulomb, self.Potential.rB)
		gprime = derivative(self.IrregCoulomb, self.Potential.rB)

		numerator = self.k*fprime-self.alpha*(1.0/np.tanh(self.alpha*self.Potential.rB))*self.RegCoulomb(self.Potential.rB)
		denominator = self.k*gprime-self.alpha*(1.0/np.tanh(self.alpha*self.Potential.rB))*self.IrregCoulomb(self.Potential.rB)

		tan_delta = numerator/denominator

		delta = mp.atan2(numerator, denominator)
		if delta < 0:
			delta += 2.0*np.pi
		self.delta_c = float(delta)

		r_range = np.linspace(0, rmax, steps)
		u = np.zeros(steps)
		self.ChiRB = self.ExteriorWavefunction(self.Potential.rB)

		print("fprime: ",fprime," gprime: ",gprime," numerator: ",numerator," denominator: ",denominator," tan: ",tan_delta," delta: ",self.delta_c," chiRB: ",self.ChiRB)
		for i in np.arange(0, steps) :
			r = r_range[i]
			if r <= self.Potential.rB :
				u[i] = self.InteriorWavefunction(r)
			else :
				u[i] = self.ExteriorWavefunction(r)

		return u

	#Aberg TPA2 approximation, assume pure coulomb solution
	def TPA2(self, steps, rmax):
		u = np.zeros(steps)
		r_range = np.linspace(0, rmax, steps)
		for i in np.arange(0,steps) :
			u[i] = float(self.RegCoulomb(r_range[i]))
		return u