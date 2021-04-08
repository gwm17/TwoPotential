#!/usr/bin/env python3

import numpy as np
import scipy as sp
from scipy import interpolate
from scipy import optimize
from NuclearPotential import *

class BoundState:
	"""
	Class representing the bound state of the two potential problem. Solves for depth of the Wood-Saxon well
	for a given bound state, as well as the wavefunction of the bound state.
	Takes in parameters describing the system (height of the barrier, barrier radius, Coulomb params) and
	an initial guess for the depth of the Wood-Saxon well, as well as bound state energy. Uses a shooting method with 
	Numerov method for solving the Schrodinger equation.
	"""
	def __init__(self, potential, Eb, tol) :
		self.Potential = potential
		self.Eb = Eb
		self.k2_b = 2.0*self.Potential.m*self.Eb/(HBARC**2.0)
		self.tolerance = tol

	def NumerovSolver(self, nsteps, rmax) :
		r_range, dr = np.linspace(1e-14, rmax, num=nsteps, retstep=True)
		r_range = np.append(r_range, r_range[nsteps-1] + dr)
		u = np.zeros(nsteps)
		fac = dr**2.0/12.0
		#initial boundary conditions
		u[1] = dr
		u_pastBoundary = 0
		for i in np.arange(2, nsteps+1) :
			k2_3 = self.k2_b - self.Potential.PotentialU(r_range[i])
			k2_2 = self.k2_b - self.Potential.PotentialU(r_range[i-1])
			k2_1 = self.k2_b - self.Potential.PotentialU(r_range[i-2])
			a = 1.0 + fac*k2_3
			b = 2.0*(1.0 - 5.0*fac*k2_2)
			c = 1.0 + fac*k2_1
			if i == nsteps :
				u_pastBoundary = (b*u[i-1] - c*u[i-2])/a
			else :
				u[i] = (b*u[i-1] - c*u[i-2])/a

		logDerivAtBoundary = (u_pastBoundary - u[nsteps-2])/(2*dr*u[nsteps-1])
		return logDerivAtBoundary, u

	#Shoot over V0 depth to find good bound state wave function
	def FindV0(self, V0_guess, nsteps, rmax) :
		V1 = V0_guess
		V2 = V0_guess+0.0001
		self.Potential.V0 = V0_guess
		self.Potential.VS = -0.2*V0_guess
		print("Solving intial guess for bound state...")
		logder, psi2 = self.NumerovSolver(nsteps, rmax)
		print("Shooting over V0 to find optimal depth...")

		self.Potential.rB, self.Potential.VB = self.Potential.FindMaximumHeight(100000, 3.0*self.Potential.R0)

		#Secant method
		while abs(V1 - V2) > self.tolerance:
			self.Potential.V0 = V2
			self.Potential.VS = -0.2*V2
			psi1 = psi2
			logder, psi2 = self.NumerovSolver(nsteps, rmax)
			deriv = (V2-V1)/(psi2[nsteps-1] - psi1[nsteps-1])
			V1, V2 =V2, V2 - psi2[nsteps-1]*(V2-V1)/(psi2[nsteps-1] - psi1[nsteps-1])
			self.Potential.rB, self.Potential.VB = self.Potential.FindMaximumHeight(100000, 3.0*self.Potential.R0)

		print("Found potential depth to be: ",V2," with precison ",self.tolerance)
		return V2, psi2




