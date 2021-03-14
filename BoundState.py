#!/usr/bin/env python3

import numpy as np
import scipy as sp

class BoundState:
	"""
	Class representing the bound state of the two potential problem. Solves for depth of the Wood-Saxon well
	for a given bound state, as well as the wavefunction of the bound state.
	Takes in parameters describing the system (height of the barrier, barrier radius, Coulomb params) and
	an initial guess for the depth of the Wood-Saxon well, as well as bound state energy. Uses a shooting method with 
	Numerov method for solving the Schrodinger equation.
	"""
	HBAR = 193.173 #MeV*fm
	def __init__(self, A, Z1, Z2, m, l, VB, rB, Potential, tol):
		self.PotentialFunction = Potential
		self.tolerance = tol
		self.A = A
		self.Z1 = Z1
		self.Z2 = Z2
		self.m = m
		self.l = l
		self.VB = VB
		self.rB = rB

	def NumerovSolver(self, nsteps, rmax, k2_b, V0) :
		r_range, dr = np.linspace(1e-14, rmax, num=nsteps, retstep=True)
		np.append(r_range, r_range[nsteps-1] + dr)
		u = np.zeros(nsteps)
		fac = dr**2.0/12.0
		#initial boundary conditions
		u[1] = dr
		u_pastBoundary = 0
		for i in np.arange(2, nsteps) :
			k2_3 = k2_b - self.PotentialFunction(r_range[i], self.A, self.Z1, self.Z2, V0, self.m, self.l, self.VB, self.rB)
			k2_2 = k2_b - self.PotentialFunction(r_range[i-1], self.A, self.Z1, self.Z2, V0, self.m, self.l, self.VB, self.rB)
			k2_1 = k2_b - self.PotentialFunction(r_range[i-2], self.A, self.Z1, self.Z2, V0, self.m, self.l, self.VB, self.rB)
			a = 1.0 + fac*k2_3
			b = 2.0*(1.0 - 5.0*fac*k2_2)
			c = 1.0 + fac*k2_1
			if i == nsteps :
				u_pastBoundary = (b*u[i-1] - c*u[i-2])/a
			else :	
				u[i] = (b*u[i-1] - c*u[i-2])/a

		logDerivAtBoundary = (u_pastBoundary - u[nsteps-2])/(2*dr*u[nsteps-1])
		return logDerivAtBoundary, u

	def FindV0(self, Eb, V0_guess, nsteps, rmax) :
		V1 = V0_guess
		V2 = V0_guess+1.0
		k2_b = 2.0*self.m*Eb/(self.HBAR**2.0)
		print("Solving intial guess for bound state...")
		logder, psi2 = self.NumerovSolver(nsteps, rmax, k2_b, V1)

		print("Shooting over V0 to find optimal depth...")
		count = 0
		flush = 100
		flush_count = 0

		while abs(V1-V2) > self.tolerance:
			count += 1
			if(count == flush) :
				count = 0
				flush_count += 1
				print("\rNumber of iterations searching for V0: ",flush_count*flush)
			psi1 = psi2
			logder, psi2 = self.NumerovSolver(nsteps, rmax, k2_b, V2)
			V1, V2 =V2, V2 - psi2[nsteps-1]*(V2-V1)/(psi2[nsteps-1] - psi1[nsteps-1])
		print(" Found potential depth to be: ",V2," with precison ",self.tolerance)
		return V2, psi2




