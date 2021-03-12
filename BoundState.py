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
	CHARGE = 1.602e-19 #Coulombs
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

	def NumerovSolver(nsteps, rmax, E, V0) :
		dr, r_range = np.linspace(0, rmax, num=nsteps, retstep=True)
		np.append(r_range, r_range[nsteps-1] + dr)
		u = np.zeros(nsteps)
		fac = dr**2.0/12.0
		#initial boundary conditions
		u[1] = dr
		u_pastBoundary = 0
		for i in np.arange(2, nsteps+1) :
			k2_3 = 2.0*m*(self.PotentialFunction(r_range[i], self.A, self.Z1, self.Z2, V0, self.m, self.l, self.VB, self.rB) - E)/HBAR**2.0
			k2_2 = 2.0*m*(self.PotentialFunction(r_range[i-1], self.A, self.Z1, self.Z2, V0, self.m, self.l, self.VB, self.rB) - E)/HBAR**2.0
			k2_1 = 2.0*m*(self.PotentialFunction(r_range[i-2], self.A, self.Z1, self.Z2, V0, self.m, self.l, self.VB, self.rB) - E)/HBAR**2.0
			a = 1.0 + fac*k2_3
			b = 2.0*(1.0 - 5.0*fac*k2_2)
			c = 1.0 + fac*k2_1
			if i == nsteps :
				u_pastBoundary = (b*u[i-1] - c*u[i-2])/a
			else :	
				u[i] = (b*u[i-1] - c*u[i-2])/a

		logDerivAtBoundary = (u_pastBoundary - u[nsteps-2])/(2*dr*u[nsteps-1])
		return logDerivAtBoundary, u

	def FindV0(Eb, V0_guess, nsteps, rmax) :
		V1 = V0_guess
		V2 = V0_guess+1.0
		logder, psi2 = NumerovSolver(nsteps, rmax, Eb, V1)

		while abs(V1-V2) > tol:
			psi1 = psi2
			logder, psi2 = NumerovSolver(nsteps, rmax, Eb, V2)
			V1 = V2
			V2 = V2 - psi2[nsteps-1]*(V2-V1)/(psi2[nsteps-1] - psi1[nsteps-1])
		return V2, psi2




