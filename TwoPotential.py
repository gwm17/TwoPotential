#!/usr/bin/env python3

import numpy as np 
import scipy as sp 
import matplotlib as mp 

CHARGE = 1.602e-19 #Coulombs
HBAR = 193.173 #MeV*fm

def ReducedMass(m1, m2) :
	return m1*m2/(m1+m2)

def WoodSaxson(r, A, V0) :
	r0 = 1.17
	a = 0.75
	R0 = r0*A**(1.0/3.0)
	return -V0/(1.0 + np.exp((r-R0)/a))

def Coulomb(r, Z1, Z2, A) :
	r0 = 1.17
	R0 = r0*A**(1.0/3.0)
	if r <= Rn :
		return Z1*Z2*CHARGE**2.0/(2.0*R0)*(3.0 - r**2.0/(R0**2.0))
	else :
		return Z1*Z2*CHARGE**2.0/r

def Centripital(r, m, l) :
	return HBAR**2.0*l*(l+1.0)/(2*m*r*r)

def PotentialV(r, A, Z1, Z2, V0, m, l) :
	return WoodSaxson(r, A, V0) + Coulomb(r, Z1, Z2, A) + Centripital(r, m, l)

def PotentialU(r, A, Z1, Z2, V0, m, l, VB, rB):
	if r<rB :
		return PotentialV(r, A, Z1, Z2, V0, m, l)
	else :
		return VB

def PotentialWTilde(r, A, Z1, Z2, V0, m, l, VB, rB) :
	if r<rB :
		return VB
	else :
		return PotentialV(r, A, Z1, Z2, V0, m, l)

def FindMaximumHeight(dr, rmax, A, Z1, Z2, V0, m, l) :
	r_range = np.linspace(0, rmax, dr)
	curMax = 0.0
	curV = 0.0
	for r in r_range :
		curV = PotentialV(r)
		if curV > curMax :
			curMax = curV
	return r, curMax

def NumerovSolver(nsteps, rmax, E, A, Z1, Z2, V0, m, l, VB, rB, Potential) :
	dr, r_range = np.linspace(0, rmax, num=nsteps, retstep=True)
	np.append(r_range, r_range[nsteps-1] + dr)
	u = np.zeros(nsteps)
	fac = dr**2.0/12.0
	#initial boundary conditions
	u[1] = dr
	u_pastBoundary = 0
	for i in np.arange(2, nsteps+1) :
		k2_3 = 2.0*m*(E - Potential(r_range[i], A, Z1, Z2, V0, m, l, VB, rB))/HBAR**2.0
		k2_2 = 2.0*m*(E - Potential(r_range[i-1], A, Z1, Z2, V0, m, l, VB, rB))/HBAR**2.0
		k2_1 = 2.0*m*(E - Potential(r_range[i-2], A, Z1, Z2, V0, m, l, VB, rB))/HBAR**2.0
		a = 1.0 + fac*k2_3
		b = 2.0*(1.0 - 5.0*fac*k2_2)
		c = 1.0 + fac*k2_1
		if i == nsteps :
			u_pastBoundary = (b*u[i-1] - c*u[i-2])/a
		else :	
			u[i] = (b*u[i-1] - c*u[i-2])/a

	logDerivAtBoundary = (u_pastBoundary - u[nsteps-2])/(2*dr*u[nsteps-1])
	return logDerivAtBoundary, u




