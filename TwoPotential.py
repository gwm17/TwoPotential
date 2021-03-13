#!/usr/bin/env python3

import numpy as np 
import scipy as sp 
import matplotlib as mp
from MassTable import Masses
import BoundState as BS

HBAR = 193.173 #MeV*fm
ALPHA = 1.0/137.0 #Fine structure const

def ReducedMass(m1, m2) :
	return m1*m2/(m1+m2)

def WoodSaxson(r, A, V0, m) :
	r0 = 1.17
	a = 0.75
	R0 = r0*A**(1.0/3.0)
	return -2.0*m*V0/(HBAR**2.0*(1.0 + np.exp((r-R0)/a)))

def Coulomb(r, Z1, Z2, A, m) :
	r0 = 1.17
	R0 = r0*A**(1.0/3.0)
	if r <= R0 :
		return ALPHA*2.0*m*Z1*Z2/(2.0*HBAR*R0)*(3.0 - r**2.0/(R0**2.0))
	else :
		return ALPHA*2.0*m*Z1*Z2/(HBAR*r)

def Centripital(r, m, l) :
	if l == 0 :
		return 0.0
	else :
		return (l**2.0 + l)/(r**2.0)

def PotentialV(r, A, Z1, Z2, V0, m, l) :
	return WoodSaxson(r, A, V0, m) + Coulomb(r, Z1, Z2, A, m) + Centripital(r, m, l)

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

def FindMaximumHeight(steps, rmax, A, Z1, Z2, V0, m, l) :
	r_range = np.linspace(0, rmax, steps)
	curMax = 0.0
	curV = 0.0
	for r in r_range :
		curV = PotentialV(r, A, Z1, Z2, V0, m, l)
		if curV > curMax :
			curMax = curV
	return r, curMax

def NumerovSolver(nsteps, rmax, k2_b, A, Z1, Z2, V0, m, l, VB, rB, Potential) :
	dr, r_range = np.linspace(0, rmax, num=nsteps, retstep=True)
	np.append(r_range, r_range[nsteps-1] + dr)
	u = np.zeros(nsteps)
	fac = dr**2.0/12.0
	#initial boundary conditions
	u[1] = dr
	u_pastBoundary = 0
	for i in np.arange(2, nsteps+1) :
		k2_3 = k2_b - Potential(r_range[i], A, Z1, Z2, V0, m, l, VB, rB)
		k2_2 = k2_b - Potential(r_range[i-1], A, Z1, Z2, V0, m, l, VB, rB)
		k2_1 = k2_b - Potential(r_range[i-2], A, Z1, Z2, V0, m, l, VB, rB)
		a = 1.0 + fac*k2_3
		b = 2.0*(1.0 - 5.0*fac*k2_2)
		c = 1.0 + fac*k2_1
		if i == nsteps :
			u_pastBoundary = (b*u[i-1] - c*u[i-2])/a
		else :	
			u[i] = (b*u[i-1] - c*u[i-2])/a

	logDerivAtBoundary = (u_pastBoundary - u[nsteps-2])/(2*dr*u[nsteps-1])
	return logDerivAtBoundary, u


def main() :
	Ap = 1
	Zp = 1
	A12C = 12
	Z12C = 6
	V0_guess = 50.0
	m12C = Masses.GetMass(6, 12)
	mproton = Masses.GetMass(1,1)
	redMass = ReducedMass(m12C, mproton)
	A = 12
	Zp = 1

	rB, VB = FindMaximumHeight(1000, 3.0*1.17*A12C**(1.0/3.0), A12C, Z12C, Zp, V0_guess, redMass, 0)

	print("Maximum height of barrier found to be: ", rB, " with a height of: ",VB)

	bsSolver = BS.BoundState(A12C, Z12C, Zp, redMass, 0, VB, rB, PotentialU, 0.000001)

	print("Solving bound state problem...")
	V0, psiB = bsSolver.FindV0(0.412, V0_guess,1500,(2.0*rB))
	print("Finished.")

if __name__ == '__main__':
	main()

