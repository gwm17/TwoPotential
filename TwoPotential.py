#!/usr/bin/env python3

import numpy as np 
import scipy as sp
from scipy import integrate
import matplotlib.pyplot as plt
from MassTable import Masses
import BoundState as BS

HBARC = 193.173 #MeV*fm
C = 3.0e8*1e15 #fm/s
HBAR = 6.5821e-16*1e-6 #MeV*s
ALPHA = 1.0/137.0 #Fine structure const

def ReducedMass(m1, m2) :
	return m1*m2/(m1+m2)

def WoodSaxson(r, A, V0, m) :
	r0 = 1.17
	a = 0.75
	R0 = r0*A**(1.0/3.0)
	return -2.0*m*V0/(HBARC**2.0*(1.0 + np.exp((r-R0)/a)))

def Coulomb(r, Z1, Z2, A, m) :
	r0 = 1.17
	R0 = r0*A**(1.0/3.0)
	if r <= R0 :
		return ALPHA*2.0*m*Z1*Z2/(2.0*HBARC*R0)*(3.0 - r**2.0/(R0**2.0))
	else :
		return ALPHA*2.0*m*Z1*Z2/(HBARC*r)

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
	R = (1.17*A**0.33)/2.0
	r_range = np.linspace(R, rmax, steps)
	curMax = 0.0
	curV = 0.0
	rMax = 0.0
	for r in r_range :
		curV = PotentialV(r, A, Z1, Z2, V0, m, l)
		if curV > curMax :
			curMax = curV
			rMax = r
	return rMax, curMax

def GeneratePotential(steps, rmax, A, Z1, Z2, V0, m, l, VB, rB, Potential) :
	r_range = np.linspace(1.0, rmax, steps)
	V_values = np.zeros(steps)
	for i in np.arange(0, steps):
		if VB == 0.0 :
			V_values[i] = HBARC**2.0/(2.0*m)*Potential(r_range[i], A, Z1, Z2, V0, m, l)
		else :
			V_values[i] = HBARC**2.0/(2.0*m)*Potential(r_range[i], A, Z1, Z2, V0, m, l, VB, rB)
	return V_values, r_range



def NumerovSolver(nsteps, rmax, k2_b, A, Z1, Z2, V0, m, l, VB, rB, Potential) :
	dr, r_range = np.linspace(1e-14, rmax, num=nsteps, retstep=True)
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


def GenerateNormWavefunc(u_array, rmax, nsteps) :
	r_range = np.linspace(1e-14, rmax, nsteps)
	#Use Simpson's Rule to integrate
	value = integrate.simps(u_array, r_range)
	print("Result: ", value)
	u_norm = u_array/value
	return u_norm, r_range

def main() :
	Ap = 1
	Zp = 1
	A12C = 12
	Z12C = 6
	V0_guess = 50.0
	l_mom = 0
	m12C = Masses.GetMass(6, 12)
	mproton = Masses.GetMass(1,1)
	redMass = ReducedMass(m12C, mproton)
	A = 12
	Zp = 1

	rB, VB = FindMaximumHeight(1000, 3.0*1.17*A12C**(1.0/3.0), A12C, Z12C, Zp, V0_guess, redMass, l_mom)

	print("Maximum height of barrier found to be: ", rB, " with a height of: ",VB)

	bsSolver = BS.BoundState(A12C, Z12C, Zp, redMass, l_mom, VB, rB, PotentialU, 0.000001)

	print("Solving bound state problem...")
	V0, uBound = bsSolver.FindV0(0.412, V0_guess,15000,(2.0*rB))
	print("Finished.")

	print("Normalizing bound state wavefunction...")
	uBoundNorm, r_wavefunc = GenerateNormWavefunc(uBound, 2.0*rB, 15000)
	print("Finished.")

	V_initial, r_range = GeneratePotential(15000, (2.0*rB),A12C,Z12C,Zp,V0_guess,redMass,l_mom,0,0,PotentialV)
	V_solved, r_solved = GeneratePotential(15000, (2.0*rB),A12C,Z12C,Zp,V0,redMass,l_mom,0,0,PotentialV)
	U_initial, r_initial = GeneratePotential(15000, (2.0*rB),A12C,Z12C,Zp,V0_guess,redMass,l_mom,VB,rB,PotentialU)
	U_final, r_final = GeneratePotential(15000, (2.0*rB),A12C,Z12C,Zp,V0,redMass,l_mom,VB,rB,PotentialU)
	figure, (ax1, ax2) = plt.subplots(2)
	#ax1.plot(r_range, V_initial, label="V guessed")
	#ax1.plot(r_solved, V_solved, label="V solved")
	ax1.plot(r_initial, U_initial, label="U guessed")
	ax1.plot(r_initial, U_final, label="U solved")

	ax2.plot(r_wavefunc, uBoundNorm, label="Bound State Wave function")
	plt.legend()


if __name__ == '__main__':
	main()
	plt.show(block=True)

