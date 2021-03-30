#!/usr/bin/env python3

import numpy as np 
import scipy as sp
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
from MassTable import Masses
from NuclearPotential import *
import BoundState as BS
import DecayState as DS

def ReducedMass(m1, m2) :
	return m1*m2/(m1+m2)

def GenerateNormWavefunc(u_array, rmax, nsteps) :
	r_range = np.linspace(1e-14, rmax, nsteps)
	#Use Simpson's Rule to integrate
	psi2 = (u_array)**2.0
	value = integrate.simps(psi2, r_range)
	u_norm = u_array/np.sqrt(value)
	return u_norm, r_range

def main() :
	print("------------------------------------------")
	print("------GWM & JCE Two Potential Solver------")
	"""13N Paramters
	Ap = 1
	Zp = 1
	AT = 12
	ZT = 6
	V0_guess = 51.8011
	VS_guess = -0.2*V0_guess
	a0 = 0.644174
	aS = 0.644174
	R0 = 2.96268
	RS = 2.82043
	l_mom = 0
	j_mom = 0.5
	Eb = 0.421
	"""

	#"""147Tm Parameters
	rWS = 1.17
	rS = 1.01
	Ap = 1
	Zp = 1
	AT = 146
	ZT = 68
	V0_guess = 53.4
	VS_guess = -0.2*V0_guess
	a0 = 0.75
	aS = 0.75
	R0 = rWS*AT**(1.0/3.0)
	RS = rS*AT**(1.0/3.0)
	#l_mom = 5
	l_mom = 2
	j_mom = 1.5
	#j_mom = 5.5
	Eb = 1.132
	#Eb = 1.071
	#"""

	mT = Masses.GetMass(ZT, AT)
	mproton = Masses.GetMass(1,1)
	redMass = ReducedMass(mT, mproton)
	nsamples = 15000
	Tolerance = 1e-6
	k2_b = 2.0*redMass*Eb/HBARC**2.0

	myPotential = NuclearPotential(AT,ZT,Zp,V0_guess,VS_guess,a0,aS,R0,rS,redMass,l_mom,j_mom)

	rB, VB = myPotential.FindMaximumHeight(100000, 3.0*R0)
	myPotential.VB = VB
	myPotential.rB = rB
	VB_MeV = myPotential.Convert2MeV(VB)
	alpha = np.sqrt(2.0*redMass*(VB_MeV-Eb))/HBARC
	rMax = rB*1.2

	print("------------------------------------------")
	print("Maximum height of barrier found to be: ", rB, " with a height of: ", VB_MeV," MeV")

	bsSolver = BS.BoundState(myPotential, Eb, Tolerance)

	print("Solving bound state problem...")
	V0, uBound = bsSolver.FindV0(V0_guess,nsamples,rMax)
	myPotential.V0 = V0
	myPotential.VS = -0.2*V0
	print("Finished.")

	print("Normalizing bound state wavefunction...")
	uBoundNorm, r_wavefunc = GenerateNormWavefunc(uBound, rMax, nsamples)
	print("Finished.")
	print("------------------------------------------")

	print("Solving scattering problem...")
	dsSolver = DS.DecayState(myPotential, Eb)
	uDecay = dsSolver.TPA2(nsamples, rMax)
	print("Finished.")
	print("------------------------------------------")

	print("Interpolating and calculating value of bound state at boundary...")
	fBoundFunction = interpolate.interp1d(r_wavefunc, uBoundNorm)
	uBound_atrB = fBoundFunction(rB)
	print("Finished. Value at boundary: ",uBound_atrB)

	print("Interpolating and calculating value of decay state at boundary...")
	fDecayFunction = interpolate.interp1d(r_wavefunc, uDecay)
	uDecay_atrB = fDecayFunction(rB)
	print("Finished. Value at boundary: ",uDecay_atrB)
	print("------------------------------------------")

	print("Calculating decay width...")
	width = (2.0*alpha*HBARC)**2.0/(redMass*np.sqrt(k2_b))*(uBound_atrB*uDecay_atrB)**2.0
	print("Finished. Width: ", width," MeV")
	tau = HBAR/width
	print("Half-life(s): ", np.log(2)*tau)
	print("------------------------------------------")

	r_plot = np.linspace(1.0, rMax, nsamples)

	V_initial = myPotential.GeneratePotential(r_plot, myPotential.PotentialV)
	V_solved = myPotential.GeneratePotential(r_plot, myPotential.PotentialV)
	U_initial = myPotential.GeneratePotential(r_plot, myPotential.PotentialU)
	U_final = myPotential.GeneratePotential(r_plot, myPotential.PotentialU)
	V_WS = myPotential.GeneratePotential(r_plot, myPotential.WoodSaxson)
	figure, (ax1, ax2, ax3) = plt.subplots(3)
	ax1.plot(r_plot, V_initial, label="V guessed")
	ax1.plot(r_plot, V_WS, label="Vn")
	ax1.plot(r_plot, V_solved, label="V solved")
	ax1.plot(r_plot, U_initial, label="U guessed")
	ax1.plot(r_plot, U_final, label="U solved")
	ax1.legend()

	ax2.plot(r_wavefunc, uBoundNorm/r_wavefunc, label="Bound State Wave function")
	ax2.legend()

	ax3.plot(r_wavefunc, uDecay/r_wavefunc, label="Scattering Wave function")
	ax3.legend()


if __name__ == '__main__':
	main()
	plt.show(block=True)

