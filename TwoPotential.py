#!/usr/bin/env python3

import numpy as np 
import scipy as sp
import sys
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
from MassTable import Masses
from Config import *
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

def main(filename) :
	print("------------------------------------------")
	print("------GWM & JCE Two Potential Solver------")
	print("------------------------------------------")
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

	"""147Tm Parameters
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
	"""
	print("Loading configuration in file: ", filename)
	myConfig = ConfigFile(filename)
	if myConfig.IsValid() == False :
		print("Unable to process the configuration file: ", filename)
		return
	print("Loaded succesfully.")
	print("------------------------------------------")
	ZT = 0
	AT = 0
	Zp = 1
	Ap = 1
	nsamples = 15000
	Tolerance = 1e-6
	k2_b = 0.0
	redMass = 0
	mT = 0
	mproton = Masses.GetMass(Zp, Ap)
	rB = 0
	VB = 0
	VB_MeV = 0
	alpha = 0
	rMax = 0

	for config in myConfig.configs :
		print("Running configuration: A: ", config.A, " Z: ", config.Z, " l: ", config.l, " j: ", config.j)
		ZT = config.Z - Zp
		AT = config.A - Ap

		mT = Masses.GetMass(ZT, AT)
		redMass = ReducedMass(mT, mproton)
		k2_b = 2.0*redMass*config.Eb/HBARC**2.0

		myPotential = NuclearPotential(AT,ZT,Zp, config.V0, config.VS, config.a0, config.aS, config.R0, config.RS,redMass, config.l, config.j)

		rB, VB = myPotential.FindMaximumHeight(100000, 3.0*config.R0)
		myPotential.VB = VB
		myPotential.rB = rB
		VB_MeV = myPotential.Convert2MeV(VB)
		alpha = np.sqrt(2.0*redMass*(VB_MeV-config.Eb))/HBARC
		rMax = rB*1.2

		print("------------------------------------------")
		print("Maximum height of barrier found to be: ", rB, " with a height of: ", VB_MeV," MeV")

		bsSolver = BS.BoundState(myPotential, config.Eb, Tolerance)

		print("Solving bound state problem (this may take some time)...")
		V0, uBound = bsSolver.FindV0(config.V0,nsamples,rMax)
		myPotential.VB = bsSolver.Potential.VB
		myPotential.rB = bsSolver.Potential.rB
		myPotential.V0 = V0
		myPotential.VS = -0.2*V0
		print("Finished.")

		print("Normalizing bound state wavefunction...")
		uBoundNorm, r_wavefunc = GenerateNormWavefunc(uBound, rMax, nsamples)
		print("Finished.")
		print("------------------------------------------")

		print("Solving scattering problem...")
		dsSolver = DS.DecayState(myPotential, config.Eb)
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
		t = np.log(2)*HBAR/width
		print("Half-life(s): ", t)
		print("------------------------------------------")
		myConfig.WriteResults(config, width, t, "TPA")

	""""
	r_plot = np.linspace(1.0, rMax, nsamples)

	V_solved = myPotential.GeneratePotential(r_plot, myPotential.PotentialV)
	U_final = myPotential.GeneratePotential(r_plot, myPotential.PotentialU)
	W_final = myPotential.GeneratePotential(r_plot, myPotential.PotentialW)
	Wt_final = myPotential.GeneratePotential(r_plot, myPotential.PotentialWTilde)
	figure1, ax1 = plt.subplots(2,2)
	figure2, ax2 = plt.subplots(2)
	ax1[0,0].plot(r_plot, V_solved, label="V(r)")
	ax1[0,0].axvline(x=myPotential.rB, label="rB", color="g",dashes=[2,2])
	ax1[0,0].axhline(y=Eb, label="Eb", color="c",dashes=[1,1])
	ax1[0,0].legend()
	ax1[0,1].plot(r_plot, U_final, label="U(r)")
	ax1[0,1].legend()
	ax1[1,0].plot(r_plot, W_final, label="W(r)")
	ax1[1,0].legend()
	ax1[1,1].plot(r_plot, Wt_final, label="WTilde(r)")
	ax1[1,1].legend()

	ax2[0].plot(r_wavefunc, uBoundNorm/r_wavefunc, label="Bound State Wave function")
	ax2[0].legend()

	ax2[1].plot(r_wavefunc, uDecay/r_wavefunc, label="Scattering Wave function")
	ax2[1].legend()
	"""


if __name__ == '__main__':
	if len(sys.argv) >= 2 :
		main(sys.argv[1])
	else:
		print("Invalid number of arguments, needs a config file!")
	#plt.show(block=True)

