from pytriqs.applications.dft.sumk_dft import *
from pytriqs.applications.dft.converters.wien2k_converter import *
from pytriqs.applications.impurity_solvers.hubbard_I.hubbard_solver import Solver
import numpy as np
import time
from cf_dmft_utils import *

import os
LDAFilename = os.getcwd().rpartition('/')[2]

#=====================================================
# switch to True run DMFT cycle
run_dmft=False

#compound-depended input parameters:
U_int = 6.00
J_hund = 0.85
Natomic=3
GSM_mult = 10

# parameters below are usually not needed to be changed
beta = 40
Chemical_Pot_Init = 0.0
Loops =  5                       # Number of DMFT sc-loops
Mix = 0.7                        # Mixing factor of Sigma after solution of the AIM
DC_type = 0                      # DC type: 0 FLL, 1 Held, 2 AMF
useBlocs = True                  # use bloc structure from LDA input
useMatrix = True                # True: Slater parameters, False: Kanamori parameters U+2J, U, U-J
use_spinflip = False             # use the full rotational invariant interaction?
useSO = True
prec_mu = 0.000001
Nmsb = 1025

#=====================================================

HDFfilename = LDAFilename+'.h5'

# Convert DMFT input:
# Can be commented after the first run
Converter = Wien2kConverter(filename=LDAFilename,repacking=True)
Converter.convert_dft_input()
mpi.barrier()

#check if there are previous runs:
previous_runs = 0
previous_present = False
if mpi.is_master_node():
    ar = HDFArchive(HDFfilename,'a')
    if 'iterations' in ar:
        previous_present = True
        previous_runs = ar['iterations']
    del ar

previous_runs    = mpi.bcast(previous_runs)
previous_present = mpi.bcast(previous_present)


# if previous runs are present, no need for recalculating the bloc structure
# It has to be commented, if you run this script for the first time, starting
# from a converted h5 archive.

# Init the SumK class
SK=SumkDFT(hdf_file=LDAFilename+'.h5',use_dft_blocks=False)
CF_tools = CF_dmft_utils(SK)

Norb = SK.corr_shells[0]['dim']
Nlm=Norb/2
mpi.report("Norb = %s , Nlm = %s "%(Norb,Nlm))
l = SK.corr_shells[0]['l']

# Init the Solver:
S = Solver(beta = beta, l = l, use_spin_orbit=True)

spinmat, orbmat = CF_tools.spin_orb_matrix(l)

if (previous_present):
    # load previous data:
    mpi.report("Using stored data for initialisation")
    if (mpi.is_master_node()):
        ar = HDFArchive(HDFfilename,'a')
        if run_dmft: S.Sigma <<= ar['SigmaF']
        del ar
        SK.chemical_potential, SK.dc_imp, SK.dc_energ = SK.load(['chemical_potential','dc_imp','dc_energ'])
    if run_dmft: S.Sigma = mpi.bcast(S.Sigma)
    SK.chemical_potential = mpi.bcast(SK.chemical_potential)
    SK.dc_imp = mpi.bcast(SK.dc_imp)
    SK.dc_energ = mpi.bcast(SK.dc_energ)

# set DC (with nominal atomic occupancy
if run_dmft:
    dc_value = U_int * (Natomic - 0.5) - J_hund * (Natomic*0.5 - 0.5)
    dm=S.G.density()
    SK.calc_dc( dm, U_interact = U_int, J_hund = J_hund, orb = 0, use_dc_formula = DC_type, use_dc_value=dc_value)
    if (mpi.is_master_node()): print 'DC : ',SK.dc_imp[0]

# DMFT loop:
for IterationNumber in range(1,Loops+1) :

    if not run_dmft: break

    itn = previous_runs + IterationNumber

    SK.put_Sigma([ S.Sigma ])                    # put Sigma into the SumK class:

    chemical_potential = SK.calc_mu( precision = 0.000001 )

    # Density
    S.G << SK.extract_G_loc()[0]
    mpi.report("Total charge of Gloc : %.6f"%S.G.total_density())

    # set atomic levels:
    eal = SK.eff_atomic_levels()[0]
    S.set_atomic_levels( eal = eal )

    # solve it:
    S.solve(U_int = U_int, J_hund = J_hund, verbosity = 1, N_lev = GSM_mult, remove_split = True, Iteration_Number=itn)

    if (mpi.is_master_node()):
        ar = HDFArchive(HDFfilename)

    # Now mix Sigma and G:
    if ((itn>1)or(previous_present)):
        if (mpi.is_master_node()):
            mpi.report("Mixing Sigma and G with factor %s"%Mix)
            if ('SigmaF' in ar):
                S.Sigma <<= Mix * S.Sigma + (1.0-Mix) * ar['SigmaF']
            if ('GF' in ar):
                S.G <<= Mix * S.G + (1.0-Mix) * ar['GF']

        S.G = mpi.bcast(S.G)
        S.Sigma = mpi.bcast(S.Sigma)

    # correlation energy calculations:
    correnerg = 0.5 * (S.G * S.Sigma).total_density()
    mpi.report("Corr. energy = %s"%correnerg)


    # update hdf5
    if (mpi.is_master_node()):
        ar = HDFArchive(HDFfilename,'a')
        ar['iterations'] = itn
        ar['SigmaF'] = S.Sigma
        ar['GF'] = S.G
        ar['correnerg%s'%itn] = correnerg
        ar['DCenerg%s'%itn] = SK.dc_energ
        del ar

    #Save essential data:
    SK.save(['chemical_potential','dc_imp','dc_energ'])

    # compute spin and orbital moments
    SK.put_Sigma([ S.Sigma ])                    # put Sigma into the SumK clas
    Gloc = SK.extract_G_loc()[0]
    dm = Gloc.density()
    spinmom=numpy.trace(numpy.dot(spinmat,dm['ud']))
    orbmom=numpy.trace(numpy.dot(orbmat,dm['ud']))
    mpi.report("On-site spin moment : %s"%(spinmom.real))
    mpi.report("On-site orbital moment : %s"%(orbmom.real))

# find exact chemical potential
SK.put_Sigma([S.Sigma])
SK.chemical_potential = SK.calc_mu(precision=0.000001)

dN, d = SK.calc_density_correction(filename=LDAFilename + '.qdmft')

# remove polarization from the Nd density matrix
timerev = CF_tools.set_timerev(l)

dN = CF_tools.apply_time_reversal(dN, timerev)

CF_tools.save_dens(dN, S.beta, LDAFilename)

dens={}
dens['ud'] = 0j
for ik in range(SK.n_k):
    dens['ud'] += SK.bz_weights[ik] * numpy.trace(dN['ud'][ik])
mpi.report("Occupancy of averaged dens. mat: %s"%(dens['ud']))


if (mpi.is_master_node()):
    if run_dmft:
        ar = HDFArchive(HDFfilename)
        itn = ar['iterations']
        correnerg = ar['correnerg%s'%itn]
        DCenerg = ar['DCenerg%s'%itn]
        del ar
        correnerg -= DCenerg[0]
        print 'Hubbard energy - E_DC: ',correnerg
    else:
        correnerg=0.0

# There are two Nd/unit cell, but we add the same correction to up and down spin
# so no multiplication is needed
#    correnerg *= 2
    with open(LDAFilename+'.qdmftup','a') as f:
        with open(LDAFilename+'.qdmftdn','a') as f1:
            f.write("%.16f\n"%correnerg)
            f1.write("%.16f\n"%correnerg)
