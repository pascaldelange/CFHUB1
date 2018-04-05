from __future__ import division
from scipy import floor, sqrt
from scipy.misc import factorial
from numpy import arange
import numpy
from math import *

eV_to_cm1 = 8065.73
eV_to_K = 11605
Tesla_to_K = 0.67169

# conversion factors between Wybourne (Lkq or Bkq) an Stevens (Akq<R^k>) CF parameters
conv_to_A={
    '0_0': 1.0,
    '2_0': 0.5,
    '2_1': sqrt(6.0),
    '2_2': sqrt(6.0)/2.0,
    '4_0': 0.125,
    '4_1': sqrt(5.0)/2.0,
    '4_2': sqrt(10.0)/4.0,
    '4_3': sqrt(35.0)/2.0,
    '4_4': sqrt(70.0)/8.0,
    '6_0': 1.0/16.0,
    '6_1': sqrt(42.0)/8.0,
    '6_2': sqrt(105.0)/16.0,
    '6_3': sqrt(105.0)/8.0,
    '6_4': 3*sqrt(14.0)/16.0,
    '6_5': sqrt(77.0)*3.0/8.0,
    '6_6': sqrt(231.0)/16.0
    }


class fit_CF_params:
    """Manipulates a CF Hamiltonian in the form $\sum B_{lm}C_{lm} + SO + B_{ex}$"""

    def __init__(self, CF_prms, SO=True, Bex=True, rot_angle_Bex=None, inversion=True,
                fixed_lamd=None, complex_eal=False, spin_pol_B=False, l=3, Y_order=-1,
                spin_order=1):
        r"""
        Initialises the class fit_CF_params with parameters of crystal-field fitting

        Parameters
        ----------
        CF_prms: list
                nested list [[k_0, q_0], [k_1, q_1]...] where k_i, q_i are
                angular qn and its projection, repectively,  for CF parameters to fit
        SO:     boolean, optional
                True (default) if spin-orbit couling is included
                False otherwise
        Bex:    boolean, optional
                True (default) if exchange field B_ex is to be fitted
                False otherwise
        rot_angle_Bex: list of 3 floats, optional
                list of Euler angles [alpha,beta,gamma], in degrees,  defining rotation O
                of the spin operator S=O*S_z*O^{-1} coupled to the exchange field B_ex
                if not provided(default) this spin operator = S_z
        inversion: boolean, optional
                True (default) : inversion symmetry is present, only CF params for q >= 0 are fitted
                False : both q and -q are fitted
        fixed_lamd: float, optional
                the value of spin-orbit parameter lambda
                if not provided(default), then lambda is fitted
        spin_pol_B: boolean, optional
                True (default) : fit CF parameters for UP and DOWN spins
                False: the same CF fit for both spins
        l :   integer
              orbital quantum number for the shell (default 3)
        Y_order:  integer (1 or -1 accepted), optional
                  1(default) : increasing order of orbitals  (from -m to +m) in non-interacting levels matrix
                  -1 : decreasing order of orbitals  (from m to -m)
        spin_order:  integer (1 or -1 accepted), optional
                  1(default) : increasing order of spins  (DOWN,UP)  in non-interacting levels matrix
                  -1 : decreasing order of spins  (UP,DOWN)

        """
        #print info
        print '\n k  q  of CF parameters to be calculated'
        for coff in CF_prms:
            print ' %s  %s'%(coff[0], coff[1])

        if SO:
            print '\nSpin-orbit switched on'
        else:
            print '\nSpin-orbit switched off'

        if not Y_order in [1, -1]:
            raise ValueError('Y_order should be either 1 or -1')
        if not spin_order in [1, -1]:
            raise ValueError('spin_order should be either 1 or -1')

        # C tensors
        self.l = l
        self.C_tens = []
        for coff in CF_prms:
            self.C_tens.append([coff[0], coff[1]])
        self.SO = SO
        self.inversion = inversion
        self.Bex = Bex
        self.rot_angle_Bex = rot_angle_Bex
        self.fixed_lamd = fixed_lamd
        self.complex_eal = complex_eal
        self.spin_pol_B = spin_pol_B
        self.Y_order_m = Y_order
        # Y_order_incr_m=1 if sph. harmonics are arranged
        # from -m to m,  self.Y_order_m=-1 if the order from m to -m
        if Y_order == 1:
            print '\nThe order of spherical harmonics is from -m to m'
        else:
            print '\nThe order of spherical harmonics is from m to -m'
        if spin_order == 1:
            self.sp_arr=[-0.5, 0.5] # order of spins
            print '\nThe order of spins: DOWN, UP'
        else:
            self.sp_arr=[0.5, -0.5] # order of spins
            print '\nThe order of spins: UP, DOWN'


    def fit(self, filename, prn_lev=1):
        r"""
        fits CF parameters, as well as (optionally) Bex and spin-orbit lambda
        from input non-interacting atomic levels matrix

        Parameters
        ----------
        filename: string
                  the name of file where  non-interacting atomic levels matrix is stored
        prn_lev: integer, optional
              printing level, prn_lev=1 prints out all C-tensors, SO,
        as well as spin operators coupled to Bex

        Returns
        --------
        E0 : float
             uniform shift
        lamd : float
             spin-orbit coupling parameter
        Lkq_par : list of floats
             list of values of crystal-field parameters
        B_ex : float
             exchange field
        """
        l = self.l
        norb = 2*l+1
        nlmt = 2*norb
        # read level positions
        with open(filename) as f:
            eal = numpy.zeros([nlmt, nlmt], numpy.complex_)
            for m in range(nlmt):
                line = f.readline()
                rline = [float(i) for i in line.split()]
                for m1 in range(nlmt):
                    if self.complex_eal:
                        eal[m, m1] = rline[m1 * 2] + 1j * rline[m1 * 2 + 1]
                    else:
                        eal[m, m1] = rline[m1]

        if prn_lev == 1:
            print '\nLevel positions (real part):'
            self.__print_mat(eal.real)
            if self.complex_eal:
                print '\nLevel positions (imaginary part):'
                self.__print_mat(eal.imag)

        C_mat = self.__calc_C_mat(l, print_mat=prn_lev)

        if self.SO:
            #set up spin-orbit matirx
            SO_mat = self.__calc_H_SO_mat(l, print_mat=prn_lev)

        if self.Bex:
            Bex_mat = self.__calc_Bex_mat(l=l, print_mat=prn_lev)
        else:
            Bex_mat = None

        # set  (4l+2)x(4l+2) matrices for lstsq
        if self.spin_pol_B:
            ncof = 2 * len(C_mat) + 3
        else:
            ncof = len(C_mat) + 3
        nlm2 = nlmt * nlmt
        a = numpy.zeros([nlm2, ncof])
        b = numpy.zeros([nlm2])

        for spin in range(2):
            shft = spin * norb
            for m in range(norb):
                for m1 in range(norb):
                    lm = nlmt * (m + shft) + shft + m1
                    for icof in range(3, ncof):
                        if self.spin_pol_B:
                            ic = int((icof - 3) / 2)
                            if icof % 2 == 1 and spin == 0:
                                a[lm, icof] = C_mat[ic][m, m1]
                            elif icof%2 == 0 and spin == 1:
                                a[lm, icof] = C_mat[ic][m, m1]
                        else:
                            a[lm, icof] = C_mat[icof - 3][m, m1]
                    if m == m1:
                        a[lm, 0] = 1.0

        lm = 0
        for m in range(nlmt):
            for m1 in range(nlmt):
                b[lm] = eal[m, m1].real
                if self.SO:
                    if self.fixed_lamd is None:
                        a[lm, 1] = SO_mat[m, m1]
                    else:
                        b[lm] -= self.fixed_lamd * SO_mat[m, m1]
                if self.Bex:
                    a[lm, 2] = Bex_mat[m, m1].real
                lm += 1

        x, resid, rank, s = numpy.linalg.lstsq(a, b)

        if self.complex_eal:
            # get complex part of CF parameters
            lm = 0
            for m in range(nlmt):
                for m1 in range(nlmt):
                    b[lm] = eal[m, m1].imag
                    a[lm, 0] = 0.0
                    a[lm, 1] = 0.0
                    a[lm, 2] = 0.0
                    lm += 1
            xim, resid, rank, s = numpy.linalg.lstsq(a, b)

        # Pack into B
        E0 = x[0]
        if self.fixed_lamd is None:
            lamd = x[1]
        else:
            lamd = self.fixed_lamd
        Bex_val = x[2]
        Lkq_par = []
        for icof in range(3, ncof):
            if not self.complex_eal:
                Lkq_par.append(x[icof])
            else:
                Lkq_par.append(x[icof] + 1j * xim[icof])

        # calculate accuracy
        if not self.SO:
            lamd = 0
        if not self.Bex:
            Bex_val = 0.0
        eal_calc = self.calc_lev_pos(CF_params=Lkq_par, E0=E0, lamd=lamd,
                                    Bex_val=Bex_val, Bex_mat_in=Bex_mat,
                                    real_mat=False, prn_lev=0)
        #diff=abs(eal-eal_calc[0:nlmt,0:nlmt].real)
        diff = abs(eal - eal_calc[0:nlmt, 0:nlmt])

        if prn_lev == 1:
            print '\nMatrix of fitting errors:'
            self.__print_mat(diff)

        self.E0 = E0
        self.lamd = lamd
        self.Lkq_par = Lkq_par
        self.Bex_val = Bex_val

        return E0, lamd, Lkq_par, Bex_val


    def calc_lev_pos(self, CF_params=None, E0=None, lamd=None, Bex_val=None,
                    Bex_mat_in=None, real_mat=False, prn_lev=1):
        r"""
        Calculates level positions from input crystal-field, SO, and Bex parameters

        Parameters
        ----------
        CF_params: list, optional
                   list of CF parameters in Lkq convention
                    if None (default) CF parameters stored in self.Lkq_par are used
        E0       : float, optional
                   uniform shift of levels
                   if None (default) the value stored in self.E0 is used
        lamd     : float,optional
                   spin-orbit coupling parameter
                   if None (default) the value stored in self.lamd is used
        Bex_val  : float,optional
                   value of exchange field
                   if None (default) the value stored in self.Bex_val is used
        Bex_mat_in : numpy 2d-array
                   spin operator coupled to Bex
                   if None (default) and self.rot_angle_Bex is None then this operator Bex_mat=2*S_z
                   if ----//---      and self.rot_angle_Bex is set then Bex_mat=2*S_rot, where S_rot is rotated by Euler angles self.rot_angle_Bex

        Returns
        -------
        eal : numpy real(self.inversion = True) or complex(self.inversion = True) 2d-array
              output atomic level matrix
        """

        if  CF_params == None:
            print '\ncalc_lev_pos: no input CF parameters, the previously fitted values are used'
            CF_params = self.Lkq_par

        if  E0 == None:
            print '\ncalc_lev_pos: no input value of uniform shift E0, the previously fitted value is used'
            E0 = self.E0

        if  lamd == None:
            print '\ncalc_lev_pos: no input value of SO parameter lambda, the previously fitted value is used'
            lamd = self.lamd

        if  Bex_val == None:
            print '\ncalc_lev_pos: no input value of exchange field Bex_val, the previously fitted value is used'
            Bex_val = self.Bex_val

        if  not isinstance(Bex_mat_in, numpy.ndarray) and self.rot_angle_Bex != None:
            print '\ncalc_lev_pos: no input exchange field spin-operator matrix Bex_mat_in, the Euler angles in rot_angle_Bex are used'

        l = self.l
        norb = 2 * l + 1
        nlmt = 2 * norb
        if real_mat:
            eal = numpy.zeros([nlmt, nlmt])
        else:
            eal = numpy.zeros([nlmt, nlmt], numpy.complex)

        # uniform term
        eal += E0*numpy.identity(nlmt)

        # put in CF contribution:
        C_mat = self.__calc_C_mat(l, print_mat=prn_lev)
        for icof in range(len(C_mat)):
            if self.spin_pol_B:
                eal[0:norb, 0:norb] += CF_params[icof * 2] * C_mat[icof]
                eal[norb:nlmt, norb:nlmt] += CF_params[icof * 2 + 1] * C_mat[icof]
            else:
                eal[0:norb, 0:norb] += CF_params[icof] * C_mat[icof]
                eal[norb:nlmt, norb:nlmt] += CF_params[icof] * C_mat[icof]

        # put in SO, order in spin (Up,DOWN), in m from -l to l

        if self.SO:
            H_SO = self.__calc_H_SO_mat(l, print_mat=prn_lev)
            eal += lamd * H_SO

        #add exchange field

        if self.Bex:
            if type(Bex_mat_in) is not numpy.ndarray:
                Bex_mat = self.__calc_Bex_mat(l=l, print_mat=prn_lev)
            else:
                Bex_mat = Bex_mat_in
            eal += Bex_mat * Bex_val

        if prn_lev == 1:
            print '\nfitted eal :'
            self.__print_mat(eal)
        return eal


    def print_cf(self, conv='Lkq', units='K'):
        r"""prints the resulting CF parameters as well as uniform shift, SO lambda and Bex.

           Parameters
           ----------
           conv : string, optional
                'Lkq' (default) prints CF parameters in Wybourne convention
                'Akq' prints CF parameters in Stevens (Akq<r^k>) convention
           units: string, optional
                  'K' (default) output CF parameters in Kelvins
                  'meV' in meV
                  'cm-1' in cm-1
        """
        n_cf = len(self.C_tens)
        print '\n\nUniform shift E0=%15.5f eV'%(self.E0)
        print '\nSO lambda = %9.5f eV'%(self.lamd)
        assert conv in ["Lkg", "Akq"], 'parameter conv=%s in print_cf not recognized'%conv
        if units == 'K':
            unit_fac = eV_to_K
        elif units == 'meV':
            unit_fac = 1000.0
        elif units == 'cm-1':
            unit_fac = eV_to_cm1
        else:
            print 'units=%s in print_cf is not recognized'%units
            raise ValueError
        print "\nCF parametes in %s convention and in units of %s"%(conv,units)
        if self.spin_pol_B:
            for sp in range(2):
                if sp == 0:
                    print '\n Spin UP:'
                else:
                    print '\n Spin DOWN:'
                for i in range(n_cf):
                    l = self.C_tens[i][0]
                    m = self.C_tens[i][1]
                    label = 'L%s%s'%(l, m) if conv == 'Lkq' else 'A%s%s<r^%s>'%(l, m, l)
                    CFP = self.Lkq_par[2 * i + sp].real if conv == 'Lkq' else self.Lkq_par[2 * i + sp].real * conv_to_A['%s_%s'%(l, m)]
                    print '%s:  %8.6f'%(label, CFP * unit_fac)
        else:
            for i in range(n_cf):
                l = self.C_tens[i][0]
                m = self.C_tens[i][1]
                label = 'L%s%s'%(l, m) if conv == 'Lkq' else 'A%s%s<r^%s>'%(l, m, l)
                CFP = self.Lkq_par[i].real if conv == 'Lkq' else self.Lkq_par[i].real*conv_to_A['%s_%s'%(l, m)]
                print '%s:  %8.6f'%(label, CFP * unit_fac)
        if self.Bex:
            print "\nExchange field Bex:"
            print"%8.6f  %8.2f (%s)  %8.2f (Tesla)"%(self.Bex_val, self.Bex_val * unit_fac, units, self.Bex_val * eV_to_K / Tesla_to_K)


###########################################
############## Private methods ############
###########################################

    def __print_mat(self,mat):
        dim=mat.shape
        comp_mat=isinstance(mat[0,0], complex)
        if comp_mat:
            if numpy.sum(numpy.abs(mat.imag)) < 1e-6: comp_mat=False
        l1=dim[0]
        l2=dim[1]
        for i in range(l1):
            str=''
            for j in range(l2):
                if not comp_mat:
                    str +='   %11.6f'%(mat[i,j].real)
                else:
                    str +='   %11.6f %11.6f'%(mat[i,j].real,mat[i,j].imag)
            print '%s'%(str)

    def __calc_C_mat(self,l,print_mat=1):
        '''compute Gaunt coeff. matrices C'''
        norb=2*l+1
        C_mat=[]
        for qm in self.C_tens:
            C_mat.append(numpy.zeros([norb,norb]))
            for m in range(-l,l+1):
                 mm=self.Y_order_m*m
                 str=' '
                 for m1 in range(-l,l+1):
                     mm1=self.Y_order_m*m1
                     #C_mat[-1][l+m,l+m1]=fortmod.gaunt(l,qm[0],l,m,-qm[1],m1)
                     C_mat[-1][l+mm,l+mm1]=self.__gaunt(l,qm[0],l,m,-qm[1],m1)
                     if qm[1]!=0 and self.inversion and qm[1]%2==0:
                         C_mat[-1][l+mm,l+mm1]+=self.__gaunt(l,qm[0],l,m,qm[1],m1)
                     elif qm[1]!=0 and self.inversion and qm[1]%2==1:
                         C_mat[-1][l+mm,l+mm1]-=self.__gaunt(l,qm[0],l,m,qm[1],m1)
            C_mat[-1] *= numpy.sqrt(4.0 *numpy.pi/(2*qm[0]+1))
            if print_mat==1 :
                if self.inversion:
                    print '\nT%s%s :'%(qm[0],qm[1])
                else:
                    print '\nC%s%s :'%(qm[0],qm[1])
                self.__print_mat(C_mat[-1])

        return C_mat


    def __Wigner3j(self,j1,j2,j3,m1,m2,m3):
    #======================================================================
    # Wigner3j.m by David Terr, Raytheon, 6-17-04
    #
    # Compute the Wigner 3j symbol using the Racah formula [1].
    #
    # Usage:
    # from wigner import Wigner3j
    # wigner = Wigner3j(j1,j2,j3,m1,m2,m3)
    #
    #  / j1 j2 j3 \
    #  |          |
    #  \ m1 m2 m3 /
    #
    # Reference: Wigner 3j-Symbol entry of Eric Weinstein's Mathworld:
    # http://mathworld.wolfram.com/Wigner3j-Symbol.html
    #======================================================================

        # Error checking
        if ( ( 2*j1 != floor(2*j1) ) | ( 2*j2 != floor(2*j2) ) | ( 2*j3 != floor(2*j3) ) | ( 2*m1 != floor(2*m1) ) | ( 2*m2 != floor(2*m2) ) | ( 2*m3 != floor(2*m3) ) ):
            print 'All arguments must be integers or half-integers.'
            return -1

        # Additional check if the sum of the second row equals zero
        if ( m1+m2+m3 != 0 ):
            #print '3j-Symbol unphysical'
            return 0

        if ( j1 - m1 != floor ( j1 - m1 ) ):
            #print '2*j1 and 2*m1 must have the same parity'
            return 0

        if ( j2 - m2 != floor ( j2 - m2 ) ):
            #print '2*j2 and 2*m2 must have the same parity'
            return; 0

        if ( j3 - m3 != floor ( j3 - m3 ) ):
            #print '2*j3 and 2*m3 must have the same parity'
            return 0

        if ( j3 > j1 + j2)  | ( j3 < abs(j1 - j2) ):
            #print 'j3 is out of bounds.'
            return 0

        if abs(m1) > j1:
            #print 'm1 is out of bounds.'
            return 0

        if abs(m2) > j2:
            #print 'm2 is out of bounds.'
            return 0

        if abs(m3) > j3:
            #print 'm3 is out of bounds.'
            return 0

        t1 = j2 - m1 - j3
        t2 = j1 + m2 - j3
        t3 = j1 + j2 - j3
        t4 = j1 - m1
        t5 = j2 + m2

        tmin = max( 0, max( t1, t2 ) )
        tmax = min( t3, min( t4, t5 ) )
        tvec = arange(tmin, tmax+1, 1)

        wigner = 0

        for t in tvec:
            wigner += (-1)**t / ( factorial(t) * factorial(t-t1) * factorial(t-t2) * factorial(t3-t) * factorial(t4-t) * factorial(t5-t) )

        return wigner * (-1)**(j1-j2-m3) * sqrt( factorial(j1+j2-j3) * factorial(j1-j2+j3) * factorial(-j1+j2+j3) / factorial(j1+j2+j3+1) * factorial(j1+m1) * factorial(j1-m1) * factorial(j2+m2) * factorial(j2-m2) * factorial(j3+m3) * factorial(j3-m3) )

    def __gaunt(self,l,l1,l2,m,m1,m2):
        w=self.__Wigner3j(l,l1,l2,-m,m1,m2)
        w0=self.__Wigner3j(l,l1,l2,0,0,0)
        res=numpy.power(-1,m)*numpy.sqrt((2*l+1.0)*(2*l1+1.0)*(2*l2+1.0)/(4.0*numpy.pi))*w*w0
        return res


    def __calc_Bex_mat(self,l,print_mat=1):
        norb=2*l+1
        nlmt=2*norb
        Bex_mat=numpy.zeros([nlmt,nlmt],numpy.complex_)
        spin_mat=numpy.zeros([2,2],numpy.complex_)
        spin_mat[0,0]=2*self.sp_arr[0]
        spin_mat[1,1]=2*self.sp_arr[1]
        if self.rot_angle_Bex != None :
            spin_rot_mat_init=numpy.zeros([2,2],numpy.complex_)
            alp=self.rot_angle_Bex[0]
            bet=self.rot_angle_Bex[1]
            gam=self.rot_angle_Bex[2]
            spin_rot_mat_init[0,0]=numpy.cos(bet/2.0)*numpy.exp(1j*(alp+gam)/2.0)
            spin_rot_mat_init[1,1]=spin_rot_mat[0,0].conjugate()
            spin_rot_mat_init[0,1]=numpy.sin(bet/2.0)*numpy.exp(1j*(alp-gam)/2.0)
            spin_rot_mat_init[1,0]=-spin_rot_mat[0,1].conjugate()
            spin_rot_mat=numpy.zeros([2,2],numpy.complex_)
            for i in range(2):
                for j in range(2):
                    ind1=int(round(0.5-self.sp_arr[i]))
                    ind2=int(round(0.5-self.sp_arr[j]))
                    spin_rot_mat[i,j]=spin_rot_mat_init[ind1,ind2]
            mat_tmp=numpy.dot(spin_mat,spin_rot_mat)
            spin_mat=numpy.dot(spin_rot_mat.transpose().conjugate(),mat_tmp)
        unit_lm=numpy.identity(norb)
        Bex_mat[0:norb,0:norb]=unit_lm*spin_mat[0,0]
        Bex_mat[norb:nlmt,0:norb]=unit_lm*spin_mat[1,0]
        Bex_mat[0:norb,norb:nlmt]=unit_lm*spin_mat[0,1]
        Bex_mat[norb:nlmt,norb:nlmt]=unit_lm*spin_mat[1,1]
        if print_mat==1 :
            print '\nBex_mat :'
            self.__print_mat(Bex_mat)
        return Bex_mat

    def __calc_H_SO_mat(self,l,print_mat=1):
        ndim=2*(2*l+1)
        nlm=ndim/2
        H_SO=numpy.zeros([ndim,ndim])
        for m in range(-l,l+1):
            mm=self.Y_order_m*m
            H_SO[l+mm,l+mm]=m*self.sp_arr[0]
            H_SO[nlm+l+mm,nlm+l+mm]=m*self.sp_arr[1]
        sp=int(round(self.sp_arr[0]+0.5)) # 1 if first spin in sp_arr is UP
        for m in range(-l,l):
            mm=self.Y_order_m*m
            sqfac=numpy.sqrt((1.0+l+m)*(l-m))
            H_SO[nlm*sp+l+mm+self.Y_order_m,nlm*(1-sp)+l+mm]=0.5*sqfac
            H_SO[nlm*(1-sp)+l+mm,nlm*sp+l+mm+self.Y_order_m]=0.5*sqfac
        if print_mat==1 :
            print '\nSO_mat :'
            self.__print_mat(H_SO)
        return H_SO
