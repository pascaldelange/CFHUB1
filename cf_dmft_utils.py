import copy
import numpy

import pytriqs.utility.mpi as mpi

class cf_dmft_utils:
    """
    This class contains all the functions required to perform the spin and orbital
    moment averaging during the DFT+HUB I calculation.
    It contains a pointer to the instance of the Sum-k class from DFT-tools to access
    the parameters.
    """
    def __init__(self, SK):
        self.SK = SK

    def spin_orb_matrix(self, L):
        """
        Constructs the spin and orbital momentum matrices.
        L = 3 for a shell of f orbitals.
        """
        Nlm = 2 * L + 1
        Norb = 2 * Nlm
        spinmat = numpy.zeros([Norb, Norb], numpy.complex_)
        orbmat = numpy.zeros([Norb, Norb], numpy.complex_)
        mat1 = numpy.identity(Nlm)
        spinmat[0:Nlm, 0:Nlm] = mat1
        spinmat[Nlm:2 * Nlm, Nlm:2 * Nlm] = -mat1
        orb1 = numpy.zeros([Nlm, Nlm], numpy.complex_)
        for m in range(L):
            orb1[m, m] = L + 1 - Nlm + m
            orb1[Nlm - m - 1, Nlm - m - 1] = Nlm - L - 1 - m
            orbmat[:Nlm, :Nlm] = orb1
            orbmat[Nlm:2 * Nlm, Nlm:2 * Nlm] = orb1
        return spinmat, orbmat

    def set_timerev(self, L):
        """
        Constructs the time reversal matrix for a shell of electrons with secondary
        quantum number L (L=3 for f electrons).
        """
        Nlm = 2 * L + 1
        Norb = 2 * Nlm
        # set time-reversal operation
        timerev = numpy.zeros([Norb, Norb], numpy.complex_)
        for lm in range(Nlm):
            m = L - Nlm + lm + 1
            lm1 = 2 * L - lm
            timerev[lm + Nlm, lm1] = numpy.power(-1.0, m)
            timerev[lm, Nlm + lm1] = numpy.power(-1.0, m)
        return timerev

    def upfold_dm(self, ik, dm, blocks=['ud']):
        """
        Upfolds the density matrix into the Bloch basis for a given k-point ik.
        Transform the CTQMC blocks (that live in the projected space) to the full
        matrix (in the space of Bloch functions).
        This function is heavily inspired from the upfold function of DFT-tools,
        acting on Green's functions.

        Parameters:
        - ik: Brillouin zone index (in the reduced Brillouin zone).
        - dm: The density matrix in the Wannier space. Dict of numpy arrays.
        - blocks: spin blocks of the matrix. Only ['ud'] (for 'up-down', for up
                    and down spins mixed by SO-coupling) is supported now.
        """
        dmf = [{} for i in range(self.SK.n_corr_shells)]
        for icrsh in xrange(self.SK.n_corr_shells):
            s = self.SK.corr_to_inequiv[icrsh]
            # s is the index of the inequivalent shell corresponding to icrsh
            for ibl in blocks:
                dmf[icrsh][self.SK.inequiv_to_corr[s]] = dm[s][ibl].copy()

        # rotation from the local to global coordinate system:
        if self.SK.use_rotations:
            for icrsh in xrange(self.SK.n_corr_shells):
                for sig, d in dmf[icrsh].iteritems():
                    d = numpy.dot(self.SK.rot_mat[icrsh].conjugate(), d)
                    d = numpy.dot(d, self.SK.rot_mat[icrsh].transpose())
        # upfolding
        ntoi = self.SK.spin_names_to_ind[self.SK.SO]
        bln = self.SK.spin_block_names[self.SK.SO]
        dm_bloch = {}
        for block in bln:
            band_number = self.SK.n_orbitals[ik][ntoi[block]]
            dm_bloch[block] = numpy.zeros([band_number, band_number], numpy.complex_)
            for icrsh in xrange(self.SK.n_corr_shells):
                proj_mat = self.SK.proj_mat[ik, ntoi[block], icrsh, :, :band_number]
                dm_bloch[block] += numpy.dot(proj_mat.conjugate().transpose(),
                                            numpy.dot(dmf[icrsh][ntoi[block]], proj_mat))

        return dm_bloch

    def downfold_dm(self, ik, icrsh, dm, blocks=['ud']):
        """
        Down the density matrix into the Wannier basis for a given k-point ik.
        Transform the full blocks (that live in the Bloch space) to the projected
        matrix (in the space of Wannier functions).
        This function is heavily inspired from the downfold function of DFT-tools,
        acting on Green's functions.

        Parameters:
        - ik: Brillouin zone index (in the reduced Brillouin zone).
        - icrsh: index of the correlated shell, as listed in the case.indmftpr file.
        - dm: The density matrix in the Bloch space. Dict of numpy arrays.
        - blocks: spin blocks of the matrix. Only ['ud'] (for 'up-down', for up
                    and down spins mixed by SO-coupling) is supported now.
        """
        dm_wann = {}
        ntoi = self.SK.spin_names_to_ind[self.SK.SO]
        for bl in blocks:
            isp = ntoi[bl]
            n_orb = self.SK.n_orbitals[ik, isp]
            proj_mat = self.SK.proj_mat[ik, isp, icrsh, :, :n_orb]
            dm_wann[bl] = numpy.dot(numpy.dot(proj_mat, dm[bl][ik]), proj_mat.conjugate().transpose())
        return dm_wann

    def apply_time_reversal(self, dN, timerev_matrix):
        """
        Apply time reversal symmetry on the local, projected part of the density matrix
        corresponding to the correlated subspace.
        - Takes as parameters:
            * dN, a density matrix in Bloch space
            * timerev_matrix, the time reversal symmetry matrix in the local, projected
            subspace

        - Returns:
            A copy of dN where the time reversal symmetry has been applied on the local,
            projected orbitals
        """
        if not self.SK.SO == 1:
            raise NotImplementedError("Must be SO calculation in order to do this. "
                                        "Non-SO calculations are not implemented yet.")

        Norb = self.SK.corr_shells[0]['dim']
        dN_out = copy.deepcopy(dN)
        bln = self.SK.spin_block_names[self.SK.SO] # bln = ["ud"]
        dm_tmrv = {}
        # first, calculate the density matrix after applying time reversal symmetry for each k point
        for ik in range(self.SK.n_k):
            ddm=[]
            for ish in range(self.SK.n_corr_shells):
                dm = self.downfold_dm(ik, ish, dN_out, bln)
                for ib in bln:
                    dm_tmp = numpy.dot(timerev_matrix, dm[ib].conjugate())
                    dm_tmrv[ib] = numpy.dot(dm_tmp, timerev_matrix)*0.5
                    dm_tmrv[ib] -= dm[ib]*0.5
                ddm.append(dm_tmrv)
            dmbloch = self.upfold_dm(ik, ddm)
            for ib in bln:
                dN_out[ib][ik] += dmbloch[ib]

        # next, compute the local (k summed) density matrix, after applying time reversal symmetry
        # and print the result
        dm_list = []
        for ish in range(self.SK.n_corr_shells):
            dm = {}
            for ib in bln:
                dm[ib] = numpy.zeros([Norb, Norb], numpy.complex_)
                for ik in range(self.SK.n_k):
                    dav = self.downfold_dm(ik, ish, dN_out, bln)
                    dm[ib] += dav[ib] * self.SK.bz_weights[ik]
            dm_list.append(dm)

        if self.SK.symm_op != 0:
            dm_list = self.SK.symmcorr.symmetrize(dm_list)

        for ish in range(self.SK.n_corr_shells):
            dm = dm_list[ish]
            if mpi.is_master_node():
                print '\n\nAveraged dens. mat for site {}: '.format(ish)
                for i in range(Norb):
                    string=' '
                    for j in range(Norb):
                        string += '%6.2f%6.2f  '%(dm['ud'][i, j].real, dm['ud'][i, j].imag)
                    print string

        return dN_out

    def save_dens(self, deltaN, beta, LDAFilename):
        """
        Updates the .qdmftup and .qdmftdn files used by TRIQS DFT tools, after
        time reversal symmetry has been applied.
        Parameters:
        - deltaN is the correction to the density matrix, calculated within the
        DFT+Hubbard I method
        - beta is the inverse temperature, in eV^-1
        - LDAFilename is the prefix of all DFT files (and also the name of the folder)
        """
        from shutil import copyfile
        # save density matrix to file
        if (mpi.is_master_node()):
            f=open(LDAFilename+'.qdmftup','w')
            # write chemical potential (in Rydberg):
            f.write("%.14f\n"%(self.SK.chemical_potential / self.SK.energy_unit))
            f.write("%.14f\n"%(beta * self.SK.energy_unit))
            # write beta in ryderg-1
            for ik in range(self.SK.n_k):
                f.write("%s\n"%self.SK.n_orbitals[ik][0])
                for inu in range(self.SK.n_orbitals[ik][0]):
                    for imu in range(self.SK.n_orbitals[ik][0]):
                        f.write("%.14f  %.14f "%(deltaN['ud'][ik][inu, imu].real, deltaN['ud'][ik][inu, imu].imag))
                    f.write("\n")
                f.write("\n")
            f.close()
            copyfile(LDAFilename + '.qdmftup', LDAFilename + '.qdmftdn')
