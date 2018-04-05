#
# This script calculates CF parameters from the input level positions 
#
import numpy
from cf_param_fit import *
File_level_pos='atomic_levels_matrix'
SO=True
Bex=True 
spin_pol_B=True
#spin_pol_B=False
# list of orb/mag. quant.numbers [k,q] for seeked CF parameters 
CF_prms=[[2,0],[4,0],[6,0],[6,6]]


CF=cf_param_fit(CF_prms,SO=SO,Bex=Bex,spin_pol_B=spin_pol_B,complex_eal=True)
E0,lambd,CF_Ham,Bex_val=CF.fit(File_level_pos,prn_lev=0)

CF.print_cf(conv='Akq',units='K')
    


