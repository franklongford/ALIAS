import numpy as np
import scipy as sp
import time, sys, os
from scipy import constants as con
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import utilities as ut
import density as den
import mdtraj as md

root = '/local/scratch/sam5g13/AMBER/TEST'
model = 'SPCE'
T = 298 
cutoff = 10

nsite, AT, Q, M, LJ, mol_sigma = ut.get_param(model)

mol_sigma_rdf, r_max = den.radial_dist(root, model, nsite, M, 0)
