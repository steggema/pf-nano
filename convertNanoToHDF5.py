#!/usr/bin/env python
import pdb
import optparse
import sys
import uproot
import numpy as np
import h5py
import progressbar
import itertools

widgets = [
    progressbar.SimpleProgress(
    ), ' - ', progressbar.Timer(), ' - ', progressbar.Bar(), ' - ', progressbar.AbsoluteETA()
]


def z_rotation(vector, theta):
    """Rotates x,y vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    rotated = np.dot(R, vector)
    return rotated[0], rotated[1]


# configuration
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-i', '--input', dest='input',
                  help='input file', default='', type='string')
parser.add_option('-o', '--output', dest='output',
                  help='output file', default='', type='string')
parser.add_option('--keeplepton', dest='keeplepton',
                  help='keep lepton (scaled to pt=1)', default=False, action='store_true')
parser.add_option('--lepframe', dest='lepframe',
                  help='rotate px,py to lepton frame', default=False, action='store_true')
parser.add_option('--nopuppi', dest='nopuppi',
                  help='do not store puppi weights', default=False, action='store_true')
(opt, args) = parser.parse_args()

if opt.input == '' or opt.output == '':
    sys.exit('Need to specify input and output files!')

met_flavours = ['', 'Calo', 'Chs', 'NoPU',
                'Puppi', 'PU', 'PUCorr', 'Raw', 'Tk']
met_branches = [
    m+t for m, t in itertools.product(met_flavours, ['MET_phi', 'MET_pt', 'MET_sumEt'])]

other_branches = ['nJet', 'Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass',
                'fixedGridRhoFastjetAll', 'fixedGridRhoFastjetCentralCalo',
                'nMuon', 'Muon_tightId', 'Muon_pt', 'Muon_eta', 'Muon_phi',
                'nPF', 'PF_*',
                ]

upfile = uproot.open(opt.input)

tree = upfile['Events'].arrays(met_branches + other_branches)

# Select events with two tight ID muons and keep only those
sel_tight_muon = (tree[b'Muon_tightId'] == 1)
for key in [b for b in tree.keys() if b.startswith(b'Muon')]:
    tree[key] = tree[key][sel_tight_muon]

tree[b'nMuon'] = tree[b'Muon_tightId'].count()

tree = {k:v[tree[b'nMuon'] == 2] for k, v in tree.items()}


# Remove PF candidates corresponding to the two muons
for i_muon in [0, 1]:
    dphi = tree[b'PF_phi'] - tree[b'Muon_phi'][:,i_muon]
    deta2 = (tree[b'PF_eta'] - tree[b'Muon_eta'][:,i_muon])**2
    
    dphi = dphi - 2*np.pi * (dphi > np.pi) + 2*np.pi * (dphi < -np.pi)
    dr2 = dphi**2 + deta2

    remove_match = dr2 > 0.000001

    for key in [b for b in tree.keys() if b.startswith(b'PF')]:
        tree[key] = tree[key][remove_match]

# TODO: Add regular and Puppi px, py for each candidate

tree[b'nPF'] = tree[b'PF_pt'].count()

# general setup
maxNPF = 4000
nFeatures = 15
normFac = 10

maxEntries = len(tree[b'nPF'])
# # input PF candidates
# X = np.zeros(shape=(maxEntries, maxNPF, nFeatures), dtype=float, order='F')
# # W recoil
# Y = np.zeros(shape=(maxEntries, 2), dtype=float, order='F')
# Simple recoil estimators
Z = np.zeros(shape=(maxEntries, 6), dtype=float, order='F')


pf_keys = [b for b in tree.keys() if b.startswith(b'PF')]
for key in pf_keys:
    tree[key] = tree[key].pad(maxNPF)
    tree[key] = np.asarray(tree[key].tolist(), dtype=float)

X = np.stack([tree[key] for key in pf_keys], axis=2)
print('Shape of X', X.shape)

muon_px = np.cos(tree[b'Muon_phi']) * tree[b'Muon_pt']
muon_py = np.sin(tree[b'Muon_phi']) * tree[b'Muon_pt']

dimuon_px = muon_px[:,1] + muon_px[:,0]
dimuon_py = muon_py[:,1] + muon_py[:,0]
Y = np.stack([dimuon_px, dimuon_py], axis=1)

for met in met_flavours:
    met = str.encode(met)
    tree[met+b'MET_px'] = np.cos(tree[met+b'MET_phi']) * tree[met+b'MET_pt']
    tree[met+b'MET_py'] = np.sin(tree[met+b'MET_phi']) * tree[met+b'MET_pt']

Z = np.stack([tree[str.encode(met)+var] for met in met_flavours for var in [b'MET_px', b'MET_py', b'MET_sumEt']])

with h5py.File(opt.output, 'w') as h5f:
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('Y', data=Y)
    h5f.create_dataset('Z', data=Z)
