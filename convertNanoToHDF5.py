#!/usr/bin/env python
import optparse
import sys
import uproot
import numpy as np
import h5py
import itertools
import os.path

# Define embeddings here so we're sure that the same integers always represent
# the same categories
d_embedding = {
    b'PF_charge':{-1.0: 0, 0.0: 1, 1.0: 2},
    b'PF_pdgId':{-211.0: 0, -13.0: 1, -11.0: 2, 0.0: 3, 1.0: 4, 2.0: 5, 11.0: 6, 13.0: 7, 22.0: 8, 130.0: 9, 211.0: 10},
    b'PF_fromPV':{0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3}
}



# configuration
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-i', '--input', dest='input',
                  help='input files', default='', type='string')
parser.add_option('-o', '--output', dest='output',
                  help='output file', default='', type='string')
parser.add_option('--nopf', dest='nopf',
                  help='do not store PF candidates', default=False, action='store_true')
parser.add_option('--append', dest='append',
                  help='Append to existing file', default=False, action='store_true')
parser.add_option('--n_leptons', dest='n_leptons',
                  help='How many ID/iso muons + electrons are required', default=2)
parser.add_option('--norm_factor', dest='norm_factor',
                  help='Divide pT/energy variables by this factor', default=50.)
parser.add_option('--compression', dest='compression',
                  help='compression', default='gzip')

(opt, args) = parser.parse_args()

if opt.input == '' or opt.output == '':
    sys.exit('Need to specify input and output files!')

met_flavours = ['',  'Chs', 'NoPU', 'Puppi', 'PU', 'PUCorr', 'Raw'] #'Calo', 'Tk'
met_branches = [
    m+t for m, t in itertools.product(met_flavours, ['MET_phi', 'MET_pt', 'MET_sumEt'])]

other_branches = ['nJet', 'Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass',
                  'fixedGridRhoFastjetAll', 'fixedGridRhoFastjetCentralCalo',
                  'nMuon', 'Muon_tightId', 'Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_pfRelIso03_all',
                  'nElectron', 'Electron_mvaFall17V2Iso_WP80', 'Electron_pt', 'Electron_eta', 'Electron_phi',
                  'nPF', 'PF_*',
                  'GenMET_pt', 'GenMET_phi'
                 ]

upfile = uproot.open(opt.input)

tree = upfile['Events'].arrays(met_branches + other_branches)

# normalisation and other transformations (none for the time being since
# everything looks reasonably regular)
for branch in tree.keys():
    if b'pt' in branch:
        tree[branch] = tree[branch]/opt.norm_factor


# Select events with "n_leptons" tight ID muons + electrons and keep only those
sel_tight_muon = (tree[b'Muon_tightId'] == 1) & (tree[b'Muon_pfRelIso03_all'] < 0.15) & (tree[b'Muon_pt'] > 20./opt.norm_factor)
for key in [b for b in tree.keys() if b.startswith(b'Muon')]:
    tree[key] = tree[key][sel_tight_muon]

tree[b'nMuon'] = tree[b'Muon_tightId'].count()

sel_tight_electron = (tree[b'Electron_mvaFall17V2Iso_WP80'] == 1) & (tree[b'Electron_pt'] > 20./opt.norm_factor)
for key in [b for b in tree.keys() if b.startswith(b'Electron')]:
    tree[key] = tree[key][sel_tight_electron]

tree[b'nElectron'] = tree[b'Electron_mvaFall17V2Iso_WP80'].count()

tree[b'nLepton'] = tree[b'nMuon'] + tree[b'nElectron']
tree[b'Lepton_phi'] = np.stack([tree[b'Muon_phi'], tree[b'Electron_phi']], axis=1)
tree[b'Lepton_eta'] = np.stack([tree[b'Muon_eta'], tree[b'Electron_eta']], axis=1)
tree[b'Lepton_pt'] = np.stack([tree[b'Muon_pt'], tree[b'Electron_pt']], axis=1)

tree = {k:v[tree[b'nLepton'] == opt.n_leptons] for k, v in tree.items()}

for key in [b'Lepton_phi', b'Lepton_eta', b'Lepton_pt']:
    for i in range(len(tree[key])):
        tree[key][i] = np.concatenate(tree[key][i])
    tree[key] = tree[key].astype(np.float32)

lepton_px = np.cos(tree[b'Lepton_phi']) * tree[b'Lepton_pt']
lepton_py = np.sin(tree[b'Lepton_phi']) * tree[b'Lepton_pt']

dimuon_px = sum(lepton_px[:,i] for i in range(opt.n_leptons)) #  lepton_px[:,1] + lepton_px[:,0]
dimuon_py = sum(lepton_py[:,i] for i in range(opt.n_leptons))

dimuon_plus_met_px = dimuon_px + np.cos(tree[b'GenMET_phi']) * tree[b'GenMET_pt']
dimuon_plus_met_py = dimuon_py + np.sin(tree[b'GenMET_phi']) * tree[b'GenMET_pt']

Y = np.stack([dimuon_plus_met_px, dimuon_plus_met_py], axis=1)

for met in met_flavours:
    met = str.encode(met)
    if met == b'PU':
        print('MET flavour PU')
        tree[met+b'MET_px'] = np.cos(tree[met+b'MET_phi']) * tree[met+b'MET_pt']
        tree[met+b'MET_py'] = np.sin(tree[met+b'MET_phi']) * tree[met+b'MET_pt']
    else:
        tree[met+b'MET_px'] = np.cos(tree[met+b'MET_phi']) * tree[met+b'MET_pt'] + dimuon_px
        tree[met+b'MET_py'] = np.sin(tree[met+b'MET_phi']) * tree[met+b'MET_pt'] + dimuon_py
        for i in range(opt.n_leptons):
            tree[met+b'MET_sumEt'] -= tree[b'Lepton_pt'][:,i]

event_vars = [str.encode(met)+var for met in met_flavours for var in [b'MET_px', b'MET_py', b'MET_sumEt']]
event_vars += [b'fixedGridRhoFastjetAll', b'fixedGridRhoFastjetCentralCalo']

Z = np.stack([tree[ev_var] for ev_var in event_vars], axis=1)

if not opt.nopf:
    # Remove PF candidates corresponding to the two muons
    for i_muon in range(opt.n_leptons):
        dphi = tree[b'PF_phi'] - tree[b'Lepton_phi'][:,i_muon]
        deta2 = (tree[b'PF_eta'] - tree[b'Lepton_eta'][:,i_muon])**2
        dphi = dphi - 2*np.pi * (dphi > np.pi) + 2*np.pi * (dphi < -np.pi)
        dr2 = dphi**2 + deta2

        remove_match = dr2 > 0.000001

        for key in [b for b in tree.keys() if b.startswith(b'PF')]:
            tree[key] = tree[key][remove_match]

    tree[b'nPF'] = tree[b'PF_pt'].count()

    # general setup
    maxNPF = 4500

    maxEntries = len(tree[b'nPF'])

    pf_keys = [b for b in tree.keys() if b.startswith(b'PF')]
    for key in pf_keys:
        tree[key] = tree[key].pad(maxNPF)
        try:
            the_list = tree[key].tolist()
            tree[key] = np.asarray(the_list, dtype=float)
        except ValueError:
            import pdb; pdb.set_trace()
        np.nan_to_num(tree[key], copy=False)
    tree[b'PF_px'] = np.cos(tree[b'PF_phi']) * tree[b'PF_pt']
    tree[b'PF_py'] = np.sin(tree[b'PF_phi']) * tree[b'PF_pt']
    pf_keys += [b'PF_px', b'PF_py']
    pf_keys = [key for key in pf_keys if key not in [b'PF_phi', b'PF_puppiWeightNoLep']] ##, b'PF_pt' <-- this may still help, e.g. for weighting in certain phase space
    pf_keys_categorical = [b'PF_charge', b'PF_pdgId', b'PF_fromPV']
    for key in pf_keys_categorical:
        if key not in d_embedding:
            vals = sorted(set(tree[key].flatten()))
            d_embedding[key] = {val:i for i, val in enumerate(vals)}
        tree[key] = np.vectorize(d_embedding[key].get)(tree[key])
    X_c = np.stack([tree[key] for key in pf_keys_categorical], axis=2)
    
    # Remove unused arrays
    for key in pf_keys_categorical:
        tree[key] = None
    print('Shape of X_c', X_c.shape)


    pf_keys = [key for key in pf_keys if key not in pf_keys_categorical]
    X = np.stack([tree[key] for key in pf_keys], axis=2)
    for key in pf_keys:
        tree[key] = None
    for i in range(X.shape[2]): # iteratively to not exceed memory
        X[:,:,i][np.where(np.abs(X[:,:,i]) > 1e+6)] = 0.
    # X[np.where(np.abs(X) > 1e+6)] = 0. # Remove outliers
    print('Shape of X', X.shape)



def nonify_first(shape):
    return tuple([None if i == 0 else el for i, el in enumerate(shape)])

if not opt.append or not os.path.isfile(opt.output):
    with h5py.File(opt.output, 'w') as h5f:
        if not opt.nopf:
            h5f.create_dataset('X', data=X, compression=opt.compression, chunks=256, maxshape=nonify_first(X.shape))
            h5f.create_dataset('X_c', data=X_c, compression=opt.compression, chunks=256, maxshape=nonify_first(X.shape))
        h5f.create_dataset('Y', data=Y, compression=opt.compression, chunks=256, maxshape=nonify_first(Y.shape))
        h5f.create_dataset('Z', data=Z, compression=opt.compression, chunks=256, maxshape=nonify_first(Z.shape))
        print('Finished')
        print(Y.shape)
        print(Z.shape)
else:
    with h5py.File(opt.output, 'a') as h5f:
        if not opt.nopf:
            h5f['X'].resize(h5f['X'].shape[0] + X.shape[0], axis=0)
            h5f['X'][-X.shape[0]:] = X
            h5f['X_c'].resize(h5f['X_c'].shape[0] + X_c.shape[0], axis=0)
            h5f['X_c'][-X_c.shape[0]:] = X_c

        h5f['Y'].resize(h5f['Y'].shape[0] + Y.shape[0], axis=0)
        h5f['Y'][-Y.shape[0]:] = Y
        h5f['Z'].resize(h5f['Z'].shape[0] + Z.shape[0], axis=0)
        h5f['Z'][-Z.shape[0]:] = Z
        print('Finished')
        print(h5f['Y'].shape)
        print(h5f['Z'].shape)
