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
parser.add_option('--n_leptons_subtract', dest='n_leptons_subtract',
                  help='How many of the ID/iso muons + electrons to subtract from the METs and PFCandidate list', default=2)
parser.add_option('--norm_factor', dest='norm_factor',
                  help='Divide pT/energy variables by this factor', default=50.)
parser.add_option('--compression', dest='compression',
                  help='compression', default='gzip')
parser.add_option('--test', dest='test', help='Test mode processing 1000 events',
                  default=False, action='store_true')
parser.add_option('--jets', dest='jets', help='Add jets',
                  default=False, action='store_true')
parser.add_option('--n_max', dest='n_max', help='Maximum number of events',
                  default=-1)
parser.add_option('--data', dest='data', help='Process data',
                  default=False, action='store_true')

(opt, args) = parser.parse_args()

if opt.input == '' or opt.output == '':
    sys.exit('Need to specify input and output files!')

if opt.compression in ['None', 'none']:
    opt.compression = None

opt.n_leptons = int(opt.n_leptons)
opt.n_leptons_subtract = int(opt.n_leptons_subtract)

# met_flavours = ['',  'Chs', 'NoPU', 'Puppi', 'PU', 'PUCorr', 'Raw'] #'Calo', 'Tk'
met_flavours = ['',  'Puppi', 'Raw'] #'Calo', 'Tk'
met_branches = [
    m+t for m, t in itertools.product(met_flavours, ['MET_phi', 'MET_pt', 'MET_sumEt'])]

other_branches = ['fixedGridRhoFastjetAll', 'fixedGridRhoFastjetCentralCalo',
                  'nMuon', 'Muon_tightId', 'Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_mass', 'Muon_pfRelIso03_all',
                  'nElectron', 'Electron_mvaFall17V2Iso_WP80', 'Electron_pt', 'Electron_eta', 'Electron_phi', 'Electron_mass',
                  'nPF', 'PF_*'
                 ]

if opt.jets:
    other_branches += ['nJet', 'Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass']

if not opt.data:
    other_branches += ['GenMET_pt', 'GenMET_phi']
    if opt.jets:
        other_branches += ['nGenJet', 'GenJet_pt', 'GenJet_eta', 'GenJet_phi', 'GenJet_mass']

upfile = uproot.open(opt.input)

tree = upfile['Events'].arrays(met_branches + other_branches, entrystop=int(opt.n_max))

if opt.test:
    for branch in tree.keys():
        tree[branch] = tree[branch][0:1000]

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
tree[b'Lepton_mass'] = np.stack([tree[b'Muon_mass'], tree[b'Electron_mass']], axis=1)

tree = {k:v[tree[b'nLepton'] == opt.n_leptons] for k, v in tree.items()}

print(opt.n_leptons, 'lepton requirement applied')

for key in [b'Lepton_phi', b'Lepton_eta', b'Lepton_pt', b'Lepton_mass']:
    for i in range(len(tree[key])):
        tree[key][i] = np.concatenate(tree[key][i])
    tree[key] = tree[key].astype(np.float32)



if opt.data:
    lepton_px = np.cos(tree[b'Lepton_phi']) * tree[b'Lepton_pt']
    lepton_py = np.sin(tree[b'Lepton_phi']) * tree[b'Lepton_pt']

    dimuon_px = sum(lepton_px[:, i] for i in range(opt.n_leptons_subtract)) if opt.n_leptons_subtract else 0.#  lepton_px[:,1] + lepton_px[:,0]
    dimuon_py = sum(lepton_py[:, i] for i in range(opt.n_leptons_subtract)) if opt.n_leptons_subtract else 0.
    lepton_energy = np.sqrt(np.maximum(0., (tree[b'Lepton_pt']*np.cosh(tree[b'Lepton_eta']))**2 + tree[b'Lepton_mass']**2))
    lepton_pz = tree[b'Lepton_pt']*np.sinh(tree[b'Lepton_eta'])
    dimuon_pz = sum(lepton_pz[:, i] for i in range(opt.n_leptons_subtract))if opt.n_leptons_subtract else 0.
    dimuon_energy = sum(lepton_energy[:, i] for i in range(opt.n_leptons_subtract))if opt.n_leptons_subtract else 0.

    dimuon_mass = np.sqrt(dimuon_energy**2 - dimuon_pz**2 - dimuon_py**2 - dimuon_px**2)
    z_mass = np.logical_and(dimuon_mass > 80./opt.norm_factor, dimuon_mass < 100./opt.norm_factor)
    dimuon_pt = np.sqrt(dimuon_px**2 + dimuon_py**2)
    z_mass_pt = np.logical_and(z_mass, dimuon_pt < 100./opt.norm_factor)
    tree = {k:v[z_mass_pt] for k, v in tree.items()}

lepton_px = np.cos(tree[b'Lepton_phi']) * tree[b'Lepton_pt']
lepton_py = np.sin(tree[b'Lepton_phi']) * tree[b'Lepton_pt']

dimuon_px = sum(lepton_px[:, i] for i in range(opt.n_leptons_subtract)) if opt.n_leptons_subtract else 0.#  lepton_px[:,1] + lepton_px[:,0]
dimuon_py = sum(lepton_py[:, i] for i in range(opt.n_leptons_subtract)) if opt.n_leptons_subtract else 0.

if opt.data:
    dimuon_plus_met_px = dimuon_px
    dimuon_plus_met_py = dimuon_py
else:
    dimuon_plus_met_px = np.cos(tree[b'GenMET_phi']) * tree[b'GenMET_pt'] + dimuon_px
    dimuon_plus_met_py = np.sin(tree[b'GenMET_phi']) * tree[b'GenMET_pt'] + dimuon_py

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
    print('Remove PF candidates corresponding to the two muons')
    for i_muon in range(opt.n_leptons_subtract):
        dphi = tree[b'PF_phi'] - tree[b'Lepton_phi'][:, i_muon]
        deta2 = (tree[b'PF_eta'] - tree[b'Lepton_eta'][:, i_muon])**2
        dphi = dphi - 2*np.pi * (dphi > np.pi) + 2*np.pi * (dphi < -np.pi)
        dr2 = dphi**2 + deta2

        remove_match = dr2 > 0.000001

        for key in [b for b in tree.keys() if b.startswith(b'PF')]:
            tree[key] = tree[key][remove_match]

    tree[b'nPF'] = tree[b'PF_pt'].count()

    # general setup
    maxNPF = 4500

    maxEntries = len(tree[b'nPF'])

    print('Prepare PF tree with', maxNPF, 'entries')

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
    if opt.jets:
        tree[b'PF_energy'] = np.sqrt(np.maximum(0., (tree[b'PF_pt']*np.cosh(tree[b'PF_eta']))**2 + tree[b'PF_mass']**2))
        tree[b'PF_pz'] = tree[b'PF_pt']*np.sinh(tree[b'PF_eta'])
        # Add pz and energy
        pf_keys += [b'PF_energy', b'PF_pz']
    pf_keys += [b'PF_px', b'PF_py']
    pf_keys = [key for key in pf_keys if key not in [b'PF_phi', b'PF_puppiWeightNoLep', b'PF_PFIdx']] ##, b'PF_pt' <-- this may still help, e.g. for weighting in certain phase space
    print('PF keys', pf_keys)
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
    print('Final PF keys', pf_keys)
    X = np.stack([tree[key] for key in pf_keys], axis=2)
    for key in pf_keys:
        tree[key] = None
    for i in range(X.shape[2]): # iteratively to not exceed memory
        X[:,:,i][np.where(np.abs(X[:,:,i]) > 1e+6)] = 0.
    # X[np.where(np.abs(X) > 1e+6)] = 0. # Remove outliers
    print('Shape of X', X.shape)

if opt.jets:
    maxNJets = 25

    for key in [b'Jet_pt', b'Jet_eta', b'Jet_mass', b'Jet_phi']:
        tree[key] = tree[key].pad(maxNJets)
        try:
            the_list = tree[key].tolist()
            tree[key] = np.asarray(the_list, dtype=float)
            np.nan_to_num(tree[key], copy=False)
        except ValueError:
            import pdb; pdb.set_trace()

    tree[b'Jet_energy'] = np.sqrt(np.maximum(0., (tree[b'Jet_pt']*np.cosh(tree[b'Jet_eta']))**2 + tree[b'Jet_mass']**2))
    tree[b'Jet_pz'] = tree[b'Jet_pt']*np.sinh(tree[b'Jet_eta'])
    tree[b'Jet_px'] = np.cos(tree[b'Jet_phi']) * tree[b'Jet_pt']
    tree[b'Jet_py'] = np.sin(tree[b'Jet_phi']) * tree[b'Jet_pt']


    jet_keys = [b'Jet_px', b'Jet_py', b'Jet_pz', b'Jet_energy']

    X_jet = np.stack([tree[key] for key in jet_keys], axis=2)

    for key in [b'GenJet_pt', b'GenJet_eta', b'GenJet_mass', b'GenJet_phi']:
        tree[key] = tree[key].pad(maxNJets)
        try:
            the_list = tree[key].tolist()
            tree[key] = np.asarray(the_list, dtype=float)
            np.nan_to_num(tree[key], copy=False)
        except ValueError:
            import pdb; pdb.set_trace()


    tree[b'GenJet_energy'] = np.sqrt(np.maximum(0., (tree[b'GenJet_pt']*np.cosh(tree[b'GenJet_eta']))**2 + tree[b'GenJet_mass']**2))
    tree[b'GenJet_pz'] = tree[b'GenJet_pt']*np.sinh(tree[b'GenJet_eta'])
    tree[b'GenJet_px'] = np.cos(tree[b'GenJet_phi']) * tree[b'GenJet_pt']
    tree[b'GenJet_py'] = np.sin(tree[b'GenJet_phi']) * tree[b'GenJet_pt']

    gen_jet_keys = [b'GenJet_px', b'GenJet_py', b'GenJet_pz', b'GenJet_energy']

    X_gen_jet = np.stack([tree[key] for key in gen_jet_keys], axis=2)

def nonify_first(shape):
    return tuple([None if i == 0 else el for i, el in enumerate(shape)])

if not opt.append or not os.path.isfile(opt.output):
    with h5py.File(opt.output, 'w') as h5f:
        if not opt.nopf:
            h5f.create_dataset('X', data=X, compression=opt.compression, chunks=(256, maxNPF, X.shape[2]), maxshape=nonify_first(X.shape))
            for i in range(X_c.shape[2]):
                h5f.create_dataset(f'X_c_{i}', data=X_c[...,i][..., None], compression=opt.compression, chunks=(256, maxNPF, 1), maxshape=nonify_first(X.shape))
        h5f.create_dataset('Y', data=Y, compression=opt.compression, chunks=(256, Y.shape[1]), maxshape=nonify_first(Y.shape))
        h5f.create_dataset('Z', data=Z, compression=opt.compression, chunks=(256, Z.shape[1]), maxshape=nonify_first(Z.shape))
        if opt.jets:
            h5f.create_dataset('X_jet', data=X_jet, compression=opt.compression, chunks=(256, X_jet.shape[1], X_jet.shape[2]), maxshape=nonify_first(X_jet.shape))
            h5f.create_dataset('X_gen_jet', data=X_gen_jet, compression=opt.compression, chunks=(256, X_gen_jet.shape[1], X_gen_jet.shape[2]), maxshape=nonify_first(X_gen_jet.shape))
        print('Finished')
        print(Y.shape)
        print(Z.shape)
else:
    with h5py.File(opt.output, 'a') as h5f:
        if not opt.nopf:
            h5f['X'].resize(h5f['X'].shape[0] + X.shape[0], axis=0)
            h5f['X'][-X.shape[0]:] = X
            for i in range(X_c.shape[2]):
                h5f[f'X_c_{i}'].resize(h5f[f'X_c_{i}'].shape[0] + X_c.shape[0], axis=0)
                h5f[f'X_c_{i}'][-X_c.shape[0]:] = X_c[..., i][..., None]

        h5f['Y'].resize(h5f['Y'].shape[0] + Y.shape[0], axis=0)
        h5f['Y'][-Y.shape[0]:] = Y
        h5f['Z'].resize(h5f['Z'].shape[0] + Z.shape[0], axis=0)
        h5f['Z'][-Z.shape[0]:] = Z
        print('Finished')
        print(h5f['Y'].shape)
        print(h5f['Z'].shape)
