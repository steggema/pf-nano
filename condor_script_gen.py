common = '''Should_Transfer_Files = NO
getenv = True
executable = /afs/cern.ch/user/s/steggema/work/analysis/CMSSW_10_2_14/src/markus/submit.sh
+MaxRuntime = 86400
+AccountingGroup = "group_u_CMST3.all"

'''


output = '''Output = con_{i}.out
Error = con_{i}.err
Log = con_{i}.log
Arguments = -i /eos/cms/store/cmst3/group/htautau/{basedir}/{sample}/myNanoProdMc_NANO_{job_id}.root -o /eos/user/s/steggema/ptmiss/{out_name}{job_id}.h5 --n_leptons_subtract {n_leptons_subtract}
Queue

'''

# basedir = 'PTMISS_2016'
# sample = 'DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8'
basedir = 'PTMISS'
# sample = 'DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8'
sample = 'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8'

job_ids = [310 + i for i in range(10)]
# out_name = '2016_dy_1lep_chunk'
# out_name = 'dy_2lep_chunk'
# out_name = 'tt_2lep_chunk'

n_leptons_subtract = 2

print common
for i, job_id in enumerate(job_ids):
    print output.format(i=i, basedir=basedir, sample=sample, job_id=job_id, out_name=out_name, n_leptons_subtract=n_leptons_subtract)
