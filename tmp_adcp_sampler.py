from subprocess import run
from tempfile import TemporaryDirectory
from pathlib import Path
from logging import warning
from functools import reduce
from scipy.stats import tmean, tvar
from numbers import Number
from typing import Tuple
import numpy as np
import pandas as pd
import os,shutil
from pathlib import Path
from Bio.Data.IUPACData import protein_letters_1to3
from random import choices,randint
from random import seed as seed_
from multiprocessing import Pool

aas=list(protein_letters_1to3.keys())
# seed(42)

def simple_aasampler(seed:int=42,sample_size:int=100):
    seed_(seed)
    o=[]
    for i in range(sample_size):
        length=randint(5,20)
        o.append(''.join(choices(aas,k=length)))
    return o

def parse_adcp_output(file:str|Path,top_k:int=3)->Tuple[Number,Number]:
    '''
    from adcp logs to estimated mean/var of binding affinity. 
    '''
    parse_flag=-1
    affinity,clustersize=[],[]
    for l in open(file,'r').readlines():
        if parse_flag==-1:
            if l.startswith('-----'):
                parse_flag+=1
        elif parse_flag<top_k:
            try:
                l_=l.strip().split()
                # print(l_)
                affinity.append(float(l_[1]))
                clustersize.append(int(l_[4]))
                parse_flag+=1
            except:
                warning(f'invalid line: {l} in {Path(file).absolute()}')
        else:
            break
    pseudo_scores=reduce(lambda a,b:a+b, [[a]*c for a,c in zip(affinity,clustersize)])
    return tmean(pseudo_scores),tvar(pseudo_scores)

def adcp_score(pep:str,trg:str|Path,wdir:str|Path='/root/ADFR/bin',
        iters:int=100000,replica:int=20,seed:int=42,):
    #TODO Dump the useful conformations & full logs
    trg_=Path(trg); parent,name=(
        Path(trg_.parent).absolute(),trg_.name)
    
    env = os.environ.copy()
    env['PATH']=str(wdir)+':'+env['PATH']
    with TemporaryDirectory() as tdir:
        os.chmod(tdir, 0o777)
        # shutil.copy(trg,wdir)
        with open(f'{tdir}/log','w') as log:
            cmds=['adcp','-cyc',
                '-t',name,
                '-s',pep,
                '-N',f'{replica}',
                '-n',f'{iters}',
                '-o',f'{tdir}/redocking',
                '-S',f'{seed}']
            run(cmds,
                cwd=parent,stdout=log,
                stderr=log,env=env
                )
        return ['#'+' '.join(cmds)+'\n']+open(f'{tdir}/log','r').readlines() 
    

def tmp_offline_db_gen():
    # from glob import glob
    seqs,scores=[],[]
    for path in Path('5LSO.db').iterdir():
        seq=path.stem.split('-')[0]
        try:
            score=parse_adcp_output(path)[0]
            score=0 if score>0 else score.item()
        except:
            score=100
            
        seqs.append(seq)
        scores.append(score)
        

    df=pd.DataFrame([seqs,scores]).T
    df.columns=['seq','score']
    def tmp_score_transform(s:float):
        if s<0:
            s=-s
        elif s==0:
            s=1e-10
        else:
            s=-100.
        return s
    df['score']=df['score'].apply(tmp_score_transform)
    df.to_csv('5LSO.db.csv',index=False)
if __name__=='__main__':
    # from functools import partial
    # iters=100000
    # replica=4
    # seed=42
    import tqdm
    import sys
    
    seed= sys.argv[1] #142857
    def write_score(pep:str):
        iters=100000
        replica=4
        # seed=42
        opts=adcp_score(pep,'/root/ADFR/bin/5LSO.trg',iters=iters,replica=replica,seed=seed)
        with open(f'5LSO.db/{pep}-i{iters}-r{replica}-s{seed}.log','w') as f:
            f.write(''.join(opts))
    
    peps=simple_aasampler(seed=seed,sample_size=2000)
    for i in tqdm.tqdm(peps):
        write_score(i)
    # pool=Pool(processes=16,maxtasksperchild=10)
    # pool.map_async(write_score,peps)
    # bar=tqdm.tqdm(total=1000)
    # update=lambda x :bar.update(1)
    # r = pool.map_async(write_score,peps,callback=update)
    # r.wait()
    
    
    
    