# %%
# %load_ext autoreload
# %autoreload 2

#%%
import torch
from tqdm import tqdm
from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid  # We use the hyper grid environment
from gfn.modules import DiscretePolicyEstimator,ScalarEstimator
from gfn.samplers import Sampler
from gfn.utils import NeuralNet 
from gfn.preprocessors import IdentityPreprocessor
from gfn.env import DiscreteEnv,Preprocessor
from typing import Literal,Optional,Tuple,List
from torchtyping import TensorType as TT
from Bio.Data.IUPACData import protein_letters_1to3
from gfn.states import DiscreteStates,States
from gfn.actions import Actions
from gfn.utils.modules import NeuralNet 
from torch import nn
from gfn.samplers import Sampler
from torch_geometric.data import Data,Batch
from gfn.containers import Trajectories
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from functools import partial
# Batch.from_data_list
#%%
# class OneHotPreprocessor:
#     def __init__(self, token_size):
#         self.token_size=token_size
        
#     def preprocess(self,states:States) -> TT["batch_shape", "input_dim"]:
#         return torch.nn.functional.one_hot(
#             (states.tensor+1).long(),num_classes=self.token_size+1)
        
    # def __call__(self, states: States) -> TT["batch_shape", "input_dim"]:
    #     return self.preprocess(states)
    # def __repr__(self):
    #     return f"{self.__class__.__name__}, token_size={self.token_size}"
    
class IdentityLongPreprocessor(Preprocessor):
    """Simple preprocessor applicable to environments with uni-dimensional states.
    This is the default preprocessor used."""

    def preprocess(self, states: States) -> TT["batch_shape", "input_dim"]:
        return states.tensor.long()
        
    def __call__(self, states: States) -> TT["batch_shape", "input_dim"]:
        return self.preprocess(states)
    
    def __repr__(self):
        return f"{self.__class__.__name__}, output_dim={self.output_dim}"

# class LoopGraphPreprocessor:
#     def __init__(self):
#         pass
#     def preprocess(self,states:States):
#         batch_shape=states.batch_shape
class ADCPCycEnv(DiscreteEnv):
    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 16,
        device_str: str = 0,
        # preprocessor: Optional[Preprocessor] = None,
    ):
        self.min_length=min_length
        self.max_length=max_length
        self.device=torch.device(device_str)
        self.aa_tokens=tuple(protein_letters_1to3.keys())
        n_actions=len(self.aa_tokens)+1 # last action reserved for exit
        
        self.s0_code,self.sf_code,self.dummy_code,self.exit_code=(
            n_actions-1,-100,-1,n_actions - 1)
        
        s0=torch.full((max_length,),fill_value=self.s0_code, dtype=torch.long, device=self.device)
        sf=torch.full((max_length,),fill_value=self.sf_code, dtype=torch.long, device=self.device)
        state_shape = (self.max_length,) 
        
        action_shape=(1,)
        # preprocessor = OneHotPreprocessor(token_size=len(self.aa_tokens))
        preprocessor = IdentityLongPreprocessor(output_dim=max_length)
        dummy_action=torch.tensor([self.dummy_code], device=self.device)
        exit_action=torch.tensor([self.exit_code], device=self.device)
        super().__init__(
            n_actions=n_actions,
            action_shape=action_shape,
            state_shape=state_shape,
            s0=s0,
            sf=sf,
            dummy_action=dummy_action,
            exit_action=exit_action,
            preprocessor=preprocessor,
            device_str=device_str,
        )
        # self.states_class=self.make_states_class()
        # self.actions_class=self.make_actions_class()
        # self.module_pf,self.module_pb=self.make_modules()
        # tmp offline sample entries
        self.offline_db=pd.read_csv('5LSO.db.csv').set_index('seq')
        self.offline_db=self.offline_db[self.offline_db['score']>0]
        self.offline_db['score']=self.offline_db['score'].apply(lambda x:0.02 if x<0.02 else x/10)
    def step(
        self, states: DiscreteStates, actions: Actions
        ) -> TT["batch_shape", "state_shape", torch.long]:
        '''
    
        here I simply make such actions ignored.
        '''
        states_tensor,action_tensor=states.tensor,actions.tensor.squeeze(-1)
        
        # last_unfill_pos
        first_unfilled_pos=self.cal_first_unfilled_pos(states_tensor)
        # make sure for full-filled states, the only action allowed is exit_action
        
        # might be redundant? as is secured in `update_masks`
        # assert torch.all(action_tensor[last_unfilled_pos==-1] == self.exit_code)
        
        # fill the last-unfilled pos with given aa codes/exit code
        # valid_mask= last_unfilled_pos!=-1
        valid_mask= ((first_unfilled_pos!=-1) & (action_tensor != self.exit_code) & (action_tensor != self.dummy_code))
        try:
            valid_indices = first_unfilled_pos[valid_mask]
        except:
            import pdb;pdb.set_trace()
        # batch_indices = torch.arange(states_tensor.shape[0]
        #         )[valid_mask].to(states_tensor.device)
        # return states_tensor,valid_mask,valid_indices
        output=states_tensor.detach()
        output[(*valid_mask.nonzero(as_tuple=True), 
                valid_indices
                )] = action_tensor[valid_mask]
        return output

    def backward_step(
        self, states: DiscreteStates, actions: Actions
        ) -> TT["batch_shape", "state_shape", torch.long]:

        states_tensor,action_tensor=states.tensor,actions.tensor.squeeze(-1)
        last_filled_pos=self.cal_last_filled_pos(states_tensor)
        # current state is not s0
        valid_mask = last_filled_pos!=-1
        valid_indices = last_filled_pos[valid_mask]
        
        # batch_indices = torch.arange(states_tensor.shape[0]
        #         )[valid_mask].to(states_tensor.device)'
        # return states_tensor,last_filled_pos,valid_mask,valid_indices
        # assert (states_tensor[(*valid_mask.nonzero(as_tuple=True),valid_indices)] == action_tensor[valid_mask]).all()
        
        output=states_tensor.detach()
        output[(*valid_mask.nonzero(as_tuple=True), 
                valid_indices
                )] = self.s0_code
        
        return output
        # mask = torch.argmax(mask_, dim=1)
        # mask[~mask_.any(dim=1)] = -1
           
    @torch.inference_mode() # is it legal?
    def seqs_to_trajectories(self,seqs:List[str],module_pf:nn.Module)->Trajectories:
        # max_len=max([len(i) for i in seqs])
        seq_tensors,act_tensors,when_is_dones=[],[],[]
        for seq in seqs:
            t=torch.full((self.max_length+2,self.max_length),self.s0_code).long()
            a=torch.full((self.max_length+1,*self.action_shape),self.dummy_code).long()
            l=len(seq)
            when_is_dones.append(l+1)
            for i,aa in enumerate(seq):
                aidx=self.aa_tokens.index(aa)
                t[i+1:l+1,i]=aidx
                a[i]=aidx
            t[l+1:]=self.sf_code
            a[l]=self.exit_code
            seq_tensors.append(t)
            act_tensors.append(a)
        
        # states, actions, when_is_done
        # with torch.no_grad(): 
        states_tensor=torch.stack(seq_tensors,dim=1).to(self.device)
        action_tensor=torch.stack(act_tensors,dim=1).to(self.device)
        states_class,action_class=self.make_states_class(),self.make_actions_class()
        states:DiscreteStates=states_class(states_tensor)
        actions:Actions=action_class(action_tensor)
        self.update_masks(states)
        when_is_done=torch.tensor(when_is_dones).to(self.device)
        
        # log_probs
        
        fw_probs=module_pf(states.tensor)
        valid_state_mask=(states.tensor!=self.sf_code).all(dim=-1)
        # fw_probs[~states.forward_masks]=-torch.inf
        fw_probs = torch.where(states.forward_masks, 
                fw_probs, -100.) # -torch.inf

        fw_probs=torch.nn.functional.softmax(fw_probs,dim=-1)
        
        # fw_probs[~valid_state_mask]=1
        
        fw_probs = torch.where(valid_state_mask.unsqueeze(-1).repeat(*[1]*len(valid_state_mask.shape),fw_probs.shape[-1]), 
                fw_probs, 1.0)
        
        # a_tensor=actions.tensor.clone()
        # import pdb;pdb.set_trace()
        # a_tensor[a_tensor==self.dummy_code]=self.exit_code
        a_tensor=torch.where(actions.tensor.clone()!=self.dummy_code,
                            actions.tensor.clone(),self.exit_code)
        
        log_probs=torch.gather(input=fw_probs,dim=-1,index=a_tensor).squeeze(-1)
        log_probs=torch.log10(log_probs).sum(dim=0)
        final_states=states[when_is_done-1,torch.arange(len(seqs)).to(self.device)]
        log_rewards = self.log_reward(final_states)
        
        trajectories = Trajectories(
            env=self,
            states=states,
            conditioning=None,
            actions=actions,
            when_is_done=when_is_done,
            is_backward=False,
            log_rewards=log_rewards,
            log_probs=log_probs,
            estimator_outputs=None,
        )
        
        return trajectories
    
    def states_to_seqs(self,final_states: DiscreteStates):
        a_arrays=np.vectorize(lambda x:self.aa_tokens[x] if x<20 else '')(final_states.tensor.cpu().numpy())
        seqs=[]
        for row in a_arrays:
            seqs.append(''.join(row))
        return seqs
    
    def log_reward(self,final_states: DiscreteStates) -> TT["batch_shape", torch.float]:
        # place holder
        seqs=self.states_to_seqs(final_states)
        return torch.log(torch.tensor([self.offline_db['score'].get(i,0.02) for i in seqs])).to(self.device)
        # return torch.randn(final_states.batch_shape,device=final_states.tensor.device)
        
    def update_masks(self, states: DiscreteStates) -> None:
        """Update the masks based on the current states."""
        states_tensor=states.tensor
        # first_unfilled_pos=self.cal_first_unfilled_pos(states_tensor)
        last_filled_pos=self.cal_last_filled_pos(states_tensor)
        
        states.forward_masks = torch.ones(
                (*states.batch_shape, self.n_actions),
                dtype=torch.bool,
                device=states_tensor.device,
            )
        
        # for full-filled states, only allow exit
        try:
            states.forward_masks[(*(last_filled_pos==self.max_length-1
                       ).nonzero(as_tuple=True),slice(None,self.n_actions-1))]=False
            # states.forward_masks[(*(last_filled_pos==self.max_length-1).nonzero(as_tuple=True),slice(None,self.n_actions-1))]
        except:
            import pdb;pdb.set_trace()
        # for l<l_min, prohibit exit actions
        states.forward_masks[(*(last_filled_pos<self.min_length-1
                       ).nonzero(as_tuple=True),self.n_actions-1)]=False
        # return 
        states.backward_masks = torch.zeros(
                (*states.batch_shape, self.n_actions-1),
                dtype=torch.bool,
                device=states_tensor.device)
        
        valid_mask = last_filled_pos!=-1
        valid_indices=states_tensor[
            (*valid_mask.nonzero(as_tuple=True),last_filled_pos[valid_mask])]
        try:
            states.backward_masks[(*valid_mask.nonzero(as_tuple=True),valid_indices)]=True
        except:
            import pdb;pdb.set_trace()   
               
    def make_random_states_tensor(self, batch_shape: Tuple[int, ...]
    ) -> TT["batch_shape", "state_shape", torch.float]:
        states_tensor=torch.full((*batch_shape,self.max_length),self.s0_code, device=self.device).long()
        fill_until=torch.randint(0,self.max_length+1,batch_shape)
        # fill_until=torch.full(batch_shape,self.max_length)
        for i in range(self.max_length):
            mask = i < fill_until
            random_numbers = torch.randint(0, self.max_length, batch_shape, device=self.device)
            states_tensor[(*mask.nonzero(as_tuple=True),i)] = random_numbers[mask]
        return states_tensor
        # states.set_default_typing()
        # # Not allowed to take any action beyond the environment height, but
        # # allow early termination.
        # states.set_nonexit_action_masks(
        #     states.tensor == self.height - 1,
        #     allow_exit=True,
        # )
        # states.backward_masks = states.tensor != 0

    def make_modules(self):
        # raise NotImplementedError
        # env=self
        encoder=CircularEncoder(self)
        module_PF,module_PB=SillyModule(self.n_actions,encoder),SillyModule(self.n_actions-1,encoder)
        
        return module_PF.to(self.device),module_PB.to(self.device)
             
    def make_offline_dataloader(self,module_pf:nn.Module,**kwargs):
        '''
        kwargs for `DataLoader`
        '''
        return DataLoader(dataset=OfflineSeqOnlyDataSet(env=self),
                   collate_fn=partial(collate_fn,env=self,module_pf=module_pf),
                   **kwargs)   
                
    def cal_first_unfilled_pos(self,states_tensor:TT["batch_shape", "state_shape"])->TT["batch_shape"]:
        '''
        -1 for full-filled states & dummy
        '''
        first_unfilled_pos_ = (states_tensor == self.s0_code).long()
        first_unfilled_pos = torch.argmax(first_unfilled_pos_, dim=-1)
        first_unfilled_pos[~first_unfilled_pos_.any(dim=-1)] = -1 # -1 for full-filled states
        first_unfilled_pos[(states_tensor==self.sf_code).any(dim=-1)]=-1
        return first_unfilled_pos
    
    def cal_last_filled_pos(self,states_tensor:TT["batch_shape", "state_shape"])->TT["batch_shape"]:
        '''
        -1 for s0 states & dummy
        '''
        # get those totally unfilled pos
        last_filled_pos=self.cal_first_unfilled_pos(states_tensor)-1
        # no unfilled -> last filled pos = len(states)
        last_filled_pos[last_filled_pos==-2] = self.max_length-1
        last_filled_pos[(states_tensor==self.sf_code).any(dim=-1)]=-1
        return last_filled_pos
def check_grads(model:nn.Module):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if torch.isnan(parameter.grad).any():
                print(f'NaN gradient in {name}')
                
class CircularEncoder(nn.Module):
    def __init__(self,env:ADCPCycEnv):
        super().__init__()
        self.embedding_dim=128
        self.nhead=4
        
        (self.sf_code,self.s0_code,self.max_length)=(
            env.sf_code,env.s0_code,env.max_length
        )
        assert self.embedding_dim%(self.nhead*2)==0,'invalid nhead'
        self.pos_eb_dim=self.embedding_dim//(self.nhead*2)
        
        self.dim_feedforward=512
        self.embedding=nn.Embedding(
            num_embeddings=len(env.aa_tokens)+1,
            embedding_dim=self.embedding_dim,
            padding_idx=len(env.aa_tokens))
        self.encoder=nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            batch_first=True)
        
        
    def forward(self,trajs:TT["batch_shape","state_shape"])->TT["batch_shape","eb_shape"]:
        trajs=torch.where(trajs!=self.sf_code,trajs,self.s0_code) 
        # trajs.clone()
        trajs[trajs==self.sf_code]=self.s0_code
        batch_shape,state_shape = trajs.shape[:-1],trajs.shape[-1]
        ori_mask = trajs==self.s0_code
        
        trajs=trajs.view(-1,state_shape)
        # src_key_padding_mask= ori_mask.view(-1,state_shape)
        
        trajs=self.embedding(trajs)
        encoder_mask=ori_mask.view(-1,state_shape)
        # print(self.positional_embedding(encoder_mask).shape)
        trajs=trajs+self.positional_embedding(encoder_mask)
        trajs=(trajs)/torch.linalg.vector_norm(trajs,dim=-1,keepdim=True)
        
        trajs=self.encoder(src=trajs,src_key_padding_mask=encoder_mask)
        trajs=trajs.view(batch_shape+trajs.shape[-2:])
        
        # src_key_padding_mask_dim=len(ori_mask.shape)
        ext_key_padding_mask=ori_mask.reshape(*ori_mask.shape,1
            ).repeat(*[1]*len(ori_mask.shape),trajs.shape[-1])
        # print(ext_key_padding_mask.shape)
        # print(trajs.shape)
        trajs[ext_key_padding_mask]=torch.nan
        return torch.nanmean(trajs,dim=-2)
    
    def positional_embedding(self,encoder_mask:TT["flat_batch_shape","state_shape"]):
        b,l,e=encoder_mask.shape[0],self.max_length,self.pos_eb_dim
        valid_length=(~encoder_mask).long().sum(dim=-1)
        _=torch.einsum("i,j,k->ijk", 1/(valid_length+1e-8),
                        torch.arange(0,l).to(valid_length.device), 
                        2*torch.pi*torch.arange(0,e).to(valid_length.device))
        ebs=torch.zeros([b,l,e*2]).to(valid_length.device)
        ebs[:,:,0::2]=torch.sin(_)
        ebs[:,:,1::2]=torch.cos(_)
        
        return ebs.repeat(1,1,self.nhead)
    

class SillyModule(nn.Module):
    def __init__(self,outdim:int,encoder:CircularEncoder):
        super().__init__()
        self.encoder=encoder
        self.outdim=outdim
        self.head=nn.Sequential(*[
            nn.Linear(encoder.embedding_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, outdim),
            nn.LayerNorm(outdim)
        ])
    def forward(self,trajs:TT["batch_shape","state_shape"])->TT["batch_shape","outdim"]:
        trajs=self.encoder(trajs)
        trajs = self.head(trajs)
        #TMP nan to euqal
        # trajs[trajs.isnan()]=1/self.outdim
        trajs=torch.where(~trajs.isnan(),trajs,1/self.outdim)
        
        return trajs
    
class OfflineSeqOnlyDataSet(Dataset):
    def __init__(self,env:ADCPCycEnv):
        super().__init__()
        self.seqs=env.offline_db[env.offline_db['score']>0].index.to_list()
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, index)->str:
        return self.seqs[index]
    
def collate_fn(seqs:List[str],env:ADCPCycEnv,module_pf:nn.Module)->Trajectories:
    trajectories=env.seqs_to_trajectories(seqs,module_pf)       
    return trajectories

if __name__=='__main__':
    exp_id=2
    writer=SummaryWriter(log_dir=f'log/exp{exp_id}')
    adcp_env=ADCPCycEnv(max_length=20)
    # so silly: 
    # bad practice to define class inside a class, make `num_workers` impossible
    module_pf,module_pb=adcp_env.make_modules()
    dataloader=adcp_env.make_offline_dataloader(module_pf=module_pf,batch_size=32) 
    
    # module_PF.named_parameters()
    pf_estimator = DiscretePolicyEstimator(
        module_pf, adcp_env.n_actions, 
        is_backward=False, 
        preprocessor=adcp_env.preprocessor)

    pb_estimator = DiscretePolicyEstimator(
        module_pb, adcp_env.n_actions, 
        is_backward=True, 
        preprocessor=adcp_env.preprocessor)

    # import pdb;pdb.set_trace()
    # clipping_value =  # arbitrary value of your choosing
    
    
    gfn = TBGFlowNet(logZ=0., pf=pf_estimator, pb=pb_estimator)
    optimizer = torch.optim.Adam(gfn.pf_pb_parameters(), lr=1e-4)
    optimizer.add_param_group({"params": gfn.logz_parameters(), "lr": 1e-3})
    # import pdb;pdb.set_trace()
    # torch.nn.utils.clip_grad_value_(module_PF.parameters(), 0.5)
    # torch.nn.utils.clip_grad_value_(gfn.logz_parameters(), 1.)

    step=0
    from tqdm import tqdm
    with torch.autograd.set_detect_anomaly(True):
        for e in range(30):
            writer.add_scalar('epoch',e,global_step=step)
            for trajectories in tqdm(dataloader):
                optimizer.zero_grad()
                # optimizer.add_param_group({"params": gfn.logz_parameters(), "lr": 1e-1})
                # import pdb;pdb.set_trace()
                loss=gfn.loss(adcp_env, trajectories)
                # print(loss)
                # import pdb;pdb.set_trace()
                loss.backward()
                writer.add_scalar('train/loss',loss.item(),global_step=step)
                optimizer.step()
                step+=1
                
            torch.save(gfn,f'log/exp{exp_id}-gfn.pt')



    # module_PF.apply(lambda m: m.register_backward_hook(lambda mod, grad_i, grad_o: check_grads(mod)))
#%%
# if 0:
#     import random
#     from torch.utils.data import Dataset
    
#     adcp_env=ADCPCycEnv(max_length=20)
#     random.shuffle(adcp_env.offline_db.index.to_list())
#     states_class=adcp_env.make_states_class()
#     action_class=adcp_env.make_actions_class()
#     # states_tensor=adcp_env.make_random_states_tensor((2,3))
#     module_PF,module_PB=adcp_env.make_modules()

#     pf_estimator = DiscretePolicyEstimator(
#         module_PF, adcp_env.n_actions, 
#         is_backward=False, 
#         preprocessor=adcp_env.preprocessor)
    
#     pb_estimator = DiscretePolicyEstimator(
#         module_PB, adcp_env.n_actions, 
#         is_backward=True, 
#         preprocessor=adcp_env.preprocessor)
    
#     trajectories=adcp_env.seqs_to_trajectories()
#     # sampler = Sampler(estimator=pf_estimator) 
#     # sample=sampler.sample_trajectories(env=adcp_env, n=2)
    
#     gfn = TBGFlowNet(logZ=0., pf=pf_estimator, pb=pb_estimator)
#     optimizer = torch.optim.Adam(gfn.pf_pb_parameters(), lr=1e-3)
#     optimizer.add_param_group({"params": gfn.logz_parameters(), "lr": 1e-1})
    
#     gfn.loss(adcp_env, sample)
    
#     for i in (pbar := tqdm(range(1000))):
#         trajectories = sampler.sample_trajectories(env=env, n=16)
#         optimizer.zero_grad()
#         loss = gfn.loss(env, trajectories)
#         loss.backward()
#         optimizer.step()
#         if i % 25 == 0:
#             pbar.set_postfix({"loss": loss.item()})
    # states,actions=adcp_env.seqs_to_states_actions(['ADCEGMFFTTENQ','AFAEAACAADA'])
    # fw_probs=module_PF(states.tensor)
    
    # valid_state_mask=(states.tensor!=adcp_env.sf_code).all(dim=-1)
    # fw_probs[~states.forward_masks]=-torch.inf
    # fw_probs=torch.nn.functional.softmax(fw_probs,dim=-1)
    # fw_probs[~valid_state_mask]=1
    
    # a_tensor=actions.tensor.clone()
    # a_tensor[a_tensor==adcp_env.dummy_code]=adcp_env.exit_code
    
    # log_probs=torch.gather(input=fw_probs,dim=-1,index=a_tensor).squeeze(-1)
    # log_probs=torch.log10(log_probs).sum(dim=0)
    

def test_ADCPCycEnv():
    raise NotImplementedError
    adcp_env=ADCPCycEnv()
    states_class=adcp_env.make_states_class()
    action_class=adcp_env.make_actions_class()
    states_tensor=adcp_env.make_random_states_tensor((2,))
    module_PF,module_PB=adcp_env.make_modules()
    
    module_PF(states_tensor)
    random_states_tensor=torch.tensor([
    [[12,  5,  4,  8, 12, 15, 11,  9, 15, 11,  3,  4,  4,  0,  0, 12],
        [ 1,  0,  6,  2,  -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  0,  6,  2,  16,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 2,  1,  1, 13, 11, 12, 14, 15,  6,  9,  2, 15, 14,  5, 14, -1]],
    [[ 0,  2, 11,  1,  4,  5, 13, 12, -1, -1, -1, -1, -1, -1, -1, -1],
        [12,  4, 10,  9, 10, 13, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1],
        [12,  5,  4,  8, 12, 15, 11,  9, 15, 11,  3,  4,  4,  0,  0, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]])
    random_actions=torch.tensor([],[])
    random_states=states_class(random_states_tensor)
    adcp_env.update_masks(random_states)
    s,a=adcp_env.seqs_to_states_actions(['ADCEG','AAAAAAAAAAAA'])
    
    
#%%

# adcp_env=ADCPCycEnv()

#%%

# nn.Conv1d


# module_PF = NeuralNet(
#     input_dim=adcp_env.preprocessor.output_dim,
#     output_dim=adcp_env.n_actions
# )  # Neural network for the forward policy, with as many outputs as there are actions

# module_PB = NeuralNet(
#     input_dim=adcp_env.preprocessor.output_dim,
#     output_dim=adcp_env.n_actions - 1,
#     trunk=module_PF.trunk  # We share all the parameters of P_F and P_B, except for the last layer
# )
# module_logF = NeuralNet(
#     input_dim=adcp_env.preprocessor.output_dim,
#     output_dim=1,  # Important for ScalarEstimators!
# )

# 3 - We define the estimators.
# pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions, is_backward=False, preprocessor=env.preprocessor)
# pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, is_backward=True, preprocessor=env.preprocessor)
# logF_estimator = ScalarEstimator(module=module_logF, preprocessor=env.preprocessor)


    


# class ADCPCycEnv(Env):
#     def __init__(self, s0, state_shape, action_shape, dummy_action, exit_action, sf = None, device_str = None, preprocessor = None):
#         super().__init__(s0, state_shape, action_shape, dummy_action, exit_action, sf, device_str, preprocessor)
        
#     def step(
#         self, states: States, actions: Actions
#         ) -> TT["batch_shape", "state_shape", torch.long]:
#         pass
#         '''
#         These functions do not need to handle masking for discrete environments, 
#         nor checking whether actions are allowed, 
#         nor checking whether a state is the sink state, etc...
#         '''
#     def backward_step(self, states, actions)  -> TT["batch_shape", "state_shape", torch.long]:
#         pass
    
#     def is_action_valid(
#         self,
#         states: States,
#         actions: Actions,
#         backward: bool = False,
#     ) -> bool:
#         '''
#         This function is used to ensure all actions are valid for both forward and backward trajectores 
#         (these are often different sets of rules) 
#         for continuous environments. 
#         It accepts a batch of states and actions, 
#         and returning True only if all actions can be taken at the given states.
#         '''
        
#     def make_random_states_tensor(self, batch_shape):
#         return super().make_random_states_tensor(batch_shape)
    
#     def reset(self, batch_shape = None, random = False, sink = False, seed = None):
#         return super().reset(batch_shape, random, sink, seed)
    
#     def log_reward(self, final_states):
#         return super().log_reward(final_states)
    
#     def make_states_class(self) -> type[States]:
#         env = self

#         class PepState(States):
#             """Defines a States class for this environment."""

#             state_shape = env.state_shape
#             s0 = env.s0
#             sf = env.sf
#             make_random_states_tensor = env.make_random_states_tensor

#         return PepState
    
    
#     def make_actions_class(self):
        # env = self

        # class AppendResAction(Actions):
        #     action_shape = env.action_shape
        #     dummy_action = env.dummy_action
        #     exit_action = env.exit_action

        # return AppendResAction