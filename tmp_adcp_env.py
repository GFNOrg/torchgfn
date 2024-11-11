# %%
# %load_ext autoreload
# %autoreload 2
from __future__ import annotations
from functools import partial
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

# %%
import torch
from Bio.Data.IUPACData import protein_letters_1to3
from torch import nn,Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.data import Batch, Data
from torchtyping import TensorType as TT
from tqdm import tqdm

from gfn.actions import Actions
from gfn.containers import Trajectories
from gfn.env import DiscreteEnv, Preprocessor
from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid  # We use the hyper grid environment
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from gfn.preprocessors import IdentityPreprocessor
from gfn.samplers import Sampler
from gfn.states import DiscreteStates, States
from gfn.utils import NeuralNet
from gfn.utils.modules import NeuralNet

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Batch.from_data_list
# %%
class IdentityLongPreprocessor(Preprocessor):
    """Simple preprocessor applicable to environments with uni-dimensional states.
    This is the default preprocessor used."""

    def preprocess(self, states: States) -> TT["batch_shape", "input_dim"]:
        return states.tensor.long()

    def __call__(self, states: States) -> TT["batch_shape", "input_dim"]:
        return self.preprocess(states)

    def __repr__(self):
        return f"{self.__class__.__name__}, output_dim={self.output_dim}"


class ADCPCycEnv(DiscreteEnv):
    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 16,
        device_str: str|int = 0,
        reward_mode:Literal['look_up','simple_pattern']='look_up',
        module_mode:Literal['CycEncoder','MLP']='CycEncoder'
        # preprocessor: Optional[Preprocessor] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.device = torch.device(device_str)
        self.aa_tokens = tuple(protein_letters_1to3.keys())
        self.reward_mode=reward_mode
        self.module_mode=module_mode
        n_actions = len(self.aa_tokens) + 1  # last action reserved for exit

        self.s0_code, self.sf_code, self.dummy_code, self.exit_code = (
            n_actions - 1,
            -100,
            -1,
            n_actions - 1,
        )

        s0 = torch.full(
            (max_length,), fill_value=self.s0_code, dtype=torch.long, device=self.device
        )
        sf = torch.full(
            (max_length,), fill_value=self.sf_code, dtype=torch.long, device=self.device
        )
        state_shape = (self.max_length,)

        action_shape = (1,)
        # preprocessor = OneHotPreprocessor(token_size=len(self.aa_tokens))
        preprocessor = IdentityLongPreprocessor(output_dim=max_length)
        dummy_action = torch.tensor([self.dummy_code], device=self.device)
        exit_action = torch.tensor([self.exit_code], device=self.device)
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
        self.offline_db = pd.read_csv("5LSO.db.csv").set_index("seq")
        self.offline_db = self.offline_db[self.offline_db["score"] > 0]
        self.offline_db["score"] = self.offline_db["score"].apply(
            lambda x: 0.02 if x < 0.02 else x / 10
        )

    def step(
        self, states: DiscreteStates, actions: Actions
    ) -> TT["batch_shape", "state_shape", torch.long]:
        """
        """
        states_tensor, action_tensor = states.tensor, actions.tensor.squeeze(-1)

        # last_unfill_pos
        first_unfilled_pos = self.cal_first_unfilled_pos(states_tensor)
        # make sure for full-filled states, the only action allowed is exit_action

        # might be redundant? as is secured in `update_masks`
        # assert torch.all(action_tensor[last_unfilled_pos==-1] == self.exit_code)

        # fill the last-unfilled pos with given aa codes/exit code
        # valid_mask= last_unfilled_pos!=-1
        valid_mask = (
            (first_unfilled_pos != -1)
            & (action_tensor != self.exit_code)
            & (action_tensor != self.dummy_code)
        )
        valid_indices = first_unfilled_pos[valid_mask]
        # batch_indices = torch.arange(states_tensor.shape[0]
        #         )[valid_mask].to(states_tensor.device)
        # return states_tensor,valid_mask,valid_indices
        output = states_tensor.detach()
        output[(*valid_mask.nonzero(as_tuple=True), valid_indices)] = action_tensor[
            valid_mask
        ]
        return output

    def backward_step(
        self, states: DiscreteStates, actions: Actions
    ) -> TT["batch_shape", "state_shape", torch.long]:

        states_tensor, action_tensor = states.tensor, actions.tensor.squeeze(-1)
        last_filled_pos = self.cal_last_filled_pos(states_tensor)
        # current state is not s0
        valid_mask = last_filled_pos != -1
        valid_indices = last_filled_pos[valid_mask]

        # batch_indices = torch.arange(states_tensor.shape[0]
        #         )[valid_mask].to(states_tensor.device)'
        # return states_tensor,last_filled_pos,valid_mask,valid_indices
        # assert (states_tensor[(*valid_mask.nonzero(as_tuple=True),valid_indices)] == action_tensor[valid_mask]).all()

        output = states_tensor.detach()
        output[(*valid_mask.nonzero(as_tuple=True), valid_indices)] = self.s0_code

        return output
        # mask = torch.argmax(mask_, dim=1)
        # mask[~mask_.any(dim=1)] = -1

    @torch.inference_mode()  # is it legal?
    def seqs_to_trajectories(
        self, seqs: List[str], module_pf: nn.Module
    ) -> Trajectories:
        """
        TODO: `Trajectories` should have a 'device' property,
        so that all tensor should be initialized on CPU.
        Otherwise, dataloader would be incompatibility with num_workers.
        """
        # max_len=max([len(i) for i in seqs])
        seq_tensors, act_tensors, when_is_dones = [], [], []
        for seq in seqs:
            t = torch.full((self.max_length + 2, self.max_length), self.s0_code).long()
            a = torch.full(
                (self.max_length + 1, *self.action_shape), self.dummy_code
            ).long()
            l = len(seq)
            when_is_dones.append(l + 1)
            for i, aa in enumerate(seq):
                aidx = self.aa_tokens.index(aa)
                t[i + 1 : l + 1, i] = aidx
                a[i] = aidx
            t[l + 1 :] = self.sf_code
            a[l] = self.exit_code
            seq_tensors.append(t)
            act_tensors.append(a)

        # states, actions, when_is_done
        # with torch.no_grad():
        states_tensor = torch.stack(seq_tensors, dim=1).to(self.device)
        action_tensor = torch.stack(act_tensors, dim=1).to(self.device)
        states_class, action_class = self.make_states_class(), self.make_actions_class()
        states: DiscreteStates = states_class(states_tensor)
        actions: Actions = action_class(action_tensor)
        self.update_masks(states)
        when_is_done = torch.tensor(when_is_dones).to(self.device)

        # log_probs

        fw_probs = module_pf(states.tensor)
        valid_state_mask = (states.tensor != self.sf_code).all(dim=-1)
        # fw_probs[~states.forward_masks]=-torch.inf
        fw_probs = torch.where(states.forward_masks, fw_probs, -torch.inf)  # -100.

        fw_probs = torch.nn.functional.softmax(fw_probs, dim=-1)

        # fw_probs[~valid_state_mask]=1

        fw_probs = torch.where(
            valid_state_mask.unsqueeze(-1).repeat(
                *[1] * len(valid_state_mask.shape), fw_probs.shape[-1]
            ),
            fw_probs,
            1.0,
        )

        # a_tensor=actions.tensor.clone()
        
        # a_tensor[a_tensor==self.dummy_code]=self.exit_code
        a_tensor = torch.where(
            actions.tensor != self.dummy_code, actions.tensor, self.exit_code
        )

        log_probs = torch.log(
            torch.gather(input=fw_probs, dim=-1, index=a_tensor).squeeze(-1)
        )
        
        # log_probs=torch.log(log_probs) .sum(dim=0)
        final_states = states[when_is_done - 1, torch.arange(len(seqs)).to(self.device)]
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

    def states_to_seqs(self, final_states: DiscreteStates) -> List[str]:
        if final_states.batch_shape[0] != 0:
            a_arrays = np.vectorize(lambda x: self.aa_tokens[x] if x < 20 else "")(
                final_states.tensor.cpu().numpy()
            )
        else:
            return []
        seqs = []

        for row in a_arrays:
            seqs.append("".join(row))
        return seqs

    def log_reward(
        self, final_states: DiscreteStates
    ) -> Tensor:
        # place holder
        seqs = self.states_to_seqs(final_states)
        if self.reward_mode=='look_up':
            return torch.tensor([self.offline_db["score"].get(i, 0.02) for i in seqs]).to(self.device)
        elif self.reward_mode=='simple_pattern':
            '''
            debug rewards:
            1. seq-length: rewards:
            length:
                [0,4]: 2*x
                [5,9]: 10-2*(x-5)
                [10,14]: 2*(x-10)
                [15,20]: 10-2*(x-15)
            
            1-5 up; 5-15 down; 15-20 up
            2. pattern rewards:
            switch(pos%3):
                0 -> KREDHQN +p;
                1 -> CGPAST +p;
                2 -> VMLVIFWY +p;
                p: pos-wise score = (10/seq-length)
            '''
            mode_dict={
                0:'KREDHQN',
                1:'CGPAST',
                2:'MLVIFWY',
            }
            def simple_pattern(seq:str):
                s=0.
                seq_len=len(seq)
                if seq_len in [0,20]:
                    return s+1e-5
                elif seq_len<=4:
                    s+=2*seq_len
                elif seq_len<=9:
                    s+=10-2*(seq_len-5)
                elif seq_len<=14:
                    s+=2*(seq_len-10)
                elif seq_len<20:
                    s+=10-2*(seq_len-15)
                    
                pos_score=10/len(seq)
                for i,a in enumerate(seq):
                    if a in mode_dict[i%3]:
                        s+=pos_score
                return s+1e-5
            return torch.log(torch.tensor([simple_pattern(i) for i in seqs])).to(self.device)
        else:
            raise ValueError
        # return torch.log(torch.tensor([self.offline_db['score'].get(i,0.02) for i in seqs])).to(self.device)
        # return torch.randn(final_states.batch_shape,device=final_states.tensor.device)

    def update_masks(self, states: DiscreteStates) -> None:
        """Update the masks based on the current states."""
        states_tensor = states.tensor
        # first_unfilled_pos=self.cal_first_unfilled_pos(states_tensor)
        last_filled_pos = self.cal_last_filled_pos(states_tensor)

        states.forward_masks = torch.ones(
            (*states.batch_shape, self.n_actions),
            dtype=torch.bool,
            device=states_tensor.device,
        )

        # for full-filled states, only allow exit
        states.forward_masks[
            (
                *(last_filled_pos == self.max_length - 1).nonzero(as_tuple=True),
                slice(None, self.n_actions - 1),
            )
        ] = False
            # states.forward_masks[(*(last_filled_pos==self.max_length-1).nonzero(as_tuple=True),slice(None,self.n_actions-1))]
        # for l<l_min, prohibit exit actions
        states.forward_masks[
            (
                *(last_filled_pos < self.min_length - 1).nonzero(as_tuple=True),
                self.n_actions - 1,
            )
        ] = False
        # return
        states.backward_masks = torch.zeros(
            (*states.batch_shape, self.n_actions - 1),
            dtype=torch.bool,
            device=states_tensor.device,
        )

        valid_mask = last_filled_pos != -1
        valid_indices = states_tensor[
            (*valid_mask.nonzero(as_tuple=True), last_filled_pos[valid_mask])
        ]
        states.backward_masks[
            (*valid_mask.nonzero(as_tuple=True), valid_indices)
        ] = True

    def make_random_states_tensor(
        self, batch_shape: Tuple[int, ...]
    ) -> TT["batch_shape", "state_shape", torch.float]:
        '''
        #TODO sf in randoms?
        '''
        states_tensor = torch.full(
            (*batch_shape, self.max_length), self.s0_code, device=self.device
        ).long()
        fill_until = torch.randint(0, self.max_length + 1, batch_shape)
        # fill_until=torch.full(batch_shape,self.max_length)
        for i in range(self.max_length):
            mask = i < fill_until
            random_numbers = torch.randint(
                0, self.max_length, batch_shape, device=self.device
            )
            states_tensor[(*mask.nonzero(as_tuple=True), i)] = random_numbers[mask]
        return states_tensor
        # states.set_default_typing()
        # # Not allowed to take any action beyond the environment height, but
        # # allow early termination.
        # states.set_nonexit_action_masks(
        #     states.tensor == self.height - 1,
        #     allow_exit=True,
        # )
        # states.backward_masks = states.tensor != 0

    def make_modules(self,**kwargs)->Tuple[nn.Module,nn.Module]:
        # raise NotImplementedError
        # env=self
        '''
        TODO set up shared truncks
        kwargs: for Module initialization
        '''
        if self.module_mode=='CycEncoder':
            encoder = CircularEncoder(self)
            module_PF, module_PB = SillyModule(self.n_actions, encoder), SillyModule(
                self.n_actions - 1, encoder
            )
        elif self.module_mode=='MLP':
             module_PF= SimplestModule(self,self.n_actions,**kwargs)
             module_PB = SimplestModule(self,self.n_actions-1,share_trunk_with=module_PF,**kwargs)
        else:
            raise ValueError
        
        return module_PF.to(self.device), module_PB.to(self.device)

    def make_offline_dataloader(
        self, module_pf: nn.Module, **kwargs
    ) -> DataLoader[Trajectories]:
        """
        kwargs for `DataLoader`
        """
        return DataLoader(
            dataset=OfflineSeqOnlyDataSet(env=self),
            collate_fn=partial(collate_fn, env=self, module_pf=module_pf),
            **kwargs,
        )

    def cal_first_unfilled_pos(
        self, states_tensor: TT["batch_shape", "state_shape"]
    ) -> TT["batch_shape"]:
        """
        -1 for full-filled states & dummy
        """
        first_unfilled_pos_ = (states_tensor == self.s0_code).long()
        first_unfilled_pos = torch.argmax(first_unfilled_pos_, dim=-1)
        first_unfilled_pos[~first_unfilled_pos_.any(dim=-1)] = (
            -1
        )  # -1 for full-filled states
        first_unfilled_pos[(states_tensor == self.sf_code).any(dim=-1)] = -1
        return first_unfilled_pos

    def cal_last_filled_pos(
        self, states_tensor: TT["batch_shape", "state_shape"]
    ) -> TT["batch_shape"]:
        """
        -1 for s0 states & dummy
        """
        # get those totally unfilled pos
        last_filled_pos = self.cal_first_unfilled_pos(states_tensor) - 1
        # no unfilled -> last filled pos = len(states)
        last_filled_pos[last_filled_pos == -2] = self.max_length - 1
        last_filled_pos[(states_tensor == self.sf_code).any(dim=-1)] = -1
        return last_filled_pos


class CircularEncoder(nn.Module):
    '''
    TODO remove ADCPCycEnv dependence
    '''
    def __init__(self, env: ADCPCycEnv):
        super().__init__()
        self.embedding_dim = 128
        self.nhead = 4

        (self.sf_code, self.s0_code, self.max_length) = (
            env.sf_code,
            env.s0_code,
            env.max_length,
        )
        assert self.embedding_dim % (self.nhead * 2) == 0, "invalid nhead"
        self.pos_eb_dim = self.embedding_dim // (self.nhead * 2)

        self.dim_feedforward = 512
        self.embedding = nn.Embedding(
            num_embeddings=len(env.aa_tokens) + 1,
            embedding_dim=self.embedding_dim,
            padding_idx=len(env.aa_tokens),
        )
        self.encoder = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
        )

    def forward(
        self, trajs: TT["batch_shape", "state_shape"]
    ) -> TT["batch_shape", "eb_shape"]:
        trajs = torch.where(trajs != self.sf_code, trajs, self.s0_code)
        # trajs.clone()
        trajs[trajs == self.sf_code] = self.s0_code
        batch_shape, state_shape = trajs.shape[:-1], trajs.shape[-1]
        ori_mask = trajs == self.s0_code

        trajs = trajs.view(-1, state_shape)
        # src_key_padding_mask= ori_mask.view(-1,state_shape)

        trajs = self.embedding(trajs)
        encoder_mask = ori_mask.view(-1, state_shape)
        # print(self.positional_embedding(encoder_mask).shape)
        trajs = trajs + self.positional_embedding(encoder_mask)
        trajs = (trajs) / torch.linalg.vector_norm(trajs, dim=-1, keepdim=True)

        trajs = self.encoder(src=trajs, src_key_padding_mask=encoder_mask)
        trajs = trajs.view(batch_shape + trajs.shape[-2:])

        # src_key_padding_mask_dim=len(ori_mask.shape)
        ext_key_padding_mask = ori_mask.reshape(*ori_mask.shape, 1).repeat(
            *[1] * len(ori_mask.shape), trajs.shape[-1]
        )
        # print(ext_key_padding_mask.shape)
        # print(trajs.shape)
        trajs[ext_key_padding_mask] = torch.nan
        return torch.nanmean(trajs, dim=-2)

    def positional_embedding(self, encoder_mask: TT["flat_batch_shape", "state_shape"]):
        b, l, e = encoder_mask.shape[0], self.max_length, self.pos_eb_dim
        valid_length = (~encoder_mask).long().sum(dim=-1)
        _ = torch.einsum(
            "i,j,k->ijk",
            1 / (valid_length + 1e-8),
            torch.arange(0, l).to(valid_length.device),
            2 * torch.pi * torch.arange(0, e).to(valid_length.device),
        )
        ebs = torch.zeros([b, l, e * 2]).to(valid_length.device)
        ebs[:, :, 0::2] = torch.sin(_)
        ebs[:, :, 1::2] = torch.cos(_)

        return ebs.repeat(1, 1, self.nhead)


class SillyModule(nn.Module):
    def __init__(self, outdim: int, encoder: CircularEncoder):
        super().__init__()
        self.encoder = encoder
        self.outdim = outdim
        self.head = nn.Sequential(
            *[
                nn.Linear(encoder.embedding_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, outdim),
                nn.LayerNorm(outdim),
            ]
        )

    def forward(
        self, trajs: TT["batch_shape", "state_shape"]
    ) -> TT["batch_shape", "outdim"]:
        trajs = self.encoder(trajs)
        trajs = self.head(trajs)
        # TMP nan to euqal
        # trajs[trajs.isnan()]=1/self.outdim
        trajs = torch.where(~trajs.isnan(), trajs, 1 / self.outdim)

        return trajs


class SimplestModule(nn.Module):
    def __init__(self,
        env:ADCPCycEnv,num_outputs:int,
        hide_dim:int=2048,num_layers:int=5,dropout:float=0.1,
        share_trunk_with:SimplestModule|None=None,
        ):
        super().__init__()
        self.max_length,self.num_tokens,self.sf_code = (
                env.max_length,
                len(env.aa_tokens),
                env.sf_code,
                )
        
        if share_trunk_with is None:
            (self.num_outputs,self.hide_dim,
            self.num_layers,self.dropout
                )=num_outputs,hide_dim,num_layers,dropout
            
            self.input = nn.Linear(self.max_length*(self.num_tokens+1),hide_dim)
            
            hidden_layers = []
            for _ in range(num_layers):
                hidden_layers.append(nn.Dropout(dropout))
                hidden_layers.append(nn.ReLU())
                hidden_layers.append(nn.Linear(hide_dim, hide_dim))
            self.hidden = nn.Sequential(*hidden_layers)
        else:
            (self.num_outputs,self.hide_dim,
            self.num_layers,self.dropout
                )=(share_trunk_with.num_outputs,
                   share_trunk_with.hide_dim,
                   share_trunk_with.num_layers,
                   share_trunk_with.dropout
                   )
            self.input=share_trunk_with.input
            self.hidden=share_trunk_with.hidden
        self.output = nn.Linear(hide_dim, num_outputs)
        
    def forward(self, states_tensor:Tensor):
        # states_tensor=torch.where(states_tensor!=self.sf_code,states_tensor,self.num_tokens+1)
        # states_tensor=F.one_hot(states_tensor,num_classes=self.num_tokens+2)
        x=torch.zeros(*states_tensor.shape,self.num_tokens+1).to(states_tensor.device)
        mask=states_tensor!=self.sf_code
        x[mask]=F.one_hot(states_tensor[mask],self.num_tokens+1).type_as(x)
        x=x.reshape(*x.shape[:-2],-1)
        x=self.input(x)
        x=self.hidden(x)
        x=self.output(x)
        return x
        
       
class SillyFeatureModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()       
        
class OfflineSeqOnlyDataSet(Dataset):
    def __init__(self, env: ADCPCycEnv):
        super().__init__()
        self.seqs = env.offline_db[env.offline_db["score"] > 0].index.to_list()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index) -> str:
        return self.seqs[index]


def collate_fn(seqs: List[str], env: ADCPCycEnv, module_pf: nn.Module) -> Trajectories:
    trajectories = env.seqs_to_trajectories(seqs, module_pf)
    return trajectories


def check_grads(model: nn.Module):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if torch.isnan(parameter.grad).any():
                print(f"NaN gradient in {name}")


def seqlength_dist(seqs:List[str]):
    fig,ax=plt.subplots(1,1)
    ax:Axes
    pd.Series([len(i)for i in seqs]).hist(ax=ax,density=True,bins=np.linspace(0,20,21))
    ax.set_xticks(np.arange(0.5,21),np.arange(0,21))
    return fig,ax

# aa_tokens = tuple(protein_letters_1to3.keys())
aa_tokens = tuple('KREDHQNCGPASTMLVIFWY')
def aa_dist(seqs:List[str]):
    fig,ax=plt.subplots(1,1)
    ax:Axes
    array=np.zeros((20,20),dtype=np.int64)
    for seq in seqs:
        for i,aa in enumerate(seq):
            array[i,aa_tokens.index(aa)]+=1
    
    
    # return array
    array=(array/(array.sum(axis=1).reshape(-1,1))).T
    
    np.nan_to_num(array,nan=0.) 
    ax.imshow(array,cmap='YlGn',vmin=0., vmax=1.)
    ax.set_xticks(np.arange(0,20),np.arange(1,21))
    ax.set_yticks(np.arange(0,20),aa_tokens)
    return fig,ax

def reward_dist(rewards:Tensor):
    fig,ax=plt.subplots(1,1)
    ax:Axes
    rewards=torch.exp(rewards).to('cpu').numpy()
    ax.hist(rewards,bins=np.linspace(0,20,21),density=True)
    return fig,ax

if __name__ == "__main__":
    reward_mode='simple_pattern'
    if reward_mode=='simple_pattern':
        exp_id = 22
        exp_name = 'exp-simple-hp-ls-lr0.01-hd1024-nl5-plateau'
        writer = SummaryWriter(log_dir=f"log/{exp_name}{exp_id}")
        # adcp_env = ADCPCycEnv(max_length=20)
        adcp_env = ADCPCycEnv(max_length=20,reward_mode='simple_pattern',module_mode='MLP',min_length=4)
        
        module_pf, module_pb = adcp_env.make_modules(hide_dim=1024,num_layers=5)
        # dataloader = adcp_env.make_offline_dataloader(
        #     module_pf=module_pf, batch_size=32, shuffle=True
        # )

        pf_estimator = DiscretePolicyEstimator(
            module_pf,
            adcp_env.n_actions,
            is_backward=False,
            preprocessor=adcp_env.preprocessor,
        )

        pb_estimator = DiscretePolicyEstimator(
            module_pb,
            adcp_env.n_actions,
            is_backward=True,
            preprocessor=adcp_env.preprocessor,
        )

        gfn = TBGFlowNet(logZ=0.0, pf=pf_estimator, pb=pb_estimator)
        # gfn:TBGFlowNet = torch.load('/root/torchgfn/log/exp-simple-hp-ls-lr0.1-hd1024-nl5-20-gfn.pt')
        gfn.to(0)
        # gfn:TBGFlowNet = torch.load('log/exp2-gfn.pt',map_location='cpu')
        # TODO don't save model as a whole. Save its parameters/states only.
        optimizer = torch.optim.Adam(gfn.pf_pb_parameters(), lr=5e-1)
        optimizer.add_param_group({"params": gfn.logz_parameters(), "lr": 5e-1})
        
        # torch.nn.utils.clip_grad_value_(module_PF.parameters(), 0.5)
        # torch.nn.utils.clip_grad_value_(gfn.logz_parameters(), 1.)

        step = 0
        from tqdm import tqdm

        with torch.autograd.set_detect_anomaly(True):
            for e in tqdm(range(300000)):
                # writer.add_scalar("epoch", e, global_step=step)
                # for trajectories in tqdm(dataloader):
                with torch.no_grad():
                    trajectories=gfn.sample_trajectories(adcp_env,32)
                    
                optimizer.zero_grad()
                # optimizer.add_param_group({"params": gfn.logz_parameters(), "lr": 1e-1})
                scheduler=ReduceLROnPlateau(optimizer,threshold=0.05,min_lr=10e-4)
                loss = gfn.loss(adcp_env, trajectories)
                # print(loss)
                
                loss.backward()
                writer.add_scalar("train/loss", loss.item(), global_step=step)
                optimizer.step()
                step += 1
                if e%100==0:
                    torch.save(gfn, f"log/{exp_name}{exp_id}-gfn.pt")
                    with torch.no_grad():
                        torch.save(gfn, f"log/{exp_name}{exp_id}-gfn.pt")
                        valid_bs=500
                        gfn.eval()
                        trajectories=gfn.sample_trajectories(adcp_env,valid_bs)
                        # trajectories.states[trajectories.when_is_done-1]
                        final_states=trajectories.states[trajectories.when_is_done - 1, torch.arange(valid_bs).to(adcp_env.device)]
                        seqs=adcp_env.states_to_seqs(final_states)
                        writer.add_figure(tag='val/length_dist',figure=seqlength_dist(seqs)[0],global_step=step)
                        writer.add_figure(tag='val/aa_dist',figure=aa_dist(seqs)[0],global_step=step)
                        writer.add_figure(tag='val/reward_dist',figure=reward_dist(trajectories.log_rewards)[0],global_step=step)
                        writer.add_text(tag='val/raw-seqs',text_string='\n'.join(seqs),global_step=step)
                        loss = gfn.loss(adcp_env, trajectories)
                        writer.add_scalar("val/loss", loss.item(), global_step=step)
                        scheduler.step(loss)
                        gfn.train()
    else:
        exp_id = 4
        exp_name = 'exp'
        writer = SummaryWriter(log_dir=f"log/{exp_name}{exp_id}")
        adcp_env = ADCPCycEnv(max_length=20)
        # adcp_env = ADCPCycEnv(max_length=20,reward_mode='simple_pattern',module_mode='MLP')
        
        module_pf, module_pb = adcp_env.make_modules()
        dataloader = adcp_env.make_offline_dataloader(
            module_pf=module_pf, batch_size=32, shuffle=True
        )

        pf_estimator = DiscretePolicyEstimator(
            module_pf,
            adcp_env.n_actions,
            is_backward=False,
            preprocessor=adcp_env.preprocessor,
        )

        pb_estimator = DiscretePolicyEstimator(
            module_pb,
            adcp_env.n_actions,
            is_backward=True,
            preprocessor=adcp_env.preprocessor,
        )

        gfn = TBGFlowNet(logZ=0.0, pf=pf_estimator, pb=pb_estimator)
        # gfn:TBGFlowNet = torch.load('log/exp2-gfn.pt',map_location='cpu')
        # TODO don't save model as a whole. Save its parameters/states only.
        optimizer = torch.optim.Adam(gfn.pf_pb_parameters(), lr=1e-4)
        optimizer.add_param_group({"params": gfn.logz_parameters(), "lr": 1e-3})
        
        # torch.nn.utils.clip_grad_value_(module_PF.parameters(), 0.5)
        # torch.nn.utils.clip_grad_value_(gfn.logz_parameters(), 1.)

        step = 0
        from tqdm import tqdm

        with torch.autograd.set_detect_anomaly(True):
            for e in range(100):
                writer.add_scalar("epoch", e, global_step=step)
                for trajectories in tqdm(dataloader):
                    optimizer.zero_grad()
                    # optimizer.add_param_group({"params": gfn.logz_parameters(), "lr": 1e-1})
                    
                    loss = gfn.loss(adcp_env, trajectories)
                    # print(loss)
                    
                    loss.backward()
                    writer.add_scalar("train/loss", loss.item(), global_step=step)
                    optimizer.step()
                    step += 1

                torch.save(gfn, f"log/exp{exp_id}-gfn.pt")

def test_ADCPCycEnv():
    raise NotImplementedError
    adcp_env = ADCPCycEnv()
    states_class = adcp_env.make_states_class()
    action_class = adcp_env.make_actions_class()
    states_tensor = adcp_env.make_random_states_tensor((2,))
    module_PF, module_PB = adcp_env.make_modules()

    module_PF(states_tensor)
    random_states_tensor = torch.tensor(
        [
            [
                [12, 5, 4, 8, 12, 15, 11, 9, 15, 11, 3, 4, 4, 0, 0, 12],
                [1, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 0, 6, 2, 16, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [2, 1, 1, 13, 11, 12, 14, 15, 6, 9, 2, 15, 14, 5, 14, -1],
            ],
            [
                [0, 2, 11, 1, 4, 5, 13, 12, -1, -1, -1, -1, -1, -1, -1, -1],
                [12, 4, 10, 9, 10, 13, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1],
                [12, 5, 4, 8, 12, 15, 11, 9, 15, 11, 3, 4, 4, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ],
        ]
    )
    random_actions = torch.tensor([], [])
    random_states = states_class(random_states_tensor)
    adcp_env.update_masks(random_states)
    s, a = adcp_env.seqs_to_states_actions(["ADCEG", "AAAAAAAAAAAA"])


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

# class LoopGraphPreprocessor:
#     def __init__(self):
#         pass
#     def preprocess(self,states:States):
#         batch_shape=states.batch_shape
