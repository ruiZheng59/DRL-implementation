import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self,batchsize):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batchsize = batchsize


    def generate_batchs(self):
        n_states = len(self.states)
        batchs_start = np.arange(0,n_states,self.batchsize)
        indices = np.arange(n_states,dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batchsize] for i in batchs_start] #indices, no order

        return np.array(self.states), \
            np.array(self.actions), \
            np.array(self.probs),\
            np.array(self.vals),\
            np.array(self.rewards),\
            np.array(self.dones),\
            batches

    def store_memory(self,state,action,probs,vals,reward,done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []



class ActorNetwork(nn.Module):
    def __init__(self,n_actions,input_dims,alpha,fc1_dims=256,
                 fc2_dims=256,chkpt_dir='tmp/ppo'):
        super(ActorNetwork,self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir,'actor_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims,fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims,fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims,n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(selfs,state):
        dist = selfs.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_checkpoint(T.load(self.checkpoint_file))




class CriticNetwork(nn.Module):
    def __init__(self,input_dims,alpha,fc1_dims=256,fc2_dims=256,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir,'critic_ppo')

        self.critic = nn.Sequential(
            nn.Linear(*input_dims,fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims,fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims,1)
        )

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,states):
        return self.critic(states)

    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_checkpoint(T.load(self.checkpoint_file))


class Agent():
    def __init__(self,input_dims,n_actions,gamma=0.99,alpha=0.0003,gae_lambda=0.95,
                 policy_clip=0.2,batch_size=64,N=2048,epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.epochs = epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims,alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self,state,action,probs,vals,reward,done):
        self.memory.store_memory(state,action,probs,vals,reward,done)

    def save_models(self):
        print('---saving models---')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('---load models---')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self,observation):
        state = T.tensor([observation],dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action,probs,value

    def learn(self):
        for _ in range(self.epochs):
            state_arr,action_arr,old_probs_arr,vals_arr,\
                reward_arr,done_arr,batchs = self.memory.generate_batchs()
            #GAE
            value = vals_arr
            advantage = np.zeros(len(reward_arr),dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t,len(reward_arr)-1):
                    a_t += discount*(reward_arr[k]+self.gamma*value[k+1]* \
                                     (1-done_arr[k])-value[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            #sample batch
            value = T.tensor(value).to(self.actor.device)
            for batch in batchs:
                states = T.tensor(state_arr[batch],dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
            #new probs
            dist = self.actor(states)
            critic_value = self.critic(states)

            critic_value = T.squeeze(critic_value)

            new_probs = dist.log_prob(actions)

            #ratio cilp
            prob_ratio = new_probs.exp()/old_probs.exp()
            weighted_probs = advantage[batch]*prob_ratio
            weighted_clipped_probs = T.clamp(prob_ratio,1-self.policy_clip,1+self.policy_clip)*advantage[batch]

            #loss
            actor_loss = -T.min(weighted_probs,weighted_clipped_probs).mean()
    #????????????????critic loss???????????????????????????
            returns = advantage[batch] + value[batch]
            critic_loss = (returns - critic_value)**2
            critic_loss = critic_loss.mean()

            total_loss = actor_loss + 0.5*critic_loss
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            total_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()

        self.memory.clear_memory()