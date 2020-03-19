import torch
import gym
import random
import numpy as np
from copy import deepcopy
from collections import namedtuple


Transition = namedtuple("Transition",("state",
									  "action",
									  "reward",
									  "next_state",
									  "terminal"))


class Net(torch.nn.Module):
	
	def __init__(self, in_size, out_size):
		super().__init__()
		
		self.lin1 = torch.nn.Linear(in_size, 64)
		self.nrm1 = torch.nn.LayerNorm(64)
		self.rel1 = torch.nn.ReLU()
		self.lin2 = torch.nn.Linear(64, 64)
		self.nrm2 = torch.nn.LayerNorm(64)
		self.rel2 = torch.nn.ReLU()
		self.lin3 = torch.nn.Linear(64, out_size)
		
		self.layers = (self.lin1, self.nrm1, self.rel1, self.lin2, self.nrm2, self.rel2, self.lin3)
		
		self.in_size = in_size
		self.out_size = out_size

	def forward(self,x):
		for layer in self.layers:
			x = layer(x)
		return x

	@staticmethod
	def param_init(module):
		"""
		Parameter initialization with relu gain.
		"""
		gain = torch.nn.init.calculate_gain("relu")
		if isinstance(module, torch.nn.Linear):
			torch.nn.init.xavier_uniform_(module.weight, gain)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)

class UniformBuffer(object):
	def __init__(self,buff_size):
		self.buff_size = buff_size
		self.queue = []
		self.cycle = 0

	def __len__(self):
		return len(self.queue)

	def push(self, **transition):
		if self.buff_size != len(self.queue):
			self.queue.append(transition)
		else:
			self.queue.append(transition)
			self.cycle = (self.cycle + 1 ) % self.buff_size
			
	def sample(self):
		if batchsize > len(self.queue):
			return None
		batch = randsample(self.queue, batchsize)
		return Transition(*zip(*batch))

class DQN(torch.nn.Module):
	"""
	Arguments:
		- valuenet: Neural network(torch.nn.Module) that will be used for the value estimination.

	"""

	def __init__(self, valuenet, buffersize = 10000):
		super().__init__()
		
		self.buffer = UniformBuffer(buffersize)
		self.network = valuenet
		self.targetnet = deepcopy(valuenet)
		self.update_count = 0


	def act(self, state, epsilon = 0.0):
		"""
		Arguments:
			- state: Batch size 1 torch tensor
			- epsilon: Random action probability
		"""

		# 
		with torch.no_grad():
			values = self.network(state)
		values = values.squeeze()

		if random.uniform(0,1) < epsilon:
			action = torch.randint(values.shape[-1], (1,)).long()
			value = values[action]
		return (action.item(), value.item())


	def _totorch(self, container, dtype):
			tensor = torch.tensor(container, dtype=dtype)
			return tensor
		
	def push(self, state, action, reward, next_state, terminal):
		self.buffer.push(**dict(state= state,
								action= action,
								reward= reward,
								next_state = next_state,
								terminal = terminal))

	def td_loss(self, gamma, batchsize):
		
		batch = self.buffer.sample()
		batch = self._batchtotorch(batch)
		
		# Gradient flow is disables in "next_values". Because we use Semi-gradient
		# q learning (a.k.a Q Learning in function approximation)
		with torch.no_grad():
			next_values = self.targetnet(batch.next_state)
			next_values = torch.max(next_values, dim=1, keepdim=True)[0]
			
		current_values = self.network(batch.state)
		current_values = current_values.gather(1, batch.action)

		target_value = next_values*(1 - batch.terminal)*gamma + batch.reward

		td_error = torch.nn.functional.smooth_l1_loss(current_values, target_value)

		return td_error
	
	
	def update(self, opt,target_update_period, gamma, batchsize, grap_clip = False):

		if self.update_count >= target_update_period:
			self.update_count = 0
			self.targetnet.load_state_dict(self.network.state_dict())
			
		# clear old gradients from the last step    
		opt.zero_grad()

		loss = self.td_loss(gamma, batchsize)

		# Compute the derivative of the loss
		loss.backward()

		# CLip the gradients to prevent graidents to diverge
		if grap_clip:
			for param in self.network.parameters():
				param.grad.data.clamp_(-1,1)

		# opt.step() causes the optimizer to take a step based on the gradients of the parameters
		opt.step()
		self.update_count += 1
		return loss.item()

		
	def _batchtotorch(self):
		
		state = self._totorch(batch.state, torch.float32)
		action = self._totorch(batch.action, torch.long).view(-1, 1)
		next_state = self._totorch(batch.next_state, torch.float32)
		terminal = self._totorch(batch.terminal, torch.float32).view(-1, 1)
		reward = self._totorch(batch.reward, torch.float32).view(-1, 1)
		return Transition(state, action, reward, next_state, terminal)


def train(args, net, opt, env) :
	for eps in range(args.episodes):
		state = env.reset()

		eps_reward = 0
		episode_value = 0
		epsilon = (args.start_epsilon - args.end_epsilon)*(1-eps/args.episodes) + args.end_epsilon

		for i in range(args.maxlength):
			torch_state = agent._totorch(state, torch.float32).view(1,-1)
			action, value = agent.act(torch_state, args.end_epsilon)
			next_state, reward, done = env.step(action)
			agent.push(state, action, reward, next_state, done)

			eps_reward += reward
			episode_value +=value

			# Start updating when the replay memory has enough samples
			if len(agent.buffer) > args.batchsize:
				td_loss = agent.update(opt, args.update_period, args.gamma, args.batchsize, args.clipgrap)

			if done:
				break
			state = next_state
		reward_list.append(eps_reward)
		print("Episode: {}, avg episodic reward: {}".format(eps, np.mean(reward_list[-10:])))
	return reward_list


hyperparams = {"episodes": 300, 
				   "buffersize": 10000,
				   "start_epsilon": 0.9,
				   "end_epsilon": 0.1,
				   "maxlenght": 300,
				   "batchsize": 64,
				   "update_period": 100,
				   "lr": 0.001,
				   "gamma": 0.95,
				   "clipgrap": True
				  }


HyperParams = namedtuple("HyperParams", hyperparams.keys())
args = HyperParams(**hyperparams)

env = gym.make("LunarLander-v2")
insize = env.observation_space.shape[0]
outsize = env.observation_space.n


net = Net(insize, outsize)
agent = DQN(net, args.buffersize)
opt = torch.optim.Adam(agent.parameters(), args.lr)

rewards = train(args, net, opt, env)

