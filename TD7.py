import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import buffer


@dataclass
class Hyperparameters:
	# Generic
	batch_size: int = 256
	buffer_size: int = 1e6
	discount: float = 0.99
	target_update_rate: int = 250
	exploration_noise: float = 0.1
	
	# TD3
	target_policy_noise: float = 0.2
	noise_clip: float = 0.5
	policy_freq: int = 2
	
	# LAP
	alpha: float = 0.4
	min_priority: float = 1
	
	# TD3+BC
	lmbda: float = 0.1
	
	# Checkpointing
	max_eps_when_checkpointing: int = 20
	steps_before_checkpointing: int = 75e4 
	reset_weight: float = 0.9
	
	# Encoder Model
	zs_dim: int = 256
	enc_hdim: int = 256
	enc_activ: Callable = F.elu
	encoder_lr: float = 3e-4
	
	# Critic Model
	critic_hdim: int = 256
	critic_activ: Callable = F.elu
	critic_lr: float = 3e-4
	
	# Actor Model
	actor_hdim: int = 256
	actor_activ: Callable = F.relu
	actor_lr: float = 3e-4

	#Continue learning
	continue_learning: bool = False
	dir_path: str = ''


def AvgL1Norm(x, eps=1e-8):
	return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)


def LAP_huber(x, min_priority=1):
	return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.relu):
		super(Actor, self).__init__()

		self.activ = activ

		self.l0 = nn.Linear(state_dim, hdim)
		self.l1 = nn.Linear(zs_dim + hdim, hdim)
		self.l2 = nn.Linear(hdim, hdim)
		self.l3 = nn.Linear(hdim, action_dim)
		

	def forward(self, state, zs):
		a = AvgL1Norm(self.l0(state))
		a = torch.cat([a, zs], 1)
		a = self.activ(self.l1(a))
		a = self.activ(self.l2(a))
		return torch.tanh(self.l3(a))


class Encoder(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
		super(Encoder, self).__init__()

		self.activ = activ

		# state encoder
		self.zs1 = nn.Linear(state_dim, hdim)
		self.zs2 = nn.Linear(hdim, hdim)
		self.zs3 = nn.Linear(hdim, zs_dim)
		
		# state-action encoder
		self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
		self.zsa2 = nn.Linear(hdim, hdim)
		self.zsa3 = nn.Linear(hdim, zs_dim)
	

	def zs(self, state):
		zs = self.activ(self.zs1(state))
		zs = self.activ(self.zs2(zs))
		zs = AvgL1Norm(self.zs3(zs))
		return zs


	def zsa(self, zs, action):
		zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
		zsa = self.activ(self.zsa2(zsa))
		zsa = self.zsa3(zsa)
		return zsa


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
		super(Critic, self).__init__()

		self.activ = activ
		
		self.q01 = nn.Linear(state_dim + action_dim, hdim)
		self.q1 = nn.Linear(2*zs_dim + hdim, hdim)
		self.q2 = nn.Linear(hdim, hdim)
		self.q3 = nn.Linear(hdim, 1)

		self.q02 = nn.Linear(state_dim + action_dim, hdim)
		self.q4 = nn.Linear(2*zs_dim + hdim, hdim)
		self.q5 = nn.Linear(hdim, hdim)
		self.q6 = nn.Linear(hdim, 1)


	def forward(self, state, action, zsa, zs):
		sa = torch.cat([state, action], 1)
		embeddings = torch.cat([zsa, zs], 1)

		q1 = AvgL1Norm(self.q01(sa))
		q1 = torch.cat([q1, embeddings], 1)
		q1 = self.activ(self.q1(q1))
		q1 = self.activ(self.q2(q1))
		q1 = self.q3(q1)

		q2 = AvgL1Norm(self.q02(sa))
		q2 = torch.cat([q2, embeddings], 1)
		q2 = self.activ(self.q4(q2))
		q2 = self.activ(self.q5(q2))
		q2 = self.q6(q2)
		return torch.cat([q1, q2], 1)


class Agent(object):
	def __init__(self, state_dim, action_dim, max_action, offline=False, demo_buffer=False, hp=Hyperparameters()): 
		# Changing hyperparameters example: hp=Hyperparameters(batch_size=128)
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.hp = hp
		self.action_dim = action_dim

		self.actor = Actor(state_dim, action_dim, hp.zs_dim, hp.actor_hdim, hp.actor_activ).to(self.device)
		self.critic = Critic(state_dim, action_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ).to(self.device)
		self.encoder = Encoder(state_dim, action_dim, hp.zs_dim, hp.enc_hdim, hp.enc_activ).to(self.device)

		self.replay_buffer = buffer.LAP(state_dim, action_dim, self.device, hp.buffer_size, hp.batch_size, 
			max_action, normalize_actions=True, prioritized=True)
		
		if demo_buffer:
			self.demo_buffer = buffer.LAP(state_dim, action_dim, self.device, hp.buffer_size, int(hp.batch_size*0.1), max_action, normalize_actions=True, prioritized=False)
			self.replay_buffer.batch_size = int(hp.batch_size-self.demo_buffer.batch_size)
		else:
			self.demo_buffer = None

		self.max_action = max_action
		print("max_action: ", max_action)
		self.offline = offline
		self.continue_learning = self.hp.continue_learning

		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=hp.encoder_lr)
		
		self.actor_target = copy.deepcopy(self.actor)
		self.critic_target = copy.deepcopy(self.critic)
		self.fixed_encoder = copy.deepcopy(self.encoder)
		self.fixed_encoder_target = copy.deepcopy(self.encoder)
		self.checkpoint_actor = copy.deepcopy(self.actor)
		self.checkpoint_encoder = copy.deepcopy(self.encoder)


		if self.continue_learning:
			self.load_model(self.hp.dir_path)
			self.replay_buffer.load_paths(np.load(os.path.join(self.hp.dir_path,"replay_buffer.npy"), allow_pickle=True))
		self.training_steps = 0

		# Checkpointing tracked values
		self.eps_since_update = 0
		self.timesteps_since_update = 0
		self.max_eps_before_update = 1
		self.min_return = 1e8
		self.best_min_return = -1e8

		# Value clipping tracked values
		self.max = -1e8
		self.min = 1e8
		self.max_target = 0
		self.min_target = 0


	def select_action(self, state, use_checkpoint=False, use_exploration=True):
		with torch.no_grad():
			state = torch.tensor(state.reshape(1,-1), dtype=torch.float, device=self.device)

			if use_checkpoint: 
				zs = self.checkpoint_encoder.zs(state)
				action = self.checkpoint_actor(state, zs) 
			else: 
				zs = self.fixed_encoder.zs(state)
				action = self.actor(state, zs) 
			
			if use_exploration: 
				action = action + torch.randn_like(action) * self.hp.exploration_noise

			return action.clamp(-1,1).cpu().data.numpy().flatten() * self.max_action


	def train(self):
		self.training_steps += 1

		state, action, next_state, reward, not_done = self.replay_buffer.sample()

		if self.demo_buffer != None:
			state_demo, action_demo, next_state_demo, reward_demo, not_done_demo = self.demo_buffer.sample()
			# state[:-self.demo_buffer.batch_size] = state_demo
			# action[:-self.demo_buffer.batch_size] = action_demo
			# next_state[:-self.demo_buffer.batch_size] = next_state_demo
			# reward[:-self.demo_buffer.batch_size] = reward_demo
			# not_done[:-self.demo_buffer.batch_size] = not_done_demo
			state = torch.cat([state, state_demo], 0)
			action = torch.cat([action, action_demo], 0)
			next_state = torch.cat([next_state, next_state_demo], 0)
			reward = torch.cat([reward, reward_demo], 0)
			not_done = torch.cat([not_done, not_done_demo], 0)
			# state


		#########################
		# Update Encoder
		#########################
		with torch.no_grad():
			next_zs = self.encoder.zs(next_state)

		zs = self.encoder.zs(state)
		pred_zs = self.encoder.zsa(zs, action)
		encoder_loss = F.mse_loss(pred_zs, next_zs)

		self.encoder_optimizer.zero_grad()
		encoder_loss.backward()
		self.encoder_optimizer.step()

		#########################
		# Update Critic
		#########################
		with torch.no_grad():
			fixed_target_zs = self.fixed_encoder_target.zs(next_state)

			noise = (torch.randn_like(action) * self.hp.target_policy_noise).clamp(-self.hp.noise_clip, self.hp.noise_clip)
			next_action = (self.actor_target(next_state, fixed_target_zs) + noise).clamp(-1,1)
			
			fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)

			Q_target = self.critic_target(next_state, next_action, fixed_target_zsa, fixed_target_zs).min(1,keepdim=True)[0]
			Q_target = reward + not_done * self.hp.discount * Q_target.clamp(self.min_target, self.max_target)

			self.max = max(self.max, float(Q_target.max()))
			self.min = min(self.min, float(Q_target.min()))

			fixed_zs = self.fixed_encoder.zs(state)
			fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

		Q = self.critic(state, action, fixed_zsa, fixed_zs)
		td_loss = (Q - Q_target).abs()
		critic_loss = LAP_huber(td_loss)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		
		#########################
		# Update LAP
		#########################
		if self.demo_buffer != None:
			td_loss = td_loss[:-self.demo_buffer.batch_size]
		priority = td_loss.max(1)[0].clamp(min=self.hp.min_priority).pow(self.hp.alpha)
		self.replay_buffer.update_priority(priority)

		#########################
		# Update Actor
		#########################
		if self.training_steps % self.hp.policy_freq == 0:
			actor = self.actor(state, fixed_zs)
			fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor)
			Q = self.critic(state, actor, fixed_zsa, fixed_zs)

			actor_loss = -Q.mean() 
			if self.offline:
				# print("actor loss 1:", actor_loss)
				actor_loss = actor_loss + self.hp.lmbda * Q.abs().mean().detach() * F.mse_loss(actor, action)
				# print("actor loss 2:", actor_loss)

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

		#########################
		# Update Iteration
		#########################
		if self.training_steps % self.hp.target_update_rate == 0:
			self.actor_target.load_state_dict(self.actor.state_dict())
			self.critic_target.load_state_dict(self.critic.state_dict())
			self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
			self.fixed_encoder.load_state_dict(self.encoder.state_dict())
			
			self.replay_buffer.reset_max_priority()

			self.max_target = self.max
			self.min_target = self.min


	# If using checkpoints: run when each episode terminates
	def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
		self.eps_since_update += 1
		self.timesteps_since_update += ep_timesteps

		self.min_return = min(self.min_return, ep_return)

		# End evaluation of current policy early
		if self.min_return < self.best_min_return:
			self.train_and_reset()

		# Update checkpoint
		elif self.eps_since_update == self.max_eps_before_update:
			self.best_min_return = self.min_return
			self.checkpoint_actor.load_state_dict(self.actor.state_dict())
			self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())
			
			self.train_and_reset()


	# Batch training
	def train_and_reset(self):
		for _ in range(self.timesteps_since_update):
			if self.training_steps == self.hp.steps_before_checkpointing:
				self.best_min_return *= self.hp.reset_weight
				self.max_eps_before_update = self.hp.max_eps_when_checkpointing
			
			self.train()

		self.eps_since_update = 0
		self.timesteps_since_update = 0
		self.min_return = 1e8

	def save_model(self, path):
		torch.save(self.actor.state_dict(), os.path.join(path,"actor_params"))
		torch.save(self.encoder.state_dict(), os.path.join(path,"encoder_params"))
		torch.save(self.critic.state_dict(), os.path.join(path,"critic_params"))
		torch.save(self.actor_optimizer.state_dict(), os.path.join(path,"actor_optimizer_params"))
		torch.save(self.encoder_optimizer.state_dict(), os.path.join(path,"encoder_optimizer_params"))
		torch.save(self.critic_optimizer.state_dict(), os.path.join(path,"critic_optimizer_params"))
		torch.save(self.actor_target.state_dict(), os.path.join(path,"actor_target_params"))
		torch.save(self.critic_target.state_dict(), os.path.join(path,"critic_target_params"))
		torch.save(self.fixed_encoder.state_dict(), os.path.join(path,"fixed_encoder_params"))
		torch.save(self.fixed_encoder_target.state_dict(), os.path.join(path,"fixed_encoder_target_params"))
		torch.save(self.checkpoint_actor.state_dict(), os.path.join(path,"checkpoint_actor_params"))
		torch.save(self.checkpoint_encoder.state_dict(), os.path.join(path,"checkpoint_encoder_params"))
	
	def load_model(self, path):
		self.actor.load_state_dict(torch.load(os.path.join(path,"actor_params")))
		self.encoder.load_state_dict(torch.load(os.path.join(path,"encoder_params")))
		self.critic.load_state_dict(torch.load(os.path.join(path,"critic_params")))
		self.actor_optimizer.load_state_dict(torch.load(os.path.join(path,"actor_optimizer_params")))
		self.encoder_optimizer.load_state_dict(torch.load(os.path.join(path,"encoder_optimizer_params")))
		self.critic_optimizer.load_state_dict(torch.load(os.path.join(path,"critic_optimizer_params")))
		self.actor_target.load_state_dict(torch.load(os.path.join(path,"actor_target_params")))
		self.critic_target.load_state_dict(torch.load(os.path.join(path,"critic_target_params")))
		self.fixed_encoder.load_state_dict(torch.load(os.path.join(path,"fixed_encoder_params")))
		self.fixed_encoder_target.load_state_dict(torch.load(os.path.join(path,"fixed_encoder_target_params")))
		self.checkpoint_actor.load_state_dict(torch.load(os.path.join(path,"checkpoint_actor_params")))
		self.checkpoint_encoder.load_state_dict(torch.load(os.path.join(path,"checkpoint_encoder_params")))
		

		# self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
		# self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
		# self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=hp.encoder_lr)
		
		# self.actor_target = copy.deepcopy(self.actor)
		# self.critic_target = copy.deepcopy(self.critic)
		# self.fixed_encoder = copy.deepcopy(self.encoder)
		# self.fixed_encoder_target = copy.deepcopy(self.encoder)
		# self.checkpoint_actor = copy.deepcopy(self.actor)
		# self.checkpoint_encoder = copy.deepcopy(self.encoder)
