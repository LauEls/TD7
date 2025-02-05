import numpy as np
import torch


class LAP(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		device,
		max_size=1e6,
		batch_size=256,
		max_action=1,
		normalize_actions=True,
		prioritized=True
	):
	
		max_size = int(max_size)
		self.max_size = max_size
		print("max_size: ", max_size)
		self.ptr = 0
		self.size = 0

		self.device = device
		self.batch_size = batch_size

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.prioritized = prioritized
		# if prioritized:
		self.priority = torch.zeros(max_size, device=device)
		self.max_priority = 1

		self.normalize_actions = max_action if normalize_actions else 1

	
	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action/self.normalize_actions
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		# if self.prioritized:
		self.priority[self.ptr] = self.max_priority

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self):
		if self.prioritized:
			csum = torch.cumsum(self.priority[:self.size], 0)
			val = torch.rand(size=(self.batch_size,), device=self.device)*csum[-1]
			self.ind = torch.searchsorted(csum, val).cpu().data.numpy()
		else:
			self.ind = np.random.randint(0, self.size, size=self.batch_size)

		return (
			torch.tensor(self.state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.action[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.next_state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.not_done[self.ind], dtype=torch.float, device=self.device)
		)


	def update_priority(self, priority):
		self.priority[self.ind] = priority.reshape(-1).detach()
		self.max_priority = max(float(priority.max()), self.max_priority)


	def reset_max_priority(self):
		self.max_priority = float(self.priority[:self.size].max())


	def load_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]
		
		if self.prioritized:
			self.priority = torch.ones(self.size).to(self.device)

	def load_paths(self, paths):
		cntr = 0
		cntr_2 = 0
		reward_sum = 0
		for path in paths:
			self.total_score = 0
			self.total_ee_wrench = 0
			self.path_steps = 0
			self.done = False
			cntr += 1
			for i, (
				obs,
				action,
				reward,
				next_obs,
				terminal
			) in enumerate(zip(
				path["observations"],
				path["actions"],
				path["rewards"],
				path["next_observations"],
				path["dones"]
			)):
				# print("terminal: ", terminal)
				cntr_2 += 1
				self.add(
					state=obs,
					action=action,
					reward=reward,
					next_state=next_obs,
					done=terminal,
				)
			# self.avg_score = self.total_score/self.path_steps
			reward_sum += self.total_score
			# print('path ', self.paths, 'score ', self.total_score)

		avg_reward = reward_sum/cntr 
		# print("Epoch average reward: ",avg_reward)
		print(self.not_done)
		# print("cnt: ", cntr)
		# print("cnt_2: ", cntr_2)
		return avg_reward    
	
	def save_paths(self, filename):
		paths = []
		paths.append(dict(
			observations=self.state[:self.size],
			actions=self.action[:self.size],
			rewards=self.reward[:self.size],
			next_observations=self.next_state[:self.size],
			dones=1-self.not_done[:self.size]
		))
		file_array = np.array(paths)

		np.save(filename, file_array)

	def save_priority(self, filename):
		np.save(filename, self.priority[:self.size].cpu().data.numpy())

	def load_priority(self, filename):
		self.priority[:self.size] = torch.tensor(np.load(filename), device=self.device)