import numpy as np
import torch
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GridWorld:
    """
    This environment is motivated by the following paper.
    https://proceedings.mlr.press/v15/boularias11a/boularias11a.pdf

    - It has 50 x 50 grids (cells).
    - The agent has four actions for moving in one of the directions of the compass.
    - [Optional] If ```transition_prob``` = True, the actions succeed with probability 0.7,
      a failure results in a uniform random transition to one of the adjacent states.
    - A reward of 10 is given for reaching the goal state, located on the bottom-right corner.
    - For the remaining states,
      the reward function was randomly set to 0 with probability 2/3
      and to −1 with probability 1/3.
    - If the agent moves across the border, it's given the fail reward (i.e, reward=`-1`).
    - The initial state is sampled from a uniform distribution.
    """


    def __init__(self, reward_map=None, valid_states=None, seed=None, transition_prob=False, max_timestep=200, device="cuda"):
        """
        Initialize class.

        Parameters
        ----------
        reward_map : float[grid_size * grid_size]
            Reward for each state.
            Set this value when you load existing world definition (when seed=None).
        valid_states : list(int[2])
            List of states, in which the agent can reach to goal state without losing any rewards.
            Each state is a 2d vector, [row, column].
            When you call reset(), the initial state is picked up from these states.
            Set this value when you load existing world definition (when seed=None).
        seed : int
            Seed value to generate new grid (maze).
            Set this value when you create a new world.
            (Above ```reward_map``` and ```valid_states``` are newly generated.)
        transition_prob : bool
            True if transition probability (above) is enabled.
            False when we generate an expert agent without noise.
            (If transition_prob=True, it only returns next states in step() function.)
        max_timestep : int
            The maximum number of time-step (horizon).
            When it doesn't have finite horizon, set None as max_timestep.
            (If max_timestep=None, it doesn't return trunc flag in step() function.)
        device : string
            Device info ("cuda", "cpu", etc).
        """

        self.device = device
        self.transition_prob = transition_prob
        self.grid_size = 50
        self.action_size = 4
        self.max_timestep = max_timestep
        self.goal_reward = 10

        if seed is None:
            ############################
            ### Load from definition ###
            ############################
            self.reward_map = torch.tensor(reward_map).to(self.device)
            self.valid_states = torch.tensor(valid_states).to(self.device)
        else:
            ################################
            ### Generate a new GridWorld ###
            ################################
            # generate grid
            self.reward_map = torch.zeros(self.grid_size * self.grid_size, dtype=torch.int).to(self.device)
            # bottom-right is goal state
            self.reward_map[-1] = self.goal_reward
            # set reward=−1 with probability 1/3
            sample_n = np.floor((self.grid_size * self.grid_size - 1) / 3).astype(int)
            rng = np.random.default_rng(seed)
            sample_loc = rng.choice(self.grid_size * self.grid_size - 1, size=sample_n, replace=False)
            sample_loc = torch.from_numpy(sample_loc).to(self.device)
            self.reward_map[sample_loc] = -1
            # seek valid states
            valid_states_list = self._greedy_seek_valid_states([self.grid_size-1, self.grid_size-1], [])
            valid_states_list.remove([self.grid_size-1, self.grid_size-1])
            self.valid_states = torch.tensor(valid_states_list).to(self.device)

    def _greedy_seek_valid_states(self, state, old_state_list):
        """
        This method recursively seeks valid state.
        e.g, if some state is surrounded by the states with reward=-1,
        this state is invalid, because it cannot reach to the goal state
        without losing rewards.

        Parameters
        ----------
        state : int[2]
            State to start seeking. It then seeks this state and all child's states.
            This state must be the list of [row, column].
        old_state_list : int[N, 2]
            List of states already checked.
            Each state must be the list of [row, column].
            These items are then skipped for seeking.

        Returns
        ----------
        valid_states : int[N, 2]
            List of new valid states.
            Each state must be the list of [row, column].
        """
        # build new list
        new_state_list = []
        # if the state is already included in the list, do nothing
        if state in old_state_list:
            return new_state_list
        # if the state has reward=-1, do nothing
        if self.reward_map[state[0]*self.grid_size+state[1]] == -1:
            return new_state_list
        # else add the state into the list
        new_state_list.append(state)
        # move up
        if state[0] > 0:
            next_state = list(map(lambda i, j: i + j, state, [-1, 0]))
            new_state_list += self._greedy_seek_valid_states(
                next_state,
                old_state_list + new_state_list)
        # move down
        if state[0] < self.grid_size - 1:
            next_state = list(map(lambda i, j: i + j, state, [1, 0]))
            new_state_list += self._greedy_seek_valid_states(
                next_state,
                old_state_list + new_state_list)
        # move left
        if state[1] > 0:
            next_state = list(map(lambda i, j: i + j, state, [0, -1]))
            new_state_list += self._greedy_seek_valid_states(
                next_state,
                old_state_list + new_state_list)
        # move right
        if state[1] < self.grid_size - 1:
            next_state = list(map(lambda i, j: i + j, state, [0, 1]))
            new_state_list += self._greedy_seek_valid_states(
                next_state,
                old_state_list + new_state_list)
        # return result
        return new_state_list

    def reset(self, batch_size):
        """
        Randomly, get initial state (single state) from valid states.

        Parameters
        ----------
        batch_size : int
            The number of returned states.

        Returns
        ----------
        state : torch.tensor((batch_size), dtype=int)
            Return the picked-up state id.
        """
        # initialize step count
        self.step_count = 0
        # pick up sample of valid states
        indices = torch.multinomial(torch.ones(len(self.valid_states)).to(self.device), batch_size, replacement=True)
        state_2d = self.valid_states[indices]
        # convert 2d index to 1d index
        state_1d = state_2d[:,0] * self.grid_size + state_2d[:,1]
        # return result
        return state_1d

    def step(self, actions, states, trans_state_only=False, transition_prob=None):
        """
        Take action, proceed step, and return the result.

        Parameters
        ----------
        actions : torch.tensor((batch_size), dtype=int)
            Actions to take
            (0=UP 1=DOWN 2=LEFT 3=RIGHT)
        states : torch.tensor((batch_size), dtype=int)
            Current state id.
        trans_state_only : bool
            Set TRUE, when you call only for getting next state by stateless without reset()
            (If transition_prob=True, it only returns next states in step() function.)
        transition_prob : bool
            Set this property, if you overrite default ```transition_prob``` property.
            (For this property, see above in __init__() method.)

        Returns
        ----------
        new-states : torch.tensor((batch_size), dtype=int)
            New state id.
        rewards : torch.tensor((batch_size), dtype=float)
            The obtained reward.
        term : torch.tensor((batch_size), dtype=bool)
            Flag to check whether it reaches to the goal and terminates.
        trunc : torch.tensor((batch_size), dtype=bool)
            Flag to check whether it's truncated by reaching to max time-step.
            (When max_timestep is None, this is not returned.)
        """
        # get batch size
        batch_size = actions.shape[0]
        # if transition prob is enabled, apply stochastic transition
        if transition_prob is None:
            trans_prob = self.transition_prob # set default
        else:
            trans_prob = transition_prob      # overrite
        if trans_prob:
            # the action succeeds with probability 0.7
            prob = torch.ones(batch_size, self.action_size).to(self.device)
            mask = F.one_hot(actions, num_classes=self.action_size).bool()
            prob = torch.where(mask, 7.0, prob)
            selected_actions = torch.multinomial(prob, 1, replacement=True)
            selected_actions = selected_actions.squeeze(dim=1)
            action_onehot = F.one_hot(selected_actions, num_classes=self.action_size)
        else:
            # deterministic (probability=1.0 in one state)
            action_onehot = F.one_hot(actions, num_classes=self.action_size)
        # get 2d state
        mod = torch.div(states, self.grid_size, rounding_mode="floor")
        reminder = torch.remainder(states, self.grid_size)
        state_2d = torch.cat((mod.unsqueeze(dim=-1), reminder.unsqueeze(dim=-1)), dim=-1)
        # move state
        # (0=UP 1=DOWN 2=LEFT 3=RIGHT)
        up_and_down = action_onehot[:,1] - action_onehot[:,0]
        left_and_right = action_onehot[:,3] - action_onehot[:,2]
        move = torch.cat((up_and_down.unsqueeze(dim=-1), left_and_right.unsqueeze(dim=-1)), dim=-1)
        new_states = state_2d + move
        # set reward
        if not(trans_state_only):
            rewards = torch.zeros(batch_size).to(self.device)
            rewards = torch.where(new_states[:,0] < 0, -1.0, rewards)
            rewards = torch.where(new_states[:,0] >= self.grid_size, -1.0, rewards)
            rewards = torch.where(new_states[:,1] < 0, -1.0, rewards)
            rewards = torch.where(new_states[:,1] >= self.grid_size, -1.0, rewards)
        # correct location
        new_states = torch.clip(new_states, min=0, max=self.grid_size-1)
        # if succeed, add reward of current state
        states_1d = new_states[:,0] * self.grid_size + new_states[:,1]
        if not(trans_state_only):
            rewards = torch.where(rewards>=0.0, rewards+self.reward_map[states_1d], rewards)
            self.step_count += 1
        # return result
        if trans_state_only:
            return states_1d
        elif self.max_timestep is None:
            return states_1d, rewards, rewards==self.reward_map[self.grid_size * self.grid_size - 1]
        else:
            return states_1d, rewards, rewards==self.reward_map[self.grid_size * self.grid_size - 1], torch.tensor(self.step_count==self.max_timestep).to(self.device).unsqueeze(dim=0).expand(batch_size)
