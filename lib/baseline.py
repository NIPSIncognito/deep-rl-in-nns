import torch
from torch import nn


class BaselineInterface:
    def update(self, **kwargs):
        pass

    def get(self, **kwargs):
        pass


class MeanBaseline(BaselineInterface):
    """ Mean reward among sessions as baseline """
    def __init__(self, initial_baseline=torch.zeros(1), baseline_moving_average=0.05, temperature=20.):
        self.alpha = baseline_moving_average
        self.temperature = temperature
        self.baseline = initial_baseline
        self.step = 0

    def update(self, rewards, session_index, device='cpu', **kwargs):
        session_lengths = torch.bincount(session_index).to(torch.float32)
        mean_session_reward = torch.bincount(session_index, weights=rewards) \
                              / torch.max(session_lengths, torch.ones(*session_lengths.shape, device=device))
        mean_reward = mean_session_reward.mean().item()

        self.temperature = max(self.temperature - self.step, 1.0)
        alpha = self.temperature * self.alpha
        self.baseline = (1.0 - alpha) * self.baseline + alpha * mean_reward
        self.step += 1
        return mean_reward

    def get(self, device='cpu', **kwargs):
        return self.baseline.to(device=device)


class SessionBaseline(BaselineInterface):
    """ Mean reward per query as baseline """
    def __init__(self, sessions_size, baseline_moving_average=0.05, temperature=20.):
        self.alpha = baseline_moving_average
        self.temperature = temperature
        self.baseline = torch.zeros(sessions_size)
        self.updated = torch.zeros(sessions_size, dtype=torch.uint8)
        self.step = 0

    def update(self, rewards, session_index, query_index, device='cpu', **kwargs):
        session_lengths = torch.bincount(session_index).to(torch.float32)
        mean_session_reward = torch.bincount(session_index, weights=rewards) \
                              / torch.max(session_lengths, torch.ones(*session_lengths.shape, device=device))
        self.temperature = max(self.temperature - self.step, 1.0)
        alpha = self.temperature * self.alpha
        self.baseline = self.baseline.to(device=device)
        self.baseline[query_index] = (1.0 - alpha) * self.baseline[query_index] + alpha * mean_session_reward
        self.updated[query_index] = torch.ones(1, dtype=torch.uint8)
        self.baseline = self.baseline.to(device='cpu')

        if self.updated.all():
            self.updated = torch.zeros_like(self.baseline, dtype=torch.uint8)
            self.step += 1

        mean_reward = mean_session_reward.mean()
        return mean_reward.item()

    def get(self, session_index, query_index, device='cpu', **kwargs):
        return self.baseline[query_index[session_index]].to(device=device)


class SessionCritic(BaselineInterface):
    """ Critic per query as a baseline """

    Optimizer = torch.optim.Adam

    def __init__(self, queries, hidden_size, activation=nn.ELU(), **optimizer_args):
        super().__init__()
        self.queries = queries

        self.critic_network = nn.Sequential(
            nn.Linear(queries.size(1), hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, 1),
        )
        self.opt = self.Optimizer(self.critic_network.parameters(), **optimizer_args)

    def update(self, rewards, session_index, query_index, device='cpu', **kwargs):
        session_lengths = torch.bincount(session_index).to(torch.float32)
        mean_session_reward = torch.bincount(session_index, weights=rewards) \
                              / torch.max(session_lengths, torch.ones(*session_lengths.shape, device=device))

        input_nn = self.queries[query_index].to(device=device)
        self.critic_network.to(device=device)
        session_baselines = self.critic_network(input_nn).view(-1)
        loss = 0.5 * ((mean_session_reward - session_baselines) ** 2).mean()
        loss.backward()

        self.opt.step()
        self.opt.zero_grad()

        mean_reward = mean_session_reward.mean()
        return mean_reward.item()

    def get(self, session_index, query_index, device='cpu', **kwargs):
        session_lengths = torch.bincount(session_index)
        input_nn = self.queries[query_index].to(device=device)

        with torch.no_grad():
            self.critic_network.to(device=device)
            session_baselines = self.critic_network(input_nn)
            baselines = [[baseline] * length.item() for baseline, length in zip(session_baselines, session_lengths)]
            baselines = list(map(torch.cat, baselines))
        return torch.cat(baselines, dim=0).view(-1)
