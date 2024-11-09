import os

import torch
import torch.nn as nn
import torch.optim as optim


class Critic(nn.Module):
    def __init__(self, num_observations=17, num_actions=6, num_neurons=256):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_observations + num_actions, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, 1)
        )

        # Apply custom weight initialization
        self.apply(self._weights_init)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

    def predict(self, state, action, device):
        self.eval()
        state = state.to(device)
        action = action.to(device)
        with torch.no_grad():
            return self(state, action)

    def _weights_init(self, module):
        """Custom weight initializer that sets all weights and biases to zero."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def save(self, step, model_dir='models'):
        """Save the model at a given step."""
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'critic_{step}.pt')
        torch.save(self.state_dict(), model_path)
        print(f"Model saved at step {step} to {model_path}")

    @classmethod
    def load(cls, step, model_dir='models', device=None):
        """Load a critic model from the specified step."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(model_dir, f'critic_{step}.pt')
        model = cls()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded from {model_path}")
        return model