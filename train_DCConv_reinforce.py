import os
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import logging
from datetime import datetime
import yaml

# Import the HELP RL environment and the new model.
from HELPRL_Env import HELP_RL_ENV
from DCConv_reinforce import HierarchicalReinforce

def load_config(config_path="config_help.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# Set up logging.
log_folder = "help_logs/hierarchical_reinforce_DCConv"
os.makedirs(log_folder, exist_ok=True)
log_filename = os.path.join(
    log_folder,
    f"hierarchical_reinforce_DCConv_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
)
logger = logging.getLogger("HierarchicalReinforce_DCConv_Training")

# Device configuration.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the environment.
env = HELP_RL_ENV(env_params=config['hierarchical_reinforce_training'])
batch_size = config['hierarchical_reinforce_training']['batch_size']

# Initialize the HierarchicalReinforce model
model = HierarchicalReinforce(
    input_dim=2 + config['hierarchical_reinforce_training']['num_demand_types'],
    hidden_dim=config['hierarchical_reinforce_training']['hidden_dim'],
    encoder_layers=config['hierarchical_reinforce_training']['encoder_layers'],
    n_heads=config['hierarchical_reinforce_training']['n_heads'],
    num_help_hubs=config['hierarchical_reinforce_training']['num_help_hubs'],
    num_vehicle_types=config['hierarchical_reinforce_training']['num_vehicle_types'],
    num_demand_types=config['hierarchical_reinforce_training']['num_demand_types'],
    travel_time_mean=config['hierarchical_reinforce_training']['travel_time_mean'],
    travel_time_std=config['hierarchical_reinforce_training']['travel_time_std'],
    num_DCC_layers=config['hierarchical_reinforce_training'].get('num_DCC_layers', 4)
).to(device)

optimizer = optim.Adam(model.parameters(), lr=float(config['hierarchical_reinforce_training']['lr']))
num_epochs = config['hierarchical_reinforce_training']['num_epochs']

for epoch in range(num_epochs):
    env.load_problems(batch_size)
    with torch.no_grad():
        reset_state, _, _ = env.reset()
    reset_state.main_hub_xy = reset_state.main_hub_xy.to(device)
    reset_state.help_hub_xy = reset_state.help_hub_xy.to(device)
    reset_state.demands = reset_state.demands.to(device)
    
    with torch.no_grad():
        step_state, _, done = env.pre_step()
    step_state.vehicle_capacities = step_state.vehicle_capacities.to(device)
    current_location = step_state.current_location.to(device)
    
    log_probs = []
    values = []
    rewards_episode = None
    
    while not done:
        action, log_prob, value = model.act(
            reset_state.main_hub_xy,
            reset_state.help_hub_xy,
            reset_state.demands,
            current_location,
            step_state.vehicle_capacities
        )
        log_probs.append(log_prob)
        values.append(value)
        
        with torch.no_grad():
            step_state, step_reward, done = env.step(action)
        step_state.vehicle_capacities = step_state.vehicle_capacities.to(device)
        current_location = step_state.current_location.to(device)
        
        if done:
            rewards_episode = step_reward.to(device)
    
    log_probs_tensor = torch.stack(log_probs, dim=0)
    values_tensor = torch.stack(values, dim=0)
    sum_log_probs = log_probs_tensor.sum(dim=0)
    
    baseline = values_tensor[-1]
    advantage = rewards_episode - baseline.detach()
    
    policy_loss = -(sum_log_probs * advantage).mean()
    critic_loss = torch.mean((baseline - rewards_episode) ** 2)
    loss = policy_loss + config['hierarchical_reinforce_training'].get('critic_loss_weight', 0.5) * critic_loss
    
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), config['hierarchical_reinforce_training']['max_grad_norm'])
    optimizer.step()
    
    torch.cuda.empty_cache()
    
    if epoch % config['hierarchical_reinforce_training']['print_interval'] == 0:
        avg_reward = rewards_episode.mean().item()
        logger.info('Epoch %4d | Loss: %7.4f | Policy Loss: %7.4f | Critic Loss: %7.4f | Avg Reward: %7.4f',
                    epoch, loss.item(), policy_loss.item(), critic_loss.item(), avg_reward)

# Save the trained model.
torch.save(model.state_dict(), 'hierarchical_reinforce_DCConv.pth')
