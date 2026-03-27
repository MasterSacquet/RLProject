# dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class QNetwork(nn.Module):
    """
    Réseau de neurones pour approximer les Q-values
    Architecture avec Dueling DQN et normalisation
    Input -> Dense(256) -> BatchNorm -> ReLU -> Dense(256) -> BatchNorm -> ReLU -> Output
    """
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        
        # Couche de feature extraction
        self.fc1 = nn.Linear(state_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Dueling Architecture: split en value et advantage streams
        # Value stream
        self.value_fc = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(256, 128)
        self.advantage = nn.Linear(128, action_dim)
        
    def forward(self, state):
        # Feature extraction
        x = torch.relu(self.bn1(self.fc1(state)))
        x = torch.relu(self.bn2(self.fc2(x)))
        
        # Value stream
        value = torch.relu(self.value_fc(x))
        value = self.value(value)
        
        # Advantage stream
        advantage = torch.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
    """
    Agent DQN amélioré avec Double DQN, Dueling Networks et stratégie d'exploration adaptée
    """
    def __init__(self, state_dim, action_dim, learning_rate=5e-4, gamma=0.99):
        """
        Args:
            state_dim: Dimension de l'observation (50 pour highway-v0)
            action_dim: Nombre d'actions (3 pour highway-v0)
            learning_rate: Taux d'apprentissage (réduit pour plus de stabilité)
            gamma: Facteur de discount
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Epsilon-greedy avec décroissance adaptée
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.99  # Décroissance plus agressif pour forcer l'exploitation
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Réseaux Q
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # Mode évaluation
        
        # Optimiseur avec weight decay pour régularisation
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss()
        
        # Paramètres d'entraînement améliorés
        self.batch_size = 32  # Réduit pour plus de mises à jour graduelles
        self.update_frequency = 50  # Update target network plus fréquemment
        self.step_count = 0
        
        # Learning rate scheduling - sera appelé UNE FOIS par épisode
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        
    def select_action(self, state, action_space):
        """
        Sélection d'action avec epsilon-greedy
        
        Args:
            state: Observation courante
            action_space: Espace d'action (gym.spaces.Discrete)
        
        Returns:
            action: Action sélectionnée
        """
        # Conversion en tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).flatten().unsqueeze(0).to(self.device)
        else:
            state_tensor = state
        
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            # Exploration: action aléatoire
            return action_space.sample()
        else:
            # Exploitation: meilleure action selon Q-network
            with torch.no_grad():
                # Mettre le réseau en mode évaluation pour éviter les problèmes de BatchNorm
                self.q_net.eval()
                q_values = self.q_net(state_tensor)
                self.q_net.train()  # Remettre en mode training
                return q_values.argmax(dim=1).item()
    
    def train(self, replay_buffer):
        """
        Entraînement du DQN avec double DQN et priorité sur la stabilité
        
        Args:
            replay_buffer: Instance de ReplayBuffer
        
        Returns:
            loss: Valeur de la loss pour ce batch, ou None si le buffer n'est pas assez grand
        """
        # Attendre que le buffer ait assez d'expériences
        if len(replay_buffer) < self.batch_size * 2:  # Plus d'données d'initialisation
            return None
        
        # Mettre le réseau en mode training
        self.q_net.train()
        
        # Sampling batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)
        
        # Conversion en tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ===== Double DQN avec Dueling Networks =====
        # 1. Sélectionner meilleure action avec Q-network courant (main network)
        with torch.no_grad():
            next_q_values = self.q_net(next_states)
            best_actions = next_q_values.argmax(dim=1, keepdim=True)
            
            # 2. Évaluer avec target network (pour moins de surapprentissage)
            target_q_values = self.target_net(next_states)
            max_next_q = target_q_values.gather(1, best_actions)
            
            # Bellman target avec clip optionnel pour stabilité
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
            target_q = torch.clamp(target_q, min=-10, max=10)  # Clip pour éviter les gradients explosifs
        
        # Prédiction avec le réseau courant
        current_q = self.q_net(states).gather(1, actions)
        
        # Loss et backprop avec gradient clipping
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clipping de gradient pour éviter les explosions de gradients
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # NE PAS appeler scheduler.step() ici!
        # Le scheduler sera appelé UNE FOIS par épisode via step_scheduler()
        
        # Update target network moins fréquemment pour plus de stabilité
        self.step_count += 1
        if self.step_count % self.update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """
        Décrémenter epsilon une fois par épisode
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def step_scheduler(self):
        """
        Étape du learning rate scheduler - APPELÉ UNE FOIS PAR ÉPISODE
        Ne pas appeler à chaque step d'optimisation pour éviter une décroissance chaotique
        """
        self.scheduler.step()
