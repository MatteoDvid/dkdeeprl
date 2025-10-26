# 🎮 PROJET DONKEY KONG - DEEP REINFORCEMENT LEARNING
## Plan Ultra-Optimisé pour un Projet de Classe Mondiale

---

## 📋 TABLE DES MATIÈRES

1. [Vue d'ensemble](#1-vue-densemble)
2. [Choix stratégiques](#2-choix-stratégiques)
3. [Architecture technique](#3-architecture-technique)
4. [Pipeline de preprocessing](#4-pipeline-de-preprocessing)
5. [Algorithme principal: DQN Amélioré](#5-algorithme-principal-dqn-amélioré)
6. [Stratégies d'optimisation](#6-stratégies-doptimisation)
7. [Métriques et évaluation](#7-métriques-et-évaluation)
8. [Timeline d'implémentation](#8-timeline-dimplémentation)
9. [Structure du rapport](#9-structure-du-rapport)
10. [Innovations clés](#10-innovations-clés)

---

## 1. VUE D'ENSEMBLE

### 1.1 Objectif Principal
Créer un agent capable de maîtriser **Donkey Kong** (Atari) en utilisant Deep Q-Learning avec des améliorations modernes, surpassant les performances baseline et démontrant une compréhension approfondie du Deep RL.

### 1.2 Environnement Cible
- **Jeu**: Donkey Kong (ALE/DonkeyKong-v5)
- **Type**: Atari 2600 via Gymnasium
- **Input**: Frames RGB 210x160x3
- **Actions**: 18 actions discrètes (Atari standard)
- **Objectif**: Maximiser le score en évitant les obstacles et en collectant les bonus

### 1.3 Différenciateurs du Projet
1. **Double DQN** avec Prioritized Experience Replay
2. **Dueling Network Architecture** pour séparer value/advantage
3. **Noisy Networks** pour l'exploration (remplace epsilon-greedy)
4. **Rainbow-inspired** optimizations
5. **Curriculum Learning** (si applicable)
6. **Visualisations avancées** (heatmaps d'attention, trajectoires)

---

## 2. CHOIX STRATÉGIQUES

### 2.1 Algorithme Principal: **Double DQN + Dueling**

**Pourquoi Double DQN?**
- Réduit l'overestimation des Q-values (problème du DQN standard)
- Plus stable et converge mieux sur Atari
- Prouvé dans les benchmarks originaux DeepMind

**Pourquoi Dueling Architecture?**
- Sépare la valeur de l'état et l'avantage de l'action
- Apprend plus efficacement (surtout quand beaucoup d'actions ont des Q-values similaires)
- Amélioration de 20-30% sur Donkey Kong selon les benchmarks

**Alternatives explorées**:
- PPO: Plus complexe à stabiliser sur Atari, meilleur pour continuous control
- A3C: Requiert multi-threading, plus difficile à débugger
- REINFORCE: Trop de variance pour Atari

### 2.2 Améliorations Avancées

**Prioritized Experience Replay (PER)**
- Sample plus fréquemment les transitions avec high TD-error
- Accélère l'apprentissage de 2-3x sur Atari
- Importance sampling pour corriger le biais

**Noisy Networks**
- Paramètres stochastiques dans les couches FC
- Exploration automatique sans epsilon decay
- Plus consistant que epsilon-greedy

**Multi-Step Learning (n-step returns)**
- Utilise n=3 steps pour les returns
- Bootstrap plus rapide
- Meilleur que 1-step sur Donkey Kong

---

## 3. ARCHITECTURE TECHNIQUE

### 3.1 Network Architecture: Dueling DQN CNN

```
INPUT: 84x84x4 (4 frames grayscale stacked)
    ↓
CONV1: 32 filters, 8x8, stride 4, ReLU
    → Output: 20x20x32
    ↓
CONV2: 64 filters, 4x4, stride 2, ReLU
    → Output: 9x9x64
    ↓
CONV3: 64 filters, 3x3, stride 1, ReLU
    → Output: 7x7x64
    ↓
FLATTEN: 3136 features
    ↓
    ┌──────────────────┴──────────────────┐
    ↓                                      ↓
VALUE STREAM:                      ADVANTAGE STREAM:
FC1: 512 neurons, ReLU            FC1: 512 neurons, ReLU
    ↓                                      ↓
FC2: 1 output (V(s))              FC2: n_actions outputs (A(s,a))
    ↓                                      ↓
    └──────────────────┬──────────────────┘
                       ↓
    AGGREGATION: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
                       ↓
                  OUTPUT: Q-values
```

**Paramètres**:
- Total params: ~2.3M
- Trainable params: ~2.3M
- Memory per batch (32): ~5MB

### 3.2 Composants Système

```
┌─────────────────────────────────────────────────┐
│             AGENT ARCHITECTURE                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────┐      ┌─────────────┐          │
│  │   ONLINE    │      │   TARGET    │          │
│  │   NETWORK   │      │   NETWORK   │          │
│  │  (training) │      │  (frozen)   │          │
│  └──────┬──────┘      └──────┬──────┘          │
│         │                    │                  │
│         │ soft update        │                  │
│         │ every τ steps      │                  │
│         ↓                    ↓                  │
│    Q-values              Target Q-values        │
│                                                 │
├─────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────┐   │
│  │  PRIORITIZED EXPERIENCE REPLAY BUFFER   │   │
│  │  Capacity: 100k transitions             │   │
│  │  Priority: |TD-error| + ε               │   │
│  └─────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────┐   │
│  │  FRAME PREPROCESSING PIPELINE           │   │
│  │  RGB → Gray → Resize → Normalize → Stack│   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

---

## 4. PIPELINE DE PREPROCESSING

### 4.1 Frame Preprocessing

```python
def preprocess_frame(frame):
    """
    Input: RGB frame (210, 160, 3) uint8
    Output: Grayscale frame (84, 84) float32 [0, 1]
    """
    # 1. Convert to grayscale (weighted average)
    gray = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]

    # 2. Crop to relevant game area (remove score, etc.)
    # Donkey Kong specific: crop top 30 pixels
    cropped = gray[30:, :]  # (180, 160)

    # 3. Resize to 84x84 using bilinear interpolation
    resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)

    # 4. Normalize to [0, 1]
    normalized = resized / 255.0

    return normalized.astype(np.float32)
```

### 4.2 Frame Stacking

**Motivation**: Donner l'information temporelle (vitesse, direction)

```python
class FrameStack:
    def __init__(self, num_frames=4):
        self.frames = deque(maxlen=num_frames)
        self.num_frames = num_frames

    def reset(self, initial_frame):
        # Fill with copies of first frame
        processed = preprocess_frame(initial_frame)
        for _ in range(self.num_frames):
            self.frames.append(processed)

    def update(self, new_frame):
        processed = preprocess_frame(new_frame)
        self.frames.append(processed)

    def get_state(self):
        # Returns (4, 84, 84) numpy array
        return np.stack(list(self.frames), axis=0)
```

### 4.3 Reward Shaping (Optionnel mais Recommandé)

**Problème**: Les rewards dans Donkey Kong sont rares (sparse rewards)

**Solution**: Reward Shaping

```python
def shape_reward(raw_reward, info, prev_info):
    """
    Ajoute des rewards intermédiaires pour accélérer l'apprentissage
    """
    shaped_reward = raw_reward

    # Bonus pour progresser verticalement
    if 'y_position' in info:
        delta_y = info['y_position'] - prev_info.get('y_position', 0)
        shaped_reward += 0.01 * delta_y  # Small bonus for climbing

    # Penalty pour rester immobile (encourage exploration)
    if raw_reward == 0:
        shaped_reward -= 0.001

    # Clip final reward [-1, 1] pour stabilité
    return np.clip(shaped_reward, -1.0, 1.0)
```

**Note**: Documenter dans le rapport si on utilise reward shaping!

---

## 5. ALGORITHME PRINCIPAL: DQN AMÉLIORÉ

### 5.1 Double DQN Loss Function

**Standard DQN**:
```
y_t = r_t + γ * max_a' Q_target(s_{t+1}, a')
Loss = MSE(Q_online(s_t, a_t), y_t)
```

**Double DQN** (meilleur):
```
# Action selection: use online network
a* = argmax_a Q_online(s_{t+1}, a)

# Q-value evaluation: use target network
y_t = r_t + γ * Q_target(s_{t+1}, a*)

Loss = MSE(Q_online(s_t, a_t), y_t)
```

**Avantage**: Réduit l'overestimation en découplant selection et evaluation

### 5.2 Prioritized Experience Replay

**Priority Calculation**:
```python
# TD-error
δ = |r + γ * Q_target(s', a') - Q_online(s, a)|

# Priority
priority = (|δ| + ε)^α
```

**Sampling Probability**:
```python
P(i) = priority_i^α / Σ_k priority_k^α
```

**Importance Sampling Weights**:
```python
# Correct for bias introduced by non-uniform sampling
w_i = (N * P(i))^(-β)
w_i = w_i / max_j(w_j)  # Normalize

# Loss with importance weights
loss = w_i * (Q - target)^2
```

**Hyperparamètres**:
- α = 0.6 (how much prioritization is used)
- β = 0.4 → 1.0 (annealed over training, compensates for bias)
- ε = 1e-6 (small constant to ensure non-zero priority)

### 5.3 Pseudo-code Complet

```python
# INITIALIZATION
online_network = DuelingDQN(state_shape, n_actions)
target_network = DuelingDQN(state_shape, n_actions)
target_network.load_state_dict(online_network.state_dict())

replay_buffer = PrioritizedReplayBuffer(capacity=100000)
optimizer = Adam(online_network.parameters(), lr=0.00025, eps=1e-4)

epsilon = 1.0  # Start with exploration
beta = 0.4  # IS weight

# TRAINING LOOP
for episode in range(num_episodes):
    frame = env.reset()
    frame_stack.reset(frame)
    state = frame_stack.get_state()
    episode_reward = 0

    for step in range(max_steps):
        # ACTION SELECTION (epsilon-greedy or noisy nets)
        if random.random() < epsilon:
            action = random.choice(range(n_actions))
        else:
            with torch.no_grad():
                q_values = online_network(state)
                action = q_values.argmax().item()

        # ENVIRONMENT STEP
        next_frame, reward, done, info = env.step(action)
        frame_stack.update(next_frame)
        next_state = frame_stack.get_state()

        # STORE TRANSITION (with max priority initially)
        replay_buffer.add(state, action, reward, next_state, done)

        # TRAINING STEP (after warmup)
        if len(replay_buffer) > batch_size and step % 4 == 0:
            # Sample batch with priorities
            batch, indices, weights = replay_buffer.sample(batch_size, beta)
            states, actions, rewards, next_states, dones = batch

            # Compute current Q-values
            current_q = online_network(states).gather(1, actions)

            # Compute target Q-values (Double DQN)
            with torch.no_grad():
                # Select action with online network
                next_actions = online_network(next_states).argmax(1, keepdim=True)
                # Evaluate with target network
                next_q = target_network(next_states).gather(1, next_actions)
                target_q = rewards + gamma * next_q * (1 - dones)

            # Compute TD-error and update priorities
            td_errors = torch.abs(current_q - target_q)
            replay_buffer.update_priorities(indices, td_errors.detach().cpu())

            # Weighted loss (importance sampling)
            loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_network.parameters(), 10)
            optimizer.step()

        # TARGET NETWORK UPDATE (soft update every τ steps)
        if step % target_update_freq == 0:
            target_network.load_state_dict(online_network.state_dict())

        # ANNEAL EXPLORATION
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        beta = min(1.0, beta + beta_increment)

        state = next_state
        episode_reward += reward

        if done:
            break

    # LOG METRICS
    print(f"Episode {episode}: Reward={episode_reward}, Epsilon={epsilon:.3f}")
```

---

## 6. STRATÉGIES D'OPTIMISATION

### 6.1 Hyperparamètres Recommandés

**Network Architecture**:
```python
config = {
    # Environment
    'env_name': 'ALE/DonkeyKong-v5',
    'frame_skip': 4,
    'repeat_action_probability': 0.0,

    # Preprocessing
    'frame_shape': (84, 84),
    'frame_stack': 4,
    'reward_clip': True,
    'reward_clip_range': (-1, 1),

    # Network
    'conv_filters': [32, 64, 64],
    'conv_kernels': [8, 4, 3],
    'conv_strides': [4, 2, 1],
    'fc_hidden': 512,

    # DQN
    'replay_buffer_size': 100000,
    'batch_size': 32,
    'gamma': 0.99,
    'learning_rate': 0.00025,
    'adam_eps': 1e-4,
    'gradient_clip': 10.0,

    # Training
    'num_episodes': 2000,
    'warmup_steps': 50000,
    'train_frequency': 4,  # Train every 4 steps
    'target_update_freq': 1000,  # Hard update every 1000 steps

    # Exploration
    'epsilon_start': 1.0,
    'epsilon_final': 0.01,
    'epsilon_decay_steps': 100000,  # Linear decay

    # Prioritized Replay
    'use_per': True,
    'per_alpha': 0.6,
    'per_beta_start': 0.4,
    'per_beta_end': 1.0,
    'per_beta_steps': 100000,
    'per_epsilon': 1e-6,

    # Checkpointing
    'save_freq': 100,  # Save every 100 episodes
    'eval_freq': 50,   # Evaluate every 50 episodes
    'eval_episodes': 5,
}
```

### 6.2 Techniques de Stabilisation

**1. Gradient Clipping**
```python
torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)
```

**2. Huber Loss (au lieu de MSE)**
```python
# Plus robuste aux outliers
loss = F.smooth_l1_loss(current_q, target_q)
```

**3. Learning Rate Schedule**
```python
# Decay learning rate après convergence initiale
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
```

**4. Target Network Updates**
```python
# Option 1: Hard update every N steps
if step % target_update_freq == 0:
    target_net.load_state_dict(online_net.state_dict())

# Option 2: Soft update (Polyak averaging) - plus stable
τ = 0.001
for target_param, param in zip(target_net.parameters(), online_net.parameters()):
    target_param.data.copy_(τ * param.data + (1 - τ) * target_param.data)
```

### 6.3 Accélération du Training

**1. Mixed Precision Training**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**2. Parallel Environment Vectorization**
```python
from gymnasium.vector import AsyncVectorEnv

envs = AsyncVectorEnv([make_env for _ in range(4)])
# Run 4 environments in parallel
```

**3. DataLoader pour Replay Buffer**
```python
from torch.utils.data import DataLoader, Dataset

class ReplayDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

dataloader = DataLoader(dataset, batch_size=32, num_workers=2)
```

---

## 7. MÉTRIQUES ET ÉVALUATION

### 7.1 Métriques Primaires

**Performance**:
- Episode reward (score du jeu)
- Moving average (window=100)
- Max reward achieved
- Success rate (episodes complétés)

**Learning Efficiency**:
- Episodes to reach threshold (score > 1000)
- Training time to convergence
- Frames per second (FPS)
- Sample efficiency (reward per frame)

**Stability**:
- Reward variance across episodes
- Loss convergence (TD-error)
- Q-value estimates over time

### 7.2 Visualisations à Créer

**1. Training Curves**
```python
plots = {
    'rewards': {
        'raw': episode_rewards,
        'moving_avg': moving_average(episode_rewards, 100),
        'best': max(episode_rewards[:i+1])
    },
    'loss': td_errors,
    'epsilon': epsilon_history,
    'q_values': avg_q_values,
}
```

**2. Performance Heatmap**
```python
# Heatmap de la distribution des rewards
import seaborn as sns

reward_matrix = np.reshape(episode_rewards, (50, 40))
sns.heatmap(reward_matrix, cmap='RdYlGn')
plt.title('Reward Evolution over Training')
```

**3. Action Distribution**
```python
# Histogramme des actions prises
action_counts = Counter(actions_taken)
plt.bar(action_counts.keys(), action_counts.values())
plt.title('Action Distribution')
```

**4. Attention Heatmap (via GradCAM)**
```python
# Où le réseau "regarde" dans l'image
gradcam = GradCAM(model, target_layer='conv3')
heatmap = gradcam(state)
overlay = overlay_heatmap(frame, heatmap)
plt.imshow(overlay)
```

**5. Trajectoires dans l'espace d'état**
```python
# UMAP/t-SNE des états visités
from umap import UMAP

embeddings = model.get_embeddings(states)
reduced = UMAP(n_components=2).fit_transform(embeddings)
plt.scatter(reduced[:, 0], reduced[:, 1], c=rewards, cmap='viridis')
```

### 7.3 Comparaisons Obligatoires

**Baselines**:
1. Random agent (baseline inférieur)
2. DQN standard (sans améliorations)
3. Double DQN (notre algo)
4. Human expert (si disponible dans literature)

**Tableau Comparatif**:
```
┌──────────────────┬───────────┬───────────┬──────────┬─────────┐
│ Algorithm        │ Avg Score │ Max Score │ Episodes │ Time(h) │
├──────────────────┼───────────┼───────────┼──────────┼─────────┤
│ Random           │    45     │    120    │   1000   │   0.5   │
│ DQN              │   850     │   2100    │   1500   │   12    │
│ Double DQN       │  1240     │   3500    │   1200   │   10    │
│ Double DQN + PER │  1680     │   4200    │   1000   │   11    │
│ Human Expert     │  2500     │   8000    │    -     │    -    │
└──────────────────┴───────────┴───────────┴──────────┴─────────┘
```

---

## 8. TIMELINE D'IMPLÉMENTATION

### 8.1 Phase 1: Setup (2-3 jours)

**Objectifs**:
- ✅ Installer dépendances (Gymnasium, PyTorch, ALE)
- ✅ Vérifier environnement Donkey Kong fonctionne
- ✅ Créer structure de projet
- ✅ Implémenter frame preprocessing
- ✅ Tester frame stacking

**Livrables**:
- `environment.py`: Wrapper pour Donkey Kong
- `preprocessing.py`: FrameStack, preprocess_frame
- Test notebook validant preprocessing

### 8.2 Phase 2: DQN Baseline (3-4 jours)

**Objectifs**:
- ✅ Implémenter Dueling DQN network
- ✅ Implémenter standard replay buffer
- ✅ Créer DQN agent (epsilon-greedy)
- ✅ Training loop basique
- ✅ Checkpointing et logging

**Livrables**:
- `networks.py`: DuelingDQN architecture
- `replay_buffer.py`: ExperienceReplayBuffer
- `agent.py`: DQNAgent class
- `train.py`: Training loop
- Premier modèle entraîné (baseline)

### 8.3 Phase 3: Améliorations Avancées (3-4 jours)

**Objectifs**:
- ✅ Implémenter Prioritized Experience Replay
- ✅ Double DQN (target network)
- ✅ Hyperparameter tuning
- ✅ Optimisations (gradient clipping, Huber loss)
- ✅ Training complet (2000 episodes)

**Livrables**:
- `prioritized_replay.py`: PrioritizedReplayBuffer
- `agent_advanced.py`: DoubleDQNAgent avec PER
- Modèles finaux entraînés
- Checkpoints à différents stades

### 8.4 Phase 4: Évaluation & Visualisation (2-3 jours)

**Objectifs**:
- ✅ Évaluer agent final (100 episodes)
- ✅ Créer toutes les visualisations
- ✅ Comparer avec baselines
- ✅ Enregistrer vidéos de gameplay
- ✅ Analyser les résultats

**Livrables**:
- `evaluate.py`: Script d'évaluation
- `visualize.py`: Génération des plots
- Tous les plots pour le rapport
- Vidéos de gameplay
- Fichier de métriques (JSON/CSV)

### 8.5 Phase 5: Rapport Final (2-3 jours)

**Objectifs**:
- ✅ Rédiger rapport (4-6 pages)
- ✅ Inclure toutes les figures
- ✅ Tableaux de résultats
- ✅ Discussion et analyse
- ✅ Relecture et polish

**Livrables**:
- `rapport.pdf`: Rapport final
- `README.md`: Documentation du projet
- Code commenté et propre

**TOTAL: 12-17 jours** (2.5 - 3.5 semaines)

---

## 9. STRUCTURE DU RAPPORT

### 9.1 Plan Détaillé (4-6 pages)

**1. Introduction (0.5 page)**
- Contexte du Deep RL
- Présentation de Donkey Kong
- Objectifs du projet
- Contributions clés

**2. Background & Related Work (1 page)**
- Q-Learning et value-based methods
- Deep Q-Networks (DQN)
- Améliorations: Double DQN, Dueling Architecture, PER
- État de l'art sur Atari benchmarks

**3. Methodology (2 pages)**

**3.1 Environment & Preprocessing**
- Description de l'environnement Donkey Kong
- State space (frames 84x84x4)
- Action space (18 actions discrètes)
- Reward structure
- Preprocessing pipeline (diagramme)

**3.2 Algorithm**
- Architecture du réseau (schéma CNN + Dueling)
- Double DQN loss function
- Prioritized Experience Replay
- Pseudo-code simplifié

**3.3 Hyperparameters**
- Tableau complet des hyperparamètres
- Justification des choix

**4. Experiments & Results (1.5 pages)**

**4.1 Training Results**
- Courbes de reward (raw + moving average)
- Convergence de la loss
- Évolution des Q-values
- Action distribution

**4.2 Baseline Comparisons**
- Tableau comparatif (Random, DQN, Double DQN, etc.)
- Graphiques de comparaison

**4.3 Qualitative Analysis**
- Vidéos/screenshots de gameplay
- Stratégies apprises par l'agent
- Attention heatmaps (optionnel)

**5. Discussion (0.5 page)**
- Analyse des résultats
- Limitations
- Difficultés rencontrées
- Pistes d'amélioration

**6. Conclusion (0.3 page)**
- Résumé des contributions
- Performance finale
- Leçons apprises

**7. References (0.2 page)**
- Papers clés: DQN (Mnih 2015), Double DQN, Dueling DQN, Rainbow
- Documentation Gymnasium/ALE

### 9.2 Figures Obligatoires

1. **Figure 1**: Architecture du réseau (Dueling DQN)
2. **Figure 2**: Pipeline de preprocessing
3. **Figure 3**: Training curves (reward over episodes)
4. **Figure 4**: Loss convergence
5. **Figure 5**: Baseline comparison (bar chart)
6. **Figure 6**: Screenshots de gameplay (avant/après training)
7. **Figure 7**: Action distribution histogram
8. **Table 1**: Hyperparameters
9. **Table 2**: Performance comparison

---

## 10. INNOVATIONS CLÉS

### 10.1 Ce Qui Rendra le Projet Exceptionnel

**1. Dueling Architecture + Double DQN + PER**
- Très peu d'étudiants implementent les 3 ensemble
- Démontre une maîtrise avancée des techniques modernes

**2. Visualisations Avancées**
- Heatmaps d'attention (GradCAM)
- UMAP des états explorés
- Analyses qualitatives poussées

**3. Comparaisons Rigoureuses**
- Ablation studies (avec/sans chaque amélioration)
- Statistiques significatives (mean ± std sur 5 runs)

**4. Code Propre & Reproductible**
- Structure modulaire
- Configuration files (YAML/JSON)
- Scripts d'entraînement automatisés
- Documentation complète
- Checkpoints partagés

**5. Bonus Points**
- Curriculum learning (si applicable)
- Transfer learning (pré-train sur autre jeu?)
- Ensemble methods
- Intrinsic motivation / curiosity

### 10.2 Différenciateurs vs Autres Projets

**La plupart des étudiants feront**:
- DQN simple sur CartPole/MountainCar
- Pas de preprocessing avancé
- Visualisations basiques
- Rapport minimal

**Votre projet aura**:
- Jeu Atari complexe (Donkey Kong)
- Algorithme state-of-the-art (Double DQN + Dueling + PER)
- Preprocessing pipeline complet
- Visualisations publication-quality
- Analyse approfondie
- Code professionnel

---

## 11. STRUCTURE DES FICHIERS

```
projetintermediatedeepl/
│
├── README.md                    # Documentation principale
├── plan.md                      # Ce fichier
├── requirements.txt             # Dépendances
├── config.yaml                  # Hyperparamètres
│
├── main.py                      # Point d'entrée
├── train.py                     # Script de training
├── evaluate.py                  # Script d'évaluation
│
├── src/
│   ├── __init__.py
│   ├── environment.py           # Wrapper Donkey Kong
│   ├── preprocessing.py         # Frame preprocessing
│   ├── networks.py              # DuelingDQN architecture
│   ├── replay_buffer.py         # Standard replay buffer
│   ├── prioritized_replay.py    # PER buffer
│   ├── agent.py                 # DQN agent
│   ├── agent_advanced.py        # Double DQN + PER agent
│   ├── trainer.py               # Training loop
│   ├── evaluator.py             # Evaluation logic
│   └── utils.py                 # Utilities
│
├── notebooks/
│   ├── 01_environment_exploration.ipynb
│   ├── 02_preprocessing_test.ipynb
│   ├── 03_training_visualization.ipynb
│   └── 04_results_analysis.ipynb
│
├── visualizations/
│   ├── plots.py                 # Plotting functions
│   ├── heatmaps.py              # Attention visualizations
│   └── videos.py                # Video recording
│
├── experiments/
│   ├── baseline/                # DQN simple
│   ├── double_dqn/              # Double DQN
│   └── full_model/              # Double DQN + PER
│
├── checkpoints/                 # Saved models
│   ├── checkpoint_ep100.pth
│   ├── checkpoint_ep500.pth
│   └── best_model.pth
│
├── logs/                        # Training logs
│   ├── tensorboard/
│   └── metrics.json
│
├── results/                     # Résultats finaux
│   ├── figures/                 # Plots pour le rapport
│   ├── videos/                  # Gameplay videos
│   └── metrics/                 # CSV/JSON des métriques
│
├── report/                      # Rapport LaTeX
│   ├── rapport.tex
│   ├── rapport.pdf
│   └── figures/
│
└── tests/                       # Unit tests
    ├── test_environment.py
    ├── test_preprocessing.py
    └── test_networks.py
```

---

## 12. RÉFÉRENCES CLÉS

### 12.1 Papers Fondamentaux

1. **DQN Original**
   - Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
   - Nature, 518(7540), 529-533

2. **Double DQN**
   - van Hasselt et al. (2016) - "Deep Reinforcement Learning with Double Q-learning"
   - AAAI 2016

3. **Dueling DQN**
   - Wang et al. (2016) - "Dueling Network Architectures for Deep Reinforcement Learning"
   - ICML 2016

4. **Prioritized Experience Replay**
   - Schaul et al. (2016) - "Prioritized Experience Replay"
   - ICLR 2016

5. **Rainbow**
   - Hessel et al. (2018) - "Rainbow: Combining Improvements in Deep Reinforcement Learning"
   - AAAI 2018

### 12.2 Ressources Techniques

- Gymnasium Documentation: https://gymnasium.farama.org/
- Atari Learning Environment: https://github.com/mgbellemare/Arcade-Learning-Environment
- PyTorch RL Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- CleanRL (clean implementations): https://github.com/vwxyzjn/cleanrl

---

## 13. CHECKLIST FINALE

### Avant de Soumettre

**Code**:
- [ ] Tous les scripts s'exécutent sans erreur
- [ ] Code commenté et documenté
- [ ] Requirements.txt complet
- [ ] README.md avec instructions
- [ ] Checkpoints sauvegardés

**Résultats**:
- [ ] Agent entraîné pendant 2000+ episodes
- [ ] Scores moyens > 1000 (cible raisonnable)
- [ ] 3+ baselines implémentées et comparées
- [ ] Toutes les métriques calculées

**Visualisations**:
- [ ] Training curves (reward, loss, Q-values)
- [ ] Baseline comparisons
- [ ] Action distributions
- [ ] Gameplay videos
- [ ] Heatmaps/attention maps

**Rapport**:
- [ ] 4-6 pages (format PDF)
- [ ] Toutes les sections complètes
- [ ] Figures de haute qualité
- [ ] Références correctes
- [ ] Relecture orthographe/grammaire

---

## 14. NEXT STEPS

### Immédiatement (Session Actuelle)

1. **Setup Project Structure**
   ```bash
   mkdir -p src notebooks visualizations experiments checkpoints logs results report tests
   touch src/__init__.py
   ```

2. **Install Dependencies**
   ```bash
   pip install gymnasium[atari] torch torchvision numpy matplotlib seaborn opencv-python tensorboard
   pip install ale-py autorom[accept-rom-license]
   ```

3. **Test Environment**
   ```python
   import gymnasium as gym
   env = gym.make('ALE/DonkeyKong-v5', render_mode='human')
   env.reset()
   for _ in range(1000):
       env.step(env.action_space.sample())
   env.close()
   ```

4. **Start with Preprocessing**
   - Implémenter `preprocessing.py`
   - Tester dans notebook

### Session Suivante

5. **Implement DuelingDQN Network**
6. **Basic DQN Agent**
7. **Training Loop v1**

---

## 15. QUESTIONS À CLARIFIER AVEC LE PROF

1. Limite de temps de computation? (peut affecter le nombre d'episodes)
2. Accès à GPU? (crucial pour Atari)
3. Format du rapport (LaTeX, Word)?
4. Peut-on travailler à plusieurs? (projet Atari → jusqu'à 4 personnes)
5. Deadline exacte?

---

## CONCLUSION

Ce plan couvre un projet de Deep RL de niveau professionnel sur Donkey Kong. En suivant cette roadmap, tu auras:

✅ Un agent performant utilisant des techniques state-of-the-art
✅ Des visualisations publication-quality
✅ Un code propre et reproductible
✅ Un rapport académique solide
✅ Des résultats comparatifs rigoureux

**Let's build the best Donkey Kong RL agent ever! 🚀🍌**
