# Task Definition: RL Project - Juliette
**Projet**: DQN vs Stable-Baselines Comparison + Safety-Aware Extension  

---

## PART 1: CORE TASK (Baseline Comparison)

### 1.1 DQN Implementation
- [x] **Custom DQN Agent Implemented** — FINI (Simon)
  - Dueling DQN architecture
  - Double DQN with target network
  - Replay buffer
  - Location: [src/rlproject/dqn_agent.py](src/rlproject/dqn_agent.py)
  - Status: FINI

### 1.2 Training Infrastructure
- [x] **Shared Configuration** — FINI (Simon)
  - Environment: highway-v0
  - Observation: Kinematics
  - Actions: DiscreteMetaAction
  - Config file: [src/rlproject/shared_core_config.py](src/rlproject/shared_core_config.py)
  - Status: FINI

- [x] **Training Scripts** — FINI (Simon)
  - Custom DQN training: [scripts/train_dqn_custom.py](scripts/train_dqn_custom.py)
  - Stable-Baselines3 training: [scripts/train_stable_baselines.py](scripts/train_stable_baselines.py)
  - Checkpoints saved in: `checkpoints_custom/` and `checkpoints_sb3/`
  - Status: FINI

### 1.3 Quantitative Evaluation (50 runs per seed)
- [x] **Multi-Seed Evaluation** — FINI (Juliette)
  - Seeds: 0, 1, 3 (at least 3 as required) 
  - Episodes per seed: 50
  - Script: [scripts/evaluate_multiseed.py](scripts/evaluate_multiseed.py)
  - Results table with mean ± std:

| Seed | Custom Mean | Custom Std | SB3 Mean | SB3 Std |
|------|-------------|-----------|----------|---------|
| 0    | 18.76       | 5.07      | 10.24    | 5.66    |
| 1    | 19.37       | 4.47      | 11.28    | 5.94    |
| 3    | 16.94       | 6.47      | 10.59    | 6.80    |
| Overall | 18.36    | 5.50      | 10.70    | 6.17    |

  - Output files: `comparison_results/multiseed_eval_*.json` and `*.md`
  - Status: FINI

### 1.4 Training Curves
- [x] **Generate Training Curves** — FINI (Simon + Juliette)
  - Script: [scripts/plot_training_curves.py](scripts/plot_training_curves.py)
  - Input: `metrics.json` from `checkpoints_custom/` and `checkpoints_sb3/`
  - Output: PNG plots in `comparison_results/`
  - Metrics plotted: episodic rewards, moving averages
  - Status: FINI

### 1.5 Qualitative Analysis
- [ ] **Behavior Analysis** — EN COURS (Juliette)
  - Describe typical behavior of Custom DQN vs SB3
  - Identify common decision patterns (lane changes, acceleration, etc.)
  - File: [analysis/qualitative_behavior.md](analysis/qualitative_behavior.md) 
  - Status: EN COURS 

- [x] **Recorded Rollouts** — FINI (Juliette)
  - Script: [scripts/record_rollout.py](scripts/record_rollout.py)
  - One rollout per agent (Custom DQN, SB3)
  - Seeds: 0, 1, 3
  - Output: Videos in `rollouts/` directory
  - Status: FINI (videos recorded, waiting for viewing/analysis)

### 1.6 Failure Mode Analysis
- [ ] **Identify and Analyze One Failure Mode** — EN COURS (Juliette)
  - Watch recorded rollouts and identify recurring issues
  - Possible failure modes: collisions, inefficient lane changes, getting stuck, etc.
  - Provide detailed explanation of **why** it happens (reward shaping, exploration, etc.)
  - File: [analysis/rollout_notes.md](analysis/rollout_notes.md)
  - Status: EN COURS

### 1.7 Comparison: Custom vs SB3
- [x] **Fair Comparison Protocol** — FINI (Juliette)
  - Same config (shared_core_config.py)
  - Same seeds for evaluation
  - Same number of episodes (50 per seed)
  - Same metrics (mean reward, std)
  - Analysis in: `comparison_results/multiseed_eval_*.md`
  - Status: FINI

### 1.8 Documentation (Core Results)
- [ ] **Summary Document** — A FAIRE
  - Short summary/discussion of design choices
  - Key insights: architecture choices, training stability, etc.
  - Status: A FAIRE

---

## PART 2: EXTENSION — Safety-Aware Reward Shaping

### Hypothesis & Rationale
**Hypothesis**: Adding safety penalties to the reward function during training will produce agents that achieve comparable performance to the baseline while maintaining **lower collision rates and more stable behavior**, thus creating an advantageous safety-performance tradeoff.

**Rationale**:
- Highway-v0 has implicit safety costs (collisions), but the default reward doesn't explicitly penalize crashes
- Safety-Aware RL is increasingly important for autonomous driving applications
- Reward shaping is a **substantive extension** (not just tuning): new training runs, new metrics, comparative analysis
- Measurable impact: we can quantify collision rates, safety metrics, and performance tradeoff

### 2.1 Reward Shaping Implementation
- [ ] **Define Safety-Aware Reward Function** — A FAIRE
  - Baseline reward: original highway-v0 reward
  - Safety-Aware reward: baseline - λ * collision_penalty - μ * crash_indicator
  - Create two variants:
    - Conservative (high safety weight): λ=10, μ=50
    - Moderate (balanced): λ=5, μ=30
  - Document the formulation: [src/rlproject/reward_shaper.py](src/rlproject/reward_shaper.py) (new file)
  - Status: A FAIRE

- [ ] **Modify DQN Training Loop** — A FAIRE
  - Create `train_dqn_safety_aware.py` script
  - Use reward_shaper in training loop
  - Train three versions:
    1. Baseline DQN (no shaping) [already exists]
    2. DQN-Conservative (high safety penalty)
    3. DQN-Moderate (balanced)
  - Save checkpoints: `checkpoints_safety_aware_conservative/` and `checkpoints_safety_aware_moderate/`
  - Save metrics with additional safety metrics (crash count, near-miss events)
  - Status: A FAIRE

### 2.2 Safety Metrics Definition
- [ ] **Implement Safety Evaluation Metrics** — A FAIRE
  - Collision rate: % of episodes with >= 1 collision
  - Mean collisions per episode
  - Min distance to obstacles (proxy for safety margin)
  - Reward per episode (performance)
  - Create utility functions in: [src/rlproject/safety_metrics.py](src/rlproject/safety_metrics.py) (new file)
  - Status: A FAIRE

### 2.3 Evaluation of Safety-Aware Variants
- [ ] **Run Evaluation on All Variants** — A FAIRE
  - Evaluate all three agents:
    1. Baseline custom DQN
    2. DQN-Conservative (safety-aware)
    3. DQN-Moderate (safety-aware)
  - Protocol: same 3 seeds (0, 1, 3), 50 episodes per seed
  - Metrics: reward (mean, std), collision_rate, mean_crashes, safety_margin
  - Output table: `comparison_results/safety_aware_eval_*.md`
  - Status: A FAIRE

- [ ] **Generate Comparative Plots** — A FAIRE
  - Training curves for all three agents (same figure)
  - Safety metrics vs performance tradeoff (scatter plot or curves)
  - Collision rate comparison (bar chart)
  - Output: `comparison_results/safety_aware_*.png`
  - Status: A FAIRE

### 2.4 Behavior Analysis for Safety-Aware Agents
- [ ] **Qualitative Observation** — A FAIRE
  - Record rollouts for both safety-aware variants (same seeds as baseline)
  - Qualitatively describe differences:
    - Is the agent more conservative (fewer lane changes)?
    - Does it maintain larger safety margins?
    - Are failure modes different (fewer collisions but lower rewards)?
  - Document in: [analysis/safety_aware_behavior.md](analysis/safety_aware_behavior.md) (new file)
  - Status: A FAIRE

- [ ] **Failure Mode Comparison** — A FAIRE
  - Compare failure modes of baseline vs safety-aware agents:
    - Baseline: what crashes/issues emerge when not penalizing safety?
    - Safety-aware: same issues? New failure modes? (e.g., overly cautious?)
  - Provide 1-2 concrete examples with video frames if possible
  - File: [analysis/safety_aware_failure_modes.md](analysis/safety_aware_failure_modes.md) (new file)
  - Status: A FAIRE

### 2.5 Generalization Experiment (Optional but Recommended)
- [ ] **Test Generalization to Modified Traffic Density** — A FAIRE (OPTIONAL)
  - Evaluate all agents (baseline + safety-aware variants) on:
    - Original density (baseline)
    - High density (more vehicles, tighter spacing)
    - Low density (fewer vehicles, safer)
  - Hypothesis: safety-aware agents generalize better to diverse traffic scenarios
  - Results file: `comparison_results/generalization_eval_*.md`
  - Status: A FAIRE (OPTIONAL)

### 2.6 Results & Analysis
- [ ] **Two Key Results** — A FAIRE
  1. **Result 1**: Safety-Performance Tradeoff Table
     - Show results table with columns: Agent | Mean Reward | Std | Collision Rate | Safety Margin
     - Clearly indicate the tradeoff (does higher safety come at performance cost?)
  
  2. **Result 2**: Comparative Plots
     - Training curves overlay (learning speed comparison)
     - Scatter plot: mean reward vs collision rate (visualize tradeoff)

  - File: [comparison_results/EXTENSION_RESULTS.md](comparison_results/EXTENSION_RESULTS.md)
  - Status: A FAIRE

### 2.7 Extension Discussion
- [ ] **Write Extension Analysis** — A FAIRE
  - What worked: which reward shaping scheme was effective?
  - What didn't: unexpected results or failure modes in safety-aware training?
  - Insights: what does the safety-performance tradeoff tell us about the learning?
  - Limitations: hyperparameter sensitivity, generalization limits, etc.
  - Future directions: continuous actions for finer safety control, curriculum learning, etc.
  - File: [analysis/extension_discussion.md](analysis/extension_discussion.md)
  - Status: A FAIRE

---

## PART 3: FINAL DELIVERABLES

### 3.1 Core Task Summary
- [ ] **Core Results Summary** — A FAIRE
  - 1-2 page summary of baseline comparison
  - Table: performance metrics
  - Key finding: Custom DQN outperforms SB3
  - One failure mode example
  - File: `CORE_RESULTS_SUMMARY.md` or section in README
  - Status: A FAIRE

### 3.2 Extension Summary
- [ ] **Extension Results Summary** — A FAIRE
  - 1 page on safety-aware extension
  - Key result: safety-performance tradeoff curve/table
  - Interpretation and implications
  - File: `EXTENSION_RESULTS_SUMMARY.md` or section in README
  - Status: A FAIRE

### 3.3 README Update
- [ ] **Update Main README** — A FAIRE
  - Add "Results" section with:
    - Core task findings (link to summary)
    - Extension findings (link to summary)
    - Quick how-to for running each experiment
  - File: [README.md](README.md)
  - Status: A FAIRE

---

## Key References

- Highway-v0 documentation: https://highway-env.readthedocs.io/
- Reward shaping literature: "Potential-Based Reward Shaping" (Ng et al., 1999)
- Safety in RL: "Safe Reinforcement Learning in Constrained Markov Decision Processes" (various)
- Dueling DQN: Wang et al., 2015
- Double DQN: Van Hasselt et al., 2015
