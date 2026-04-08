# OpenEnv Round 1 - Technical Documentation

## Architecture Deep Dive

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                   HuggingFace Space                     │
│  ┌────────────────────────────────────────────────┐   │
│  │              Flask Server (server.py)          │   │
│  │  - /reset endpoint                             │   │
│  │  - /step endpoint                              │   │
│  │  - /health endpoint                            │   │
│  └────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                            │
                            │ HTTP API
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Inference Script (inference.py)            │
│  ┌────────────────────────────────────────────────┐   │
│  │            Agent Core                          │   │
│  │  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │    Memory    │  │  LLM Client  │          │   │
│  │  │   System     │  │  (OpenAI)    │          │   │
│  │  └──────────────┘  └──────────────┘          │   │
│  │          │                  │                  │   │
│  │          └──────┬───────────┘                  │   │
│  │                 │                              │   │
│  │          ┌──────▼──────┐                       │   │
│  │          │  Decision   │                       │   │
│  │          │   Engine    │                       │   │
│  │          └─────────────┘                       │   │
│  └────────────────────────────────────────────────┘   │
│                         │                              │
│                         │ Actions                      │
│                         ▼                              │
│  ┌────────────────────────────────────────────────┐   │
│  │        Environment Interface                   │   │
│  │        (my_env_v4)                             │   │
│  └────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Agent Memory System

The `AgentMemory` class provides:

1. **Short-term Memory**: Last N steps (configurable)
2. **Performance Tracking**: Running totals, averages
3. **Best Action Caching**: Identifies highest-reward actions
4. **Context Generation**: Formats history for LLM

```python
class AgentMemory:
    - history: List[Dict]           # Recent steps
    - total_reward: float           # Cumulative reward
    - best_action: str              # Top performing action
    - best_reward: float            # Highest reward seen
    
    Methods:
    - add_step()                    # Record new step
    - get_summary()                 # Format history
    - get_insights()                # Generate statistics
```

### Decision Flow

```
Step N begins
    │
    ├─► Build Context
    │   ├─ Current observation
    │   ├─ Last reward
    │   ├─ Memory summary
    │   └─ Performance insights
    │
    ├─► Query LLM
    │   ├─ System prompt (strategy)
    │   ├─ User prompt (context)
    │   └─ Temperature/tokens config
    │
    ├─► Parse Response
    │   ├─ Extract action string
    │   ├─ Clean formatting
    │   └─ Validate output
    │
    ├─► Execute in Environment
    │   └─ env.step(action)
    │
    └─► Update State
        ├─ Record to memory
        ├─ Append to rewards
        └─ Log step output
```

## Prompt Engineering

### System Prompt Design

The system prompt establishes:
- Agent identity and goal
- Environment understanding
- Strategic guidelines
- Output format requirements

### User Prompt Structure

Each user prompt includes:

1. **Current State**
   - Step number
   - Latest observation
   - Most recent reward

2. **Performance Context**
   - Total reward accumulated
   - Average reward per step
   - Best action identified

3. **Historical Context**
   - Last N steps with rewards
   - Patterns observed
   - Learning opportunities

### Example Prompt Flow

**System:**
```
You are an intelligent agent interacting with an environment.
Your task is to maximize cumulative reward...
```

**User (Step 5):**
```
STEP 5

Current Observation:
echoed_message: "hello world"

Last Reward: 1.10

Current Performance:
- Total reward so far: 5.50
- Average reward per step: 1.10
- Best reward achieved: 1.30

Recent History:
Step 1: 'hi' -> reward +0.30
Step 2: 'test message' -> reward +1.20
Step 3: 'another test' -> reward +1.30
Step 4: 'hello world' -> reward +1.10

What is your next action?
```

## Scoring Algorithm

### Reward Collection

```python
rewards = []  # Store all step rewards

for step in range(1, MAX_STEPS + 1):
    result = env.step(action)
    reward = result.reward or 0.0
    rewards.append(reward)
```

### Normalization

Score is normalized to [0, 1] range:

```python
total_reward = sum(rewards)
max_possible = MAX_STEPS * MAX_REWARD_PER_STEP

score = total_reward / max_possible
score = min(max(score, 0.0), 1.0)  # Clamp
```

### Success Criteria

```python
success = score >= SUCCESS_SCORE_THRESHOLD

# Default threshold: 0.5
# Means agent must achieve ≥50% of maximum possible reward
```

## Output Specification

### Format Requirements

All output must follow this exact format:

#### START Line
```
[START] task=<task_name> env=<benchmark> model=<model_name>
```

#### STEP Lines
```
[STEP] step=<N> action=<action_str> reward=<X.XX> done=<true|false> error=<msg|null>
```

Rules:
- `step`: Integer, 1-indexed
- `action`: Action string (sanitized, max 100 chars for display)
- `reward`: Float with 2 decimal places
- `done`: Lowercase boolean (`true` or `false`)
- `error`: Error message or `null`

#### END Line
```
[END] success=<true|false> steps=<N> score=<X.XXX> rewards=<r1,r2,...,rN>
```

Rules:
- `success`: Lowercase boolean
- `steps`: Total steps taken
- `score`: Float with 3 decimal places, in [0, 1]
- `rewards`: Comma-separated list, 2 decimals each

### Example Output

```
[START] task=echo env=my_env_v4 model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=hello world reward=1.10 done=false error=null
[STEP] step=2 action=testing the environment reward=2.30 done=false error=null
[STEP] step=3 action=maximize reward with longer messages reward=4.10 done=false error=null
[END] success=true steps=3 score=0.850 rewards=1.10,2.30,4.10
```

## Error Handling

### Levels of Fallback

1. **LLM API Failure**
   ```python
   try:
       response = client.chat.completions.create(...)
   except Exception:
       return "fallback action"  # Default action
   ```

2. **Environment Step Failure**
   ```python
   try:
       result = await env.step(action)
   except Exception:
       # Log error, use last known state
   ```

3. **Environment Close Failure**
   ```python
   finally:
       try:
           await env.close()
       except Exception:
           # Log but continue to END output
   ```

### Guaranteed END Output

The script ALWAYS produces an END line:

```python
finally:
    # This block ALWAYS executes
    try:
        await env.close()
    except:
        pass  # Don't let close() prevent END
    
    log_end(success, steps, score, rewards)
```

## Performance Optimization

### Token Efficiency

- **Truncate history**: Keep only last 5 steps
- **Concise observations**: Limit observation strings
- **Focused prompts**: No unnecessary verbosity

### Inference Speed

- **No streaming**: Use synchronous completion
- **Cached client**: Reuse OpenAI client
- **Async environment**: Non-blocking env operations

### Memory Usage

- **Bounded history**: Fixed-size lists
- **String limits**: Truncate long texts
- **Cleanup**: Proper resource disposal

## Environment Variables Reference

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | `hf_xxxxx` |
| `IMAGE_NAME` | Docker image name | `my-env:latest` |

### Optional (with defaults)

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `MY_ENV_V4_TASK` | `echo` | Task name for logging |
| `MY_ENV_V4_BENCHMARK` | `my_env_v4` | Benchmark name |

## Testing Strategy

### Local Testing

```bash
# 1. Mock environment test
python test_local.py

# 2. Server test
python server.py &
curl -X POST http://localhost:7860/reset

# 3. Integration test
./validate.sh https://localhost:7860
```

### Validation Checks

The `validate.sh` script verifies:

1. **Space Liveness**: POST to /reset returns 200
2. **Docker Build**: Dockerfile builds successfully
3. **OpenEnv Compliance**: Passes `openenv validate`

## Deployment Guide

### HuggingFace Space Setup

1. **Create Space**
   - Go to huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Docker" SDK

2. **Upload Files**
   ```
   ├── Dockerfile
   ├── server.py
   ├── inference.py
   ├── requirements.txt
   └── README_HF.md (rename to README.md)
   ```

3. **Configure Secrets**
   - Settings → Repository secrets
   - Add `HF_TOKEN`
   - Add `IMAGE_NAME`

4. **Deploy**
   - Space auto-builds from Dockerfile
   - Check logs for errors
   - Test /reset endpoint

### Troubleshooting Deployment

**Build fails**
- Check Dockerfile syntax
- Verify all files uploaded
- Review build logs

**Space crashes**
- Check environment variables set
- Review server.py logs
- Verify port 7860 exposed

**API errors**
- Confirm HF_TOKEN valid
- Check MODEL_NAME available
- Test API_BASE_URL accessible

## Advanced Configuration

### Custom Environment

To use a different environment:

```python
# In inference.py
from custom_env import CustomEnv, CustomAction

# Update initialization
env = await CustomEnv.from_docker_image(IMAGE_NAME)
result = await env.step(CustomAction(...))
```

### Different LLM Provider

```python
# For OpenAI
API_BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "gpt-4"
API_KEY = os.getenv("OPENAI_API_KEY")

# For Azure OpenAI
API_BASE_URL = "https://your-resource.openai.azure.com/"
MODEL_NAME = "your-deployment-name"
```

### Hyperparameter Tuning

Key parameters to adjust:

```python
# Exploration vs Exploitation
TEMPERATURE = 0.7  # Higher = more random
                   # Lower = more deterministic

# Response length
MAX_TOKENS = 200   # Longer responses
                   # = more detailed actions

# Memory depth
max_history = 5    # More history
                   # = better context
                   # = more tokens

# Success threshold
SUCCESS_SCORE_THRESHOLD = 0.5  # Lower = easier success
                                # Higher = stricter
```

## Code Quality

### Best Practices Implemented

- ✅ Type hints throughout
- ✅ Docstrings for all functions
- ✅ Error handling with try/except
- ✅ Resource cleanup with finally
- ✅ Logging with flush=True
- ✅ Environment variable validation
- ✅ Async/await properly used

### Code Structure

```
inference.py (~350 lines)
├── Imports & Configuration  (50 lines)
├── Logging Functions       (30 lines)
├── AgentMemory Class       (80 lines)
├── Prompt Building         (40 lines)
├── LLM Integration         (40 lines)
└── Main Execution Logic    (110 lines)
```

## Metrics & Monitoring

### What Gets Logged

- Episode start (task, env, model)
- Each step (step #, action, reward, done, error)
- Episode end (success, steps, score, all rewards)
- Debug messages (API failures, cleanup errors)

### Example Log Output

```
[START] task=echo env=my_env_v4 model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=hello reward=0.50 done=false error=null
[STEP] step=2 action=testing reward=0.70 done=false error=null
[DEBUG] Model request failed: Connection timeout
[STEP] step=3 action=fallback reward=0.80 done=false error=null
[END] success=true steps=3 score=0.667 rewards=0.50,0.70,0.80
```

## Future Enhancements

Potential improvements:

1. **Multi-turn reasoning**: Chain-of-thought prompting
2. **Reward prediction**: Estimate reward before action
3. **Action caching**: Remember successful patterns
4. **Dynamic temperature**: Adjust based on confidence
5. **Ensemble methods**: Multiple LLM queries
6. **Fine-tuning**: Train on episode history

---

**End of Technical Documentation**
