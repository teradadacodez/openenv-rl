# OpenEnv Round 1 Submission

## Meta PyTorch Hackathon - OpenEnv Challenge

This project implements an intelligent LLM agent that interacts with OpenEnv environments to maximize rewards through strategic decision-making.

---

## 🎯 Project Overview

This submission includes:
- **Inference Script**: `inference.py` - Main agent logic using LLM for decision-making
- **Server**: `server.py` - HuggingFace Space API server
- **Docker Support**: Containerized deployment
- **Agent Memory**: Learning from past actions to improve performance

---

## 📋 Requirements

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Required
HF_TOKEN=your_huggingface_token
IMAGE_NAME=your_docker_image_name

# API Configuration (defaults provided)
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# Task Configuration
MY_ENV_V4_TASK=echo
MY_ENV_V4_BENCHMARK=my_env_v4
```

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `openai` - LLM client
- `openenv-core` - Environment interface
- `flask` - Web server
- `python-dotenv` - Environment management

---

## 🚀 Quick Start

### 1. Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_token
export IMAGE_NAME=your_image

# Run inference
python inference.py
```

### 2. Docker Deployment

```bash
# Build the image
docker build -t openenv-submission .

# Run the container
docker run -p 8000:8000 \
  -e HF_TOKEN=your_token \
  -e IMAGE_NAME=your_image \
  openenv-submission
```

### 3. HuggingFace Space

Deploy to HuggingFace Spaces:
1. Create a new Space
2. Upload all files
3. Set environment variables in Space settings
4. Space will auto-deploy

---

## 🏗️ Architecture

### Agent Design

The agent uses a **memory-enhanced LLM approach**:

1. **Memory System**: Tracks action history, rewards, and best-performing actions
2. **Strategic Prompting**: Provides LLM with context about past performance
3. **Adaptive Learning**: Adjusts strategy based on reward feedback
4. **Error Handling**: Robust fallbacks for API failures

### Inference Flow

```
Start Episode
    ↓
Reset Environment
    ↓
┌─────────────────────┐
│  For each step:     │
│  1. Build context   │
│  2. Query LLM       │
│  3. Execute action  │
│  4. Update memory   │
│  5. Log results     │
└─────────────────────┘
    ↓
Calculate Score
    ↓
End Episode
```

### Output Format

The script produces standardized output:

```
[START] task=echo env=my_env_v4 model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=hello world reward=1.10 done=false error=null
[STEP] step=2 action=test message reward=1.20 done=false error=null
...
[END] success=true steps=15 score=0.850 rewards=1.10,1.20,...
```

---

## 🧠 Agent Strategy

### Key Features

1. **Context-Aware Decision Making**
   - Analyzes previous observations
   - Learns from reward patterns
   - Adapts action selection

2. **Memory Management**
   - Keeps recent history (last 5 steps)
   - Tracks best-performing actions
   - Computes running statistics

3. **Prompt Engineering**
   - Clear system instructions
   - Structured context format
   - Performance insights

4. **Robust Error Handling**
   - Fallback actions on API errors
   - Graceful environment cleanup
   - Comprehensive logging

### Hyperparameters

```python
MAX_STEPS = 15              # Maximum steps per episode
TEMPERATURE = 0.7           # LLM sampling temperature
MAX_TOKENS = 200            # Maximum response length
SUCCESS_THRESHOLD = 0.5     # Score threshold for success
```

---

## 📊 Performance Optimization

### Strategies Used

1. **Reward Maximization**
   - Analyzes correlation between actions and rewards
   - Reinforces high-reward patterns
   - Provides feedback in prompts

2. **Efficient Token Usage**
   - Concise prompts
   - Truncated history
   - Focused context

3. **Adaptive Behavior**
   - Learns from mistakes
   - Adjusts strategy mid-episode
   - Balances exploration vs exploitation

---

## 🔧 Configuration

### Model Selection

Supports any OpenAI-compatible API:

```python
# HuggingFace models
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"

# OpenAI models (if using OpenAI API)
MODEL_NAME = "gpt-4"
MODEL_NAME = "gpt-3.5-turbo"
```

### Environment Customization

Modify agent behavior in `inference.py`:

```python
# Adjust memory size
memory = AgentMemory(max_history=10)

# Change scoring
SUCCESS_SCORE_THRESHOLD = 0.7

# Tune LLM parameters
TEMPERATURE = 0.5
MAX_TOKENS = 300
```

---

## 🧪 Testing

### Unit Tests

```bash
# Run basic tests
python -m pytest tests/

# Test with mock environment
python inference.py --mock
```

### Integration Tests

```bash
# Test server endpoints
curl -X POST http://localhost:8000/reset

curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action":"test"}'
```

---

## 📁 Project Structure

```
openenv-submission/
├── inference.py          # Main agent implementation
├── server.py            # HuggingFace Space server
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── .env.example        # Environment template
├── README.md           # This file
└── validate.sh         # Validation script
```

---

## 🎓 Technical Details

### LLM Integration

Uses OpenAI Python client for compatibility:

```python
from openai import OpenAI

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[...],
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS
)
```

### Environment Interface

Async interaction pattern:

```python
# Initialize
env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)

# Reset
result = await env.reset()

# Step
result = await env.step(MyEnvV4Action(message=action))

# Cleanup
await env.close()
```

---

## 🐛 Troubleshooting

### Common Issues

**ImportError: No module named 'my_env_v4'**
- Solution: This is expected without OpenEnv installed. The script includes fallback logic.

**API Authentication Error**
- Solution: Check `HF_TOKEN` is set correctly in environment variables

**Docker Build Fails**
- Solution: Ensure Docker daemon is running and you have sufficient disk space

**Low Scores**
- Solution: Tune hyperparameters (temperature, max_tokens, threshold)
- Try different models
- Adjust memory size

---

## 📚 Resources

- [OpenEnv Documentation](https://github.com/openenv/openenv)
- [HuggingFace Spaces](https://huggingface.co/spaces)
- [OpenAI API Reference](https://platform.openai.com/docs)

---

## 🤝 Contributing

This is a hackathon submission. For improvements:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📄 License

MIT License - Feel free to use and modify

---

## 👤 Author

**Your Name**
- Hackathon: Meta PyTorch Hackathon
- Round: 1 (OpenEnv Challenge)
- Date: 2024

---

## 🎉 Acknowledgments

- Meta AI for organizing the hackathon
- OpenEnv team for the framework
- HuggingFace for hosting infrastructure
- PyTorch community

---

## 📝 Notes

### Submission Checklist

- [x] `inference.py` in root directory
- [x] Uses OpenAI client for LLM calls
- [x] Correct stdout format ([START], [STEP], [END])
- [x] Environment variables documented
- [x] Docker configuration included
- [x] HuggingFace Space compatible
- [x] README with clear instructions
- [x] Requirements.txt complete

### Validation

Run the validation script:

```bash
./validate.sh https://your-space.hf.space
```

Should pass all 3 checks:
1. HF Space is live
2. Docker build succeeds
3. OpenEnv validate passes

---

**Good luck with the hackathon! 🚀**
