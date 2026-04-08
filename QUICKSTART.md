# 🚀 Quick Start Guide

Get your OpenEnv agent running in 5 minutes!

---

## ⚡ Fastest Path to Running

### 1️⃣ Clone & Setup (30 seconds)

```bash
# Clone the repo
git clone <your-repo-url>
cd openenv-submission

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configure Environment (1 minute)

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your values
nano .env  # or use your favorite editor
```

Required values:
```bash
HF_TOKEN=hf_your_token_here
IMAGE_NAME=your_docker_image_name
```

### 3️⃣ Run Locally (30 seconds)

```bash
# Test with mock environment
python test_local.py

# Or run actual inference (requires OpenEnv setup)
python inference.py
```

### 4️⃣ Deploy to HuggingFace (3 minutes)

```bash
# 1. Create a new Space on huggingface.co/spaces
# 2. Upload all files to the Space
# 3. Add HF_TOKEN and IMAGE_NAME as secrets
# 4. Space auto-deploys!
```

---

## 🎯 What You Get

After setup, you have:
- ✅ LLM-powered agent (inference.py)
- ✅ HuggingFace Space server (server.py)
- ✅ Docker containerization
- ✅ Validation scripts
- ✅ Local testing capabilities

---

## 📋 Pre-Submission Checklist

- [ ] inference.py in root directory
- [ ] Environment variables configured
- [ ] Docker builds successfully
- [ ] HF Space responds to /reset
- [ ] Validation script passes
- [ ] README is clear

---

## 🆘 Quick Troubleshooting

**Import Error: my_env_v4**
→ Use `python test_local.py` for local testing

**API Authentication Failed**
→ Check HF_TOKEN validity and permissions

**Docker Build Fails**
→ Verify Docker daemon running and files present

**Low Scores**
→ Adjust TEMPERATURE, MAX_TOKENS, or model

---

## 💡 Pro Tips

1. Start with `python test_local.py`
2. Validate with `./validate.sh <space-url>`
3. Monitor HF Space logs for debugging
4. Test endpoints: `curl -X POST <space>/reset`
5. Iterate on hyperparameters

---

Ready to submit? Run `./validate.sh` first! 🎉
