# =========================
# BASE IMAGE
# =========================
FROM python:3.10-slim

# =========================
# WORKDIR
# =========================
WORKDIR /app

# =========================
# COPY FILES
# =========================
COPY . /app

# =========================
# INSTALL DEPENDENCIES
# =========================
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# =========================
# ENV VARIABLES (SAFE DEFAULTS)
# =========================
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# =========================
# EXPOSE PORT (HF REQUIRES 7860)
# =========================
EXPOSE 7860

# =========================
# RUN SERVER
# =========================
CMD ["python", "server.py"]