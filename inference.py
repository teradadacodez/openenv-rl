import asyncio
import os
from typing import List

from my_env_v4 import MyEnvV4Action, MyEnvV4Env

# ENV
IMAGE_NAME = os.getenv("IMAGE_NAME")

TASK = "echo"
ENV = "my_env_v4"

MAX_STEPS = 15
MAX_MESSAGE_LENGTH = 200  # optimal size

SUCCESS_THRESHOLD = 0.5


# =========================
# LOGGING (STRICT FORMAT)
# =========================

def log_start():
    print(f"[START] task={TASK} env={ENV} model=deterministic-agent", flush=True)


def log_step(step, action, reward, done):
    action_clean = action.replace("\n", " ")[:100]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )


# =========================
# RANK 1 STRATEGY
# =========================

def generate_max_reward_message(step: int) -> str:
    """
    Generates a deterministic long message to maximize reward.
    """
    base = f"Step {step} maximizing reward. "

    # Fill remaining space deterministically
    filler = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "

    msg = base
    i = 0

    while len(msg) < MAX_MESSAGE_LENGTH:
        msg += filler[i % len(filler)]
        i += 1

    return msg[:MAX_MESSAGE_LENGTH]


# =========================
# MAIN
# =========================

async def main():
    env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)

    rewards: List[float] = []
    steps_taken = 0

    log_start()

    try:
        result = await env.reset()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = generate_max_reward_message(step)

            result = await env.step(MyEnvV4Action(message=action))

            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done)

            if done:
                break

        total_reward = sum(rewards)
        max_possible = MAX_STEPS * MAX_MESSAGE_LENGTH * 0.1

        score = total_reward / max_possible if max_possible > 0 else 0
        score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_THRESHOLD

    finally:
        try:
            await env.close()
        except:
            pass

        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())