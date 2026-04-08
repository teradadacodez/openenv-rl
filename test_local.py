#!/usr/bin/env python3
"""
Local Testing Script for OpenEnv Agent
Simulates the environment for local development without Docker
"""

import asyncio
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class MockObservation:
    """Mock observation for testing."""
    echoed_message: str
    step_count: int = 0


@dataclass
class MockResult:
    """Mock result from environment step."""
    observation: MockObservation
    reward: float
    done: bool
    error: Optional[str] = None


class MockEnv:
    """Mock environment for local testing."""
    
    def __init__(self):
        self.step_count = 0
        self.max_steps = 15
    
    async def reset(self):
        """Reset the environment."""
        self.step_count = 0
        return MockResult(
            observation=MockObservation(echoed_message="", step_count=0),
            reward=0.0,
            done=False
        )
    
    async def step(self, action):
        """Execute a step."""
        self.step_count += 1
        message = action.message
        
        # Calculate reward based on message length
        reward = len(message) * 0.1
        
        # Check if done
        done = self.step_count >= self.max_steps
        
        return MockResult(
            observation=MockObservation(
                echoed_message=message,
                step_count=self.step_count
            ),
            reward=reward,
            done=done
        )
    
    async def close(self):
        """Close the environment."""
        pass
    
    @staticmethod
    async def from_docker_image(image_name: str):
        """Create mock environment (replaces Docker initialization)."""
        return MockEnv()


@dataclass
class MockAction:
    """Mock action."""
    message: str


def setup_mock_env():
    """Setup mock environment for testing."""
    import sys
    from types import ModuleType
    
    # Create mock module
    mock_module = ModuleType('my_env_v4')
    mock_module.MyEnvV4Env = MockEnv
    mock_module.MyEnvV4Action = MockAction
    
    # Add to sys.modules
    sys.modules['my_env_v4'] = mock_module
    
    print("[TEST] Mock environment loaded")


async def test_inference():
    """Test the inference script with mock environment."""
    
    print("=" * 60)
    print("OpenEnv Local Testing")
    print("=" * 60)
    print()
    
    # Set test environment variables
    os.environ['IMAGE_NAME'] = 'mock-image'
    os.environ['HF_TOKEN'] = 'mock-token'
    os.environ['API_BASE_URL'] = 'https://router.huggingface.co/v1'
    os.environ['MODEL_NAME'] = 'Qwen/Qwen2.5-72B-Instruct'
    
    # Setup mock environment
    setup_mock_env()
    
    # Import and run inference
    print("[TEST] Running inference script...")
    print()
    
    try:
        # Import the main module
        import inference
        
        # Run the main function
        await inference.main()
        
        print()
        print("=" * 60)
        print("[TEST] Test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"[TEST] Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🧪 Starting local test...\n")
    asyncio.run(test_inference())
