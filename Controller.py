"""Inspect the Tetris environment and report one seeded random episode."""
import cv2
import gymnasium as gym
import numpy as np

import sys

import tetris_gymnasium.envs  # Registers Tetris so gym.make() can find it.

# Keep the run settings together so they are easy to find and change.
ENVIRONMENT_SEED = 42
ACTION_SEED = 123
MAX_STEPS = 500


def main() -> None:
    """Set up the environment, run one episode, and print its results."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")

    try:
        # The game and the random action sampler have separate randomness.
        # Seed each once, before the episode starts.
        observation, info = env.reset(seed=ENVIRONMENT_SEED)
        env.action_space.seed(ACTION_SEED)

        print(f"Action space: {env.action_space}")
        print(f"Starting info: {info}")
        print("Starting observation fields:")
        for field_name, field_array in observation.items():
            print(
                f"  {field_name}: "
                f"shape={field_array.shape}, dtype={field_array.dtype}"
            )

        # These totals belong to the whole episode, so initialize them once.
        total_reward = 0.0
        total_lines_cleared = 0
        episode_steps = 0
        episode_over = False
        terminated = False
        truncated = False

        while not episode_over:
            # In rgb_array mode, render() returns pixels; it opens no window.
            if episode_steps == 0:
                frame = env.render()
                assert isinstance(frame, np.ndarray)
                print(frame.shape, frame.dtype)
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                output_path = r"C:\MLproject\captures\step_0.png"
                output_bi_path = r"C:\MLproject\captures\step_0.npy"
                saved = cv2.imwrite(output_path, bgr_frame)
                np.save(output_bi_path, observation["board"])
                loaded_board = np.load(output_bi_path)
                print(saved)

                if np.array_equal(loaded_board, observation["board"]):
                    print(loaded_board.shape , loaded_board.dtype)


            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            # Each step reports only its own reward and cleared lines.
            total_reward += float(reward)
            total_lines_cleared += info["lines_cleared"]
            episode_steps += 1

            if episode_steps == 5:
                frame = env.render()
                assert isinstance(frame, np.ndarray)
                print(frame.shape, frame.dtype)
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                output_path = r"C:\MLproject\captures\step_5.png"
                output_bi_path = r"C:\MLproject\captures\step_5.npy"
                saved = cv2.imwrite(output_path, bgr_frame)
                np.save(output_bi_path, observation["board"])
                loaded_board = np.load(output_bi_path)
                print(saved)

                if np.array_equal(loaded_board, observation["board"]):
                    print(loaded_board.shape , loaded_board.dtype)

            # Our step cap can stop a run even if the environment has not ended.
            episode_over = (
                terminated or truncated or episode_steps >= MAX_STEPS
            )


        # Preserve the environment's ending reason if it also hits our cap.
        if terminated:
            end_reason = "terminated"
        elif truncated:
            end_reason = "truncated"
        else:
            end_reason = "step_cap"

        print(
            "\nEpisode summary\n"
            f"  Total reward: {total_reward}\n"
            f"  Total steps: {episode_steps}\n"
            f"  Lines cleared: {total_lines_cleared}\n"
            f"  End reason: {end_reason}\n"
            f"  Environment seed: {ENVIRONMENT_SEED}\n"
            f"  Action seed: {ACTION_SEED}"
        )
    finally:
        # Release environment resources even if an earlier operation fails.
        env.close()


if __name__ == "__main__":
    main()
