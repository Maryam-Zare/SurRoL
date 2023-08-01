"""
Data generation for the case of Psm Envs and demonstrations.
Refer to
https://github.com/openai/baselines/blob/master/baselines/her/experiment/data_generation/fetch_data_generation.py
"""
import os
import argparse
import gym
import time
import numpy as np
import imageio
from surrol.const import ROOT_DIR_PATH

parser = argparse.ArgumentParser(description='generate demonstrations for imitation')
parser.add_argument('--env', type=str, required=True,
                    help='the environment to generate demonstrations')
parser.add_argument('--video', action='store_true',
                    help='whether or not to record video')
parser.add_argument('--steps', type=int,
                    help='how many steps allowed to run')
args = parser.parse_args()

actions = []
observations = []
truncated = []
dones = []
rewards = []

images = []  # record video
masks = []


def main():
    env = gym.make(args.env, render_mode='human')  # 'human'
    num_itr = 150 if not args.video else 1
    cnt = 0
    init_state_space = 'square'
    env.reset()
    print("Reset!")
    init_time = time.time()

    if args.steps is None:
        args.steps = env._max_episode_steps

    print()
    #while len(actions) < num_itr:
    while cnt < num_itr:
        obs = env.reset()
        print("ITERATION NUMBER ", cnt)
        goToGoal(env, obs)
        cnt += 1

    file_name = "data_"
    file_name += args.env
    file_name += "_" + init_state_space
    file_name += "_" + str(cnt)
    file_name += ".npz"

    folder = 'demo' if not args.video else 'video'
    folder = os.path.join(ROOT_DIR_PATH, 'data', folder)
    

    dataset = {'observations' : observations, 'actions': actions,
    'rewards':rewards,
    'dones':dones}
    
               
    

# check the directory does not exist
    if not(os.path.exists(folder)):
    # create the directory you want to save to
        os.mkdir(folder)
    
    np.savez_compressed(os.path.join(folder, file_name), **dataset)  # save the file

    if args.video:
        video_name = "video_"
        video_name += args.env + ".mp4"
        writer = imageio.get_writer(os.path.join(folder, video_name), fps=20)
        for img in images:
            writer.append_data(img)
        writer.close()

        if len(masks) > 0:
            mask_name = "mask_"
            mask_name += args.env + ".npz"
            np.savez_compressed(os.path.join(folder, mask_name),
                                masks=masks)  # save the file

    used_time = time.time() - init_time
    print("Saved data at:", folder)
    print("Time used: {:.1f}m, {:.1f}s\n".format(used_time // 60, used_time % 60))
    print(f"Trials: {num_itr}/{cnt}")
    env.close()


def goToGoal(env, last_obs):
    episode_acs = []
    episode_obs = []
    episode_info = []
    episode_dones = []
    episode_rewards = []

    time_step = 0  # count the total number of time steps
    episode_init_time = time.time()
    episode_obs.append(last_obs)

    obs, success = last_obs, False

    while time_step < min(env._max_episode_steps, args.steps):
        action = env.get_oracle_action(obs)
        if args.video:
            # img, mask = env.render('img_array')
            img = env.render('rgb_array')
            images.append(img)
            # masks.append(mask)

        obs, reward, done, info = env.step(action)
        #print(f" -> obs: {obs}, reward: {reward}, done: {done}, info: {info}.")
        time_step += 1

        if isinstance(obs, dict) and info['is_success'] > 0 and not success:
            print("Timesteps to finish:", time_step)
            success = True

        actions.append(action)
        observations.append(obs)
        dones.append(done)
        rewards.append(reward)
        
    print("Episode time used: {:.2f}s\n".format(time.time() - episode_init_time), time_step)

    #if success:
    #actions.append(episode_acs)
    #observations.append(episode_obs)
    #dones.append(episode_dones)
    #rewards.append(episode_rewards)
        


if __name__ == "__main__":
    main()
