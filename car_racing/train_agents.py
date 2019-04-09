"""
Game is solved when agent consistently gets 900+ points. Track is random every episode.
"""

import numpy as np
import gym
import time, tqdm
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import car_racing

import cma
import multiprocessing as mp

from train_vae import load_vae

_RENDER = True
_EMBEDDING_SIZE = 32
_NUM_PREDICTIONS = 2
_NUM_ACTIONS = 3
_NUM_PARAMS = _NUM_PREDICTIONS * _EMBEDDING_SIZE + _NUM_PREDICTIONS
car_racing.VIDEO_H = 1000
car_racing.VIDEO_W = 1200
car_racing.FPS = 60


def normalize_observation(observation):
    return observation.astype('float32') / 255.


def get_weights_bias(params):
    weights = params[:_NUM_PARAMS - _NUM_PREDICTIONS]
    bias = params[-_NUM_PREDICTIONS:]
    weights = np.reshape(weights, [_EMBEDDING_SIZE, _NUM_PREDICTIONS])
    return weights, bias


def decide_action(sess, network, observation, params):
    observation = normalize_observation(observation)
    embedding = sess.run(network.z, feed_dict={network.image: observation[None, :, :, :]})
    weights, bias = get_weights_bias(params)
    
    action = np.zeros(_NUM_ACTIONS)
    prediction = np.matmul(np.squeeze(embedding), weights) + bias
    prediction = np.tanh(prediction)
    
    action[0] = prediction[0]
    if prediction[1] < 0:
        action[1] = np.abs(prediction[1])
        action[2] = 0
    else:
        action[2] = prediction[1]
        action[1] = 0
    
    return action


env = car_racing.CarRacing()


def play(params, verbose=False):
    sess, network = load_vae()
    _NUM_TRIALS = 12
    agent_reward = 0
    for trial in range(_NUM_TRIALS):
        observation = env.reset()
        # Little hack to make the Car start at random positions in the race-track
        np.random.seed(int(str(time.time() * 1000000)[10:13]))
        position = np.random.randint(len(env.track))
        env.car = Car(env.world, *env.track[position][1:4])
        
        total_reward = 0.0
        steps = 0
        while True:
            if _RENDER:
                env.render("rgb_array")
            action = decide_action(sess, network, observation, params)
            observation, r, done, info = env.step(action)
            total_reward += r
            # NB: done is not True after 1000 steps when using the hack above for
            # 	  random init of position
            if verbose and (steps % 200 == 0 or steps == 999):
                print("\naction " + str(["{:+0.2f}".format(x) for x in action]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            
            steps += 1
            if steps == 999:
                break
        
        # If reward is out of scale, clip it
        total_reward = np.maximum(-100, total_reward)
        agent_reward += total_reward
    
    return - (agent_reward / _NUM_TRIALS)


def train():
    es = cma.CMAEvolutionStrategy(_NUM_PARAMS * [0], 0.1, {'popsize': 16})
    rewards_through_gens = []
    generation = 1
    try:
        while not es.stop():
            solutions = es.ask()
            with mp.Pool(mp.cpu_count()) as p:
                rewards = list(tqdm.tqdm(p.imap(play, list(solutions)), total=len(solutions)))
            
            es.tell(solutions, rewards)
            
            rewards = np.array(rewards) * (-1.)
            print("\n**************")
            print("Generation: {}".format(generation))
            print("Min reward: {:.3f}\nMax reward: {:.3f}".format(np.min(rewards), np.max(rewards)))
            print("Avg reward: {:.3f}".format(np.mean(rewards)))
            print("**************\n")
            
            generation += 1
            rewards_through_gens.append(rewards)
            np.save('rewards', rewards_through_gens)
            np.save('best_params', es.best.get()[0])
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")
    except Exception as e:
        print("Exception: {}".format(e))
    return es


if __name__ == '__main__':
    es = train()
    
    input("Press enter to play... ")
   
    score = play(es.best.get()[0], verbose=True)
    print("Final Score: {}".format(-score))
