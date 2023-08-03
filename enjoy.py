import numpy as np
import pickle
import torch

from docopt import docopt
from model import ActorCriticModel
from utils import create_env
import torchvision
import  gym
gym.envs.register(
     id='VisualGroundingEnv-v0',
     entry_point='environments:VisualGroundingEnv',
)

from dataset import RefCOCOg

validation = RefCOCOg('..', 'val')

def init_transformer_memory(trxl_conf, max_episode_steps, device):
    """Returns initial tensors for the episodic memory of the transformer.

    Arguments:
        trxl_conf {dict} -- Transformer configuration dictionary
        max_episode_steps {int} -- Maximum number of steps per episode
        device {torch.device} -- Target device for the tensors

    Returns:
        memory {torch.Tensor}, memory_mask {torch.Tensor}, memory_indices {torch.Tensor} -- Initial episodic memory, episodic memory mask, and sliding memory window indices
    """
    # Episodic memory mask used in attention
    memory_mask = torch.tril(torch.ones((trxl_conf["memory_length"], trxl_conf["memory_length"])), diagonal=-1)
    # Episdic memory tensor
    memory = torch.zeros((1, max_episode_steps, trxl_conf["num_blocks"], trxl_conf["embed_dim"])).to(device)
    # Setup sliding memory window indices
    repetitions = torch.repeat_interleave(torch.arange(0, trxl_conf["memory_length"]).unsqueeze(0), trxl_conf["memory_length"] - 1, dim = 0).long()
    memory_indices = torch.stack([torch.arange(i, i + trxl_conf["memory_length"]) for i in range(max_episode_steps - trxl_conf["memory_length"] + 1)]).long()
    memory_indices = torch.cat((repetitions, memory_indices))
    return memory, memory_mask, memory_indices

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        enjoy.py [options]
        enjoy.py --help
    
    Options:
        --model=<path>              Specifies the path to the trained model [default: ./models/run.nn].
    """
    options = docopt(_USAGE)
    model_path = options["--model"]

    # Set inference device and default tensor type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

    # Load model and config
    dict = torch.load(model_path, map_location=device)
    state_dict = dict["model"]
    config = dict["params"]
    # Instantiate environment
    env = create_env(config["environment"], 1, render=True)

    # Initialize model and load its parameters
    model = ActorCriticModel(config, env.observation_space, (env.action_space.n,), env.max_episode_steps)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    iou = 0
    iou_count = 0
    for ep in range(len(validation)):
        # Run and render episode
        done = False
        episode_rewards = []
        memory, memory_mask, memory_indices = init_transformer_memory(config["transformer"], env.max_episode_steps, device)
        memory_length = config["transformer"]["memory_length"]
        t = 0
        obs, info = env.reset()
        i = 0
        # print("obs is ",type(obs),"with shape ",obs.shape)
        while i<env.max_episode_steps and not(info["trigger_pressed"]):
            # Prepare observation and memory
            # tmp = np.expand_dims(obs, 0)
            # print("tmp is ",tmp.shape)
            obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            in_memory = memory[0, memory_indices[t].unsqueeze(0)]
            t_ = max(0, min(t, memory_length - 1))
            mask = memory_mask[t_].unsqueeze(0)
            indices = memory_indices[t].unsqueeze(0)
            # # Render environment
            # env.render()
            # Forward model
            # print(obs.shape, in_memory.shape, mask.shape, indices.shape)
            policy, value, new_memory = model(obs, in_memory, mask, indices)
            memory[:, t] = new_memory
            # Sample action
            action = []
            for action_branch in policy:
                action.append(action_branch.sample().item())
            # print("taken action #",action)
            # Step environemnt
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            i += 1
        iou += info["iou"]
        iou_count += 1
        
        # after done, render last state
        # env.render()
        print("Episode: {}".format(ep+1), end="\t\t")
        print("Length: {}".format(info["length"]), end="\t\t")
        print("Reward: {:.3f}".format(info["reward"]), end="\t\t")
        print("IoU: {:.3f}".format(info["iou"]), end="\t\t")
        print("Mean IoU: {:.3f}".format(iou / iou_count), end="\t\t")
        print("Predicted bbox: {}".format(info["pred_bbox"]), end="\t\t")
        print("Ground truth bbox: {}".format(info["target_bbox"]))

    env.close()

if __name__ == "__main__":
    main()
