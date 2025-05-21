import sys
sys.path.insert(0, '/home/shapla/LLaVA')
from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
