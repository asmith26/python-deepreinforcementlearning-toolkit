#!/usr/bin/env bash

# Based on https://github.com/tambetm/simple_dqn/blob/master/profile_*

# never explore, always predict
python -m cProfile -s cumtime $* src/main.py --random_steps=10000 --train_steps=0 --test_steps=0 --epochs=1 roms/pong.bin

# never explore, always predict
python -m cProfile -s cumtime $* src/main.py --exploration_rate_test=0 --random_steps=0 --train_steps=0 --test_steps=10000 --epochs=1 roms/pong.bin

# predict all moves by random, to separate prediction and training time
python -m cProfile -s cumtime $* src/main.py --exploration_rate_end=1 --random_steps=5 --train_steps=5000 --test_steps=0 --epochs=1 --train_frequency 1 --target_steps 0 roms/pong.bin
