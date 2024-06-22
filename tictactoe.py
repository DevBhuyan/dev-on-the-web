#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 01:27:02 2023

@author: dev
"""

import numpy as np
import pickle
import os
from copy import deepcopy


class Board():
    def __init__(self):
        self.base_mat = np.zeros((3, 3), dtype='int')
        self.player = True      # 1st turn is of the player
        self.result = 0
        self.turn_ctr = 0

    def reset(self):
        self.base_mat = np.zeros((3, 3), dtype='int')
        self.player = True
        self.result = 0
        self.turn_ctr = 0

    def step(self, action=None):
        if not isinstance(action, int):
            action = np.random.choice(self.valid_moves())
        x, y = divmod(action, 3)
        self.base_mat[x, y] = 1 if self.player else -1
        self.turn_ctr += 1
        # self.status()
        self.x = x
        self.y = y
        self.compute_result()
        self.player = not self.player

    def compute_result(self):
        '''
        Bot wins get rewards
        '''
        x = self.x
        y = self.y
        base_mat = self.base_mat
        if (tuple(base_mat[x]) == (1, 1, 1) or
            tuple(base_mat.T[y]) == (1, 1, 1) or
            (base_mat[0, 0], base_mat[1, 1], base_mat[2, 2]) == (1, 1, 1) or
                (base_mat[0, 2], base_mat[1, 1], base_mat[2, 0]) == (1, 1, 1)):
            print('Player won!')
            self.result = -5/self.turn_ctr
        elif (tuple(base_mat[x]) == (-1, -1, -1) or
              tuple(base_mat.T[y]) == (-1, -1, -1) or
              (base_mat[0, 0], base_mat[1, 1], base_mat[2, 2]) == (-1, -1, -1) or
              (base_mat[0, 2], base_mat[1, 1], base_mat[2, 0]) == (-1, -1, -1)):
            print('Bot won!')
            self.result = 1/self.turn_ctr
        elif self.turn_ctr == 9:
            print('Game Draw!')
            self.result = -0.05

    def status(self):
        print('Turn count: ', self.turn_ctr, end=" | ")
        print("Player's turn") if self.player else print("Bot's turn")
        self.display_board()
        return

    def display_board(self):
        for row in self.base_mat:
            for mark in row:
                if mark == 1:
                    print(' X ', end="")
                elif mark == -1:
                    print(' O ', end="")
                else:
                    print('   ', end="")
            print('\n-----------')

    def valid_moves(self):
        valid_moves = []
        for i in range(3):
            for j in range(3):
                if self.base_mat[i, j] == 0:
                    valid_moves.append(i * 3 + j)
        return valid_moves


class Agent():
    '''
    First iterate through the total number of paths and compute reward.
    Distribute reward to each step of path. Now each step in a path must have a distributed reward score
    The remaining step is to map the step-wise changes to a reward system
    '''

    def __init__(self):
        self.step_scores = []

    def convert_state_path_on_the_go(self, path):
        steps, reward = path[0], path[1]
        for i in range(len(steps)//2):
            move = [tuple(steps[2*i].flatten()),
                    tuple(steps[2*i+1].flatten()), reward, 1]
            if move[:-2] not in [m[:-2] for m in self.step_scores]:
                # Structure of move is [current, next, reward]
                self.step_scores.append(move)
            else:
                # LATEST UPDATE: AVERAGING BY 2 REDUCES THE WEIGHTS OF OLDER MOVES, KEEP A PROVISION TO AVERAGE REWARDS WITH WEIGHTS. E.G. THE THIRD ENTRY FOR SAME REWARD WILL BECOME (2*(OLD)+1*NEW)/3
                idx = [m[:-2] for m in self.step_scores].index(move[:-2])
                self.step_scores.append([move[0], move[1], (move[2]*1+self.step_scores[idx][2] *
                                        self.step_scores[idx][3])/(1+self.step_scores[idx][3]), 1+self.step_scores[idx][3]])
                # POP OLDER ENTRY
                self.step_scores.pop(idx)

    def recommend_best_action(self, base_mat, board):
        # TODO: ALSO KEEP A PROVISION TO CHECK IF PLAYER IS COOKING A MOVE, PREVENT PLAYER FROM GETTING 3(X) IN A LINE
        # TODO: CHECK FOR CONTINUOUS (X)S OR (O)S, IF O, FOIL THE NEXT EMPTY BOX, IF X, FILL NEXT EMPTY BOX IF AVAILABLE
        # DONE: ALSO CHECK FOR SAME MARKS IN OPPOSITE CORNERS
        for i in range(3):
            for j in range(3):
                # checking opposite corners
                if base_mat[i, j]:
                    if base_mat[i, j] == base_mat[2-i, 2-j]:
                        if base_mat[1, 1] == 0:
                            # move can be made, recommend best action as 4
                            print('return 1 invoked')
                            return 4
                    # check adjacent corners
                    # row-wise
                    if base_mat[i, j] == base_mat[i, 2-j]:
                        if base_mat[i, 1] == 0:
                            print('return 2 invoked')
                            return 3*i+1
                    # column-wise
                    if base_mat[i, j] == base_mat[2-i, j]:
                        if base_mat[1, j] == 0:
                            print('return 3 invoked')
                            return 3+j

        max_reward = 0
        best_config = None
        flag = 1
        for move in self.step_scores:
            if move[0] == tuple(base_mat.flatten()):
                flag = 0
                # IF REWARD IS POSITIVE. I.E. BOT REMEMBERS A WINNING MOVE
                if move[2] > max_reward:
                    max_reward = move[2]
                    best_config = move[1]
        if best_config == None:
            # BOT DOESN'T KNOW ANY WINNING MOVE
            # FIND THE MOVES WHERE BOT WON'T WIN, DO SOMETHING OTHER THAN THOSE
            dont_take_this_no = []
            for move in self.step_scores:
                if move[0] == tuple(base_mat.flatten()):
                    flag = 0
                    if move[2] < 0:
                        # DON'T DO THIS MOVE
                        dont_take_this_no.append(
                            np.argmax(np.array(list(move[1])) - np.array(base_mat.flatten())))
            print('return 4 invoked')
            best_action = np.random.choice(
                [i for i in board.valid_moves() if i not in dont_take_this_no])
        else:
            print('return 5 invoked')
            best_action = np.argmax(
                np.array(list(best_config)) - np.array(base_mat.flatten()))
        if flag:
            # Unseen move
            return None
        else:
            return best_action

# TODO: WE NEED THE BOT TO SPATIALLY UNDERSTAND THE GAME, ESP. THE RULE OF THREE IN A LINE, THIS WILL HELP IT TO FOIL THE PLAYERS MOVES AS WELL AS GUESS MOST LIKELY MOVES
# DONE: IF BOT HAS TWO (O)S IN ONE LINE, AND THE THIRD BOX IS EMPTY ALREADY, PUT THE (O) IN THE EMPTY BOX IMMEDIATELY TO SCORE A WIN, I HOPE THE BOT WILL LEARN THIS OVER TIME, BUT THE BOT LOOKS TOO DUMB IF IT CAN'T DO IT ALREADY


def play_a_turn(board, agent, state_path, player=True):
    if player:
        board.display_board()
        print("Your turn!")
        valid_moves = board.valid_moves()
        print("Valid moves:", valid_moves)
        your_action = int(input("Enter your move (0-8): "))
        board.step(your_action)

    else:
        # Bot's turn
        board.display_board()
        print("Bot's turn!")

        if agent.recommend_best_action(deepcopy(board.base_mat), board) == None:
            bot_action = np.random.choice(board.valid_moves())
        else:
            bot_action = agent.recommend_best_action(
                deepcopy(board.base_mat), board)
        board.step(bot_action)

    state_path.append(deepcopy(board.base_mat))

    return board, state_path


def learn_from_player(agent, board):
    board.reset()
    print(f'\n\nNew Game')
    state_path = []
    while board.result == 0:
        board, state_path = play_a_turn(board, agent, state_path, player=True)

        if board.result != 0:
            board.display_board()
            break

        # Bot's turn
        board, state_path = play_a_turn(board, agent, state_path, player=False)

        if board.result != 0:
            board.display_board()
            break

    board.compute_result()
    print("Game Over")
    agent.convert_state_path_on_the_go([state_path, board.result])
    return agent


def load_dump_agent(agent=None):
    if agent == None:
        # LOAD
        agent = pickle.load(open('./agent.pkl', 'rb')
                            ) if os.path.exists('./agent.pkl') else Agent()
        return agent
    else:
        # DUMP
        pickle.dump(agent, open('./agent.pkl', 'wb'))
        return None


def start():
    print('You: (X)')
    print('Bot: (O)')
    board = Board()
    agent = load_dump_agent()

    for _ in range(10):
        agent = learn_from_player(agent, board)

    load_dump_agent(agent)


if __name__ == "__main__":
    start()
