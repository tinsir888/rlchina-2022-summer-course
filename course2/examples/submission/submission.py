# -*- coding:utf-8  -*-
# Time  : 2022/8/10 下午4:14
# Author: Yahui Cui

"""
# =================================== Important =========================================
Notes:
1. this agents is random agents , which can fit any env in Jidi platform.
2. if you want to load .pth file, please follow the instruction here:
https://github.com/jidiai/ai_lib/blob/master/examples/demo
"""
import numpy as np
import random as rd

epoch = 10000 # simulation time
infty = 100000000
eps = 0.1
direct = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
forward_step = 5

vis = {}
rtn = {}

def place(board,x,y,color,w,h):
    """
    place chess, calc new state
    params:
        board: chess board
        x: pos x
        y: pos y
        color: chess color
        w: the width of board
        h: the height of board
    """
    valid = False
    if x < 0:
        return valid
    board[x][y] = color
    for d in range(8):
        i = x + direct[d][0]
        j = y + direct[d][1]
        while 0 <= i and i < w and 0 <= j and j < h and board[i][j] == -color:
            i += direct[d][0]
            j += direct[d][1]
        if 0 <= i and i < w and 0 <= j and j < h and board[i][j] == color:
            while True:
                i -= direct[d][0]
                j -= direct[d][1]
                if i == x and j == y:
                    break
                valid = True
                board[i][j] = color
    return valid

def eval(board, color, w, h):
    """
    evaluate current state
    params:
        board: current chess board
        color: current player's standing
        w: width
        h: height
    """
    score = 0
    for i in range(w):
        for j in range(h):
            score += board[i][j] * color
    if score > 0:
        return 1
    elif score < 0:
        return -1
    else:
        return 0


def get_move(board, color, best_choice, w, h):
    """
    choose the next move
    params:
        board: current chess board
        color: current player's standing
        best_choice: whether choose the best move or randomly choose
        w: width
        h: height
    """
    moves = []
    for i in range(w):
        for j in range(h):
            if board[i][j] == 0:
                new_board = board.copy()
                if place(new_board, i, j, color, w, h):
                    moves.append((i,j))
    best = -infty
    x = y = -1
    if len(moves) == 0:
        return x, y
    for (i, j) in moves:
        average = infty
        if (color, i, j) in vis:
            average = rtn[color, i, j] / vis[color, i, j]
        if average > best:
            best = average
            x, y = i, j
    if best_choice or rd.random() > eps:
        return x, y
    return rd.choice(moves)

def simulate(board, color, w, h):
    """
    simulate monte-carlo process
    """
    x, y = get_move(board, color, False, w, h)
    no_move = True if x < 0 else False
    if no_move:
        color = -color
        x, y = get_move(board, color, False, w, h)
        if x < 0:
            return eval(board, -color, w, h)
    new_board = board.copy()
    place(new_board, x, y, color, w, h)
    res = -simulate(new_board, -color, w, h)
    global vis, rtn
    if (color, x, y) not in vis:
        vis[color, x, y] = 1
        rtn[color, x, y] = res
    else:
        vis[color, x, y] += 1
        rtn[color, x, y] += res
    if no_move:
        return -res
    else:
        return res

def mcts(board, color, w, h):
    for idx in range(epoch):
        simulate(board, color, w, h)
    return get_move(board, color, True, w, h)

def action_format(x, y, w, h):
    act = [[0] * w, [0] * h]
    act[0][x] = 1
    act[1][y] = 1
    return act

def my_controller(observation, action_space, is_act_continuous=True):
    my_color = 1 if observation["chess_player_idx"] == 1 else -1
    h = observation["board_height"]
    w = observation["board_width"]
    board = [[0 for _ in range(w)] for __ in range(h)]
    for i in range(w):
        for j in range(h):
            board[i][j] = 0
    for pos in observation[1]:
        board[pos[0]][pos[1]] = 1
    for pos in observation[2]:
        board[pos[0]][pos[1]] = -1
    x, y = mcts(board, my_color, w, h)
    return action_format(x, y, w, h)
