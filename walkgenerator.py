# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import networkx as nx
import numpy as np
import random

class Metapathgenerator:
    def __init__(self,file):
        self.file = file
        self.id_pin = dict()
        self.id_user = dict()
        self.id_board = dict()
        self.pin_user = dict()
        self.user_pin = dict()
        self.pin_board = dict()
        self.board_pin = dict()
        self.user_board = dict()
        self.board_user = dict()
    
    def load_data(self):
        self.graph = nx.read_edgelist(self.file, delimiter=None, create_using=None, nodetype=None, data=True, edgetype=None, encoding='utf-8')
     
    def read_data(self):
        id_pin = {}
        id_user = {}
        id_board = {}
        index = 1
        for x in self.graph.nodes:
                if x[0] == 'p':
                    id_pin[index] = x
                elif x[0] == 'u':
                    id_user[index] = x
                elif x[0] == 'b':
                    id_board[index] = x
                index += 1
        self.id_pin = id_pin
        self.id_user = id_user
        self.id_board = id_board
    
    def construct_dict(self):
        def find_key(input_dict, value):
            return next((k for k, v in input_dict.items() if v == value), None)
        
        pin_board = []
        pin_user = []   
        for x in self.graph.edges:
            if x[0][0] == 'p' and x[1][0] == 'b':
                pin_board.append((find_key(self.id_pin,x[0]),find_key(self.id_board,x[1])))
            elif x[0][0] == 'b' and x[1][0] == 'p':
                pin_board.append((find_key(self.id_pin,x[1]),find_key(self.id_board,x[0])))                            
            elif x[0][0] == 'p' and x[1][0] == 'u':
                pin_user.append((find_key(self.id_pin,x[0]),find_key(self.id_user,x[1])))  
            elif x[0][0] == 'u' and x[1][0] == 'p':
                pin_user.append((find_key(self.id_pin,x[1]),find_key(self.id_user,x[0]))) 
        
        pin_board_dict = {}
        board_pin_dict = {}
        for x in pin_board:
            p, b = x[0],x[1]
            if p not in pin_board_dict:
                pin_board_dict[p] = []
            pin_board_dict[p].append(b)
            if b not in board_pin_dict:
                board_pin_dict[b] = []
            board_pin_dict[b].append(p)
        self.pin_board = pin_board_dict
        self.board_pin = board_pin_dict

        pin_user_dict = {}
        user_pin_dict = {}
        for x in pin_user:
            p, u = x[0],x[1]
            if p not in pin_user_dict:
                pin_user_dict[p] = []
            pin_user_dict[p].append(u)
            if u not in user_pin_dict:
                user_pin_dict[u] = []
            user_pin_dict[u].append(p)
        self.pin_user = pin_user_dict
        self.user_pin = user_pin_dict
                    
    def generate_path(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')
        for user in self.user_pin:
            user0 = user
            for j in range(0, numwalks ): #wnum walks
                outline = self.id_user[user0]
                for i in range(0, walklength):
                    if user not in self.user_pin:
                        num = len(self.user_pin)
                        userid = random.randrange(num)
                        user = list(self.user_pin)[userid]
                    pins = self.user_pin[user]
                    num = len(pins)
                    pinid = random.randrange(num)
                    pin = pins[pinid]
                    outline += " " + self.id_pin[pin]
                    
                    if pin not in self.pin_board:
                        num = len(self.pin_board)
                        pinid = random.randrange(num)
                        pin = list(self.pin_board)[pinid]                   
                    boards = self.pin_board[pin]
                    num = len(boards)
                    boardid = random.randrange(num)
                    board = boards[boardid]
                    outline += " " + self.id_board[board]
 
                    if board not in self.board_pin:
                        num = len(self.board_pin)
                        boardid = random.randrange(num)
                        board = list(self.board_pin)[boardid] 
                    pins = self.board_pin[board]
                    num = len(pins)
                    pinid = random.randrange(num)
                    pin = pins[pinid]
                    outline += " " + self.id_pin[pin]
                    
                    if pin not in self.pin_user:
                        num = len(self.pin_user)
                        pinid = random.randrange(num)
                        pin = list(self.pin_user)[pinid] 
                    users = self.pin_user[pin]
                    num = len(users)
                    userid = random.randrange(num)
                    user = users[userid]
                    outline += " " + self.id_user[user]
                
                outfile.write(outline + "\n")
        outfile.close()

