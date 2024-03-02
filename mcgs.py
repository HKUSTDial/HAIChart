#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : Administrator
from datetime import datetime
import os
import random
import math
import time

import numpy as np
import pandas as pd
from state import State
from copy import deepcopy

np.random.seed(15)
random.seed(15)

from user_model.test_env_model import get_config_score
from user_model.vega_zero_trans import vega_zero_trans_config


def formatting_query(query):
    # M [T] D v_i EC x [X] y AGG [AggFunction] [Y] C (Z)
    # TF [TransForm] F [F] G [G] B x by [B] S [S] K [K]
    querys = query.split(' ')
    group_by_who = querys[querys.index("group") + 1]
    groups = group_by_who.split(",")
    if len(groups) > 1:
        querys[querys.index("color") + 1] = groups[0]
    if len(groups) == 1:  
        querys[querys.index("color") + 1] = "none"
    querys[querys.index("mark")] = "M"
    querys[querys.index("aggregate")] = "A"
    querys[querys.index("transform")] = "TF"
    querys[querys.index("encoding")] = "E"
    querys[querys.index("filter")] = "F"
    querys[querys.index("group")] = "G"
    querys[querys.index("color")] = "C"
    querys[querys.index("bin")] = "B"
    querys[querys.index("sort")] = "S"
    querys[querys.index("topk")] = "K"
    if "true" in querys:
        querys[querys.index("true")] = "T"
    if "false" in querys:
        querys[querys.index("false")] = "F"
    if "none" in querys:
        querys[querys.index("none")] = "N"

    return ' '.join(querys)


class Node:

    global_node_storage = {}  # Used to store all nodes for easy lookup

    def __init__(self, parent=None, identifier='', action=''):
        self.identifier = identifier
        self.edges = {}
        self.action = action  # The current node's value, used during traversal, root node is empty
        self.parent = parent
        self.visited = False  # Not visited when initialized
        self.Q = 0  # The ultimate reward value of the node
        self.N = 0  # The number of times the node has been visited


    def get_untried_actions(self, current_state, dp,constraints):
        available_actions = current_state.get_available_actions(dp,constraints)

        # Extract action value
        tried_action_values = {identifier.split('_')[1] for identifier in self.edges.keys()}

        untried_actions = [action for action in available_actions if action not in tried_action_values]
        return untried_actions

    @staticmethod
    def get_random_action(available_actions):
        action_number = len(available_actions)
        action_index = np.random.choice(range(action_number))
        return available_actions[action_index]

    def select(self, current_state, dp, constraints, c_param=10):
        """
        Selects the optimal action based on the current child nodes and returns the child node
        :param c_param: Exploration parameter used for the proportion of exploration
        :return: Optimal action, child node under the optimal action
        """
        # Here we need to decide, if it is a leaf node and has been visited, then another one needs to be chosen
        weights = []
        # Here, based on the current state, random noise can be added
        alpha = 0.8
        max_brackets = 7
        num_brackets = current_state.query.count('[')  # Count the number of selected clauses
        # Use exponential decay to calculate the probability of random selection
        exploration_rate = math.exp(-alpha * (1 - num_brackets / max_brackets))

        # Filter out the nodes that can be selected based on the current state
        legal_actions = current_state.get_available_actions(dp, constraints)
        # Now, we filter self.edges to only retain those edges that correspond to legal actions
        legal_edges = {k: v for k, v in self.edges.items() if k.split("_")[1] in legal_actions}

        # Decide whether to explore based on the exploration rate randomly
        if random.random() < exploration_rate:
            identifier = random.choice(list(legal_edges.keys()))
            action_type, action = identifier.split("_")

        else:
            # Calculate the weight for each edge
            for target_node, edge_info in legal_edges.items():
                visits = edge_info['visits']
                value = edge_info['value']
                parent_visits = self.N

                if visits != 0:
                    w = value / visits + c_param * np.sqrt(2 * np.log(parent_visits) / visits)
                else:
                    w = float("inf")  # N==0 means it hasn't been explored, so it should be prioritized
                weights.append(w)

            # Choose the action with the highest weight
            identifier = pd.Series(data=weights, index=legal_edges.keys()).idxmax()
            action_type, action = identifier.split("_")

        next_node = Node.global_node_storage[identifier]

        current_state.get_next_state(action)
        current_state.edge_path.append((next_node, action))
        return action, next_node


    def expand(self, current_state, dp, constraints):
        """
        Expands a child node and returns the newly expanded child node.
        :return: The newly expanded child node.
        """
        untried_actions = self.get_untried_actions(current_state, dp, constraints)
        # Randomly choose one
        random_index = random.randrange(len(untried_actions))
        # Choose from untried nodes
        action = untried_actions.pop(random_index)
        # Add action to the query statement
        q = current_state.query

        # Get the current action's type and value
        action_type = ''  
        action_value = action  

        if q.find('[T]') != -1:
            action_type = '[T]'
            if action == 'line':
                q = q.replace('[S]', 'none')
                q = q.replace('[T]', action)
            else:
                q = q.replace('[T]', action)
        elif q.find('[X]') != -1:
            action_type = '[X]'
            q = q.replace('[X]', action)

        elif q.find('[TransForm]') != -1:
            action_type = '[TransForm]'
            if action == 'false':  # If false, subsequent all are none
                q = q.replace('[TransForm]', action)
                q = q.replace('[AggFunction]', 'none')
                q = q.replace('[F]', 'none')
                q = q.replace('[G]', 'none')
                q = q.replace('[B]', 'none')
                q = q.replace('[S]', 'none')
                q = q.replace('[K]', 'none')
            else:
                q = q.replace('[TransForm]', action)

        elif q.find('[AggFunction]') != -1:
            action_type = '[AggFunction]'
            q = q.replace('[AggFunction]', action)
        elif q.find('[Y]') != -1:
            action_type = '[Y]'
            q = q.replace('[Y]', action)
        elif q.find('[F]') != -1:
            action_type = '[F]'
            q = q.replace('[F]', action)
        elif q.find('[G]') != -1:
            action_type = '[G]'
            if len(action.split(",")) > 1:
                q = q.replace('[B]', 'none')
            q = q.replace('[G]', action)
        elif q.find('[B]') != -1:
            action_type = '[B]'
            if action == 'ZERO':  # If bin ZERO is chosen, no need for sort and topk
                q = q.replace('[S]', 'none')
                q = q.replace('[K]', 'none')
                q = q.replace('[B]', action)
            else:
                q = q.replace('[B]', action)
        elif q.find('[S]') != -1:
            action_type = '[S]'
            q = q.replace('[S]', action)
        elif q.find('[K]') != -1:
            action_type = '[K]'
            q = q.replace('[K]', action)

        # Generate a unique identifier, action type + action value, for example [X]_delay
        identifier = f"{action_type}_{action_value}"

        # Check if this node already exists in global_node_storage
        if identifier in Node.global_node_storage:
            child_node = Node.global_node_storage[identifier]
        else:
            # If this node does not exist, create a new node
            child_node = Node(self, identifier, action)
            Node.global_node_storage[identifier] = child_node

        current_state.query = q
        # Record it for updating
        current_state.edge_path.append((child_node, action))
        # Update edge information
        if identifier not in self.edges:
            self.edges[identifier] = {'visits': 0, 'value': 0, 'UCB': 0}

        return child_node



    def update(self, new_query, current_state, dp, history_score, good_view, current_view):
        """
        After simulation, update the node's value and visit count
        :param new_query: The returned query statement
        :return:
        """

        # Get the edge path of the current state
        edge_path = current_state.edge_path

        if new_query.find('*!STOP!*') != -1:
            view_score = 0  # This path is blocked!
        else:
            view_score = get_view_score(new_query, dp, history_score, good_view, current_view)

        # Update along the edge_path in reverse
        for idx, (node, edge_identifier) in enumerate(edge_path):

            # The first one is a special case, as it's the root, the parent will not be wrong
            if idx == 0:
                parent_node = node.parent
            elif idx < len(edge_path):  # Ensure the index range is not exceeded
                parent_node = edge_path[idx - 1][0]  # Get the parent node of the current node
                # Update the visit count of the parent node
            parent_node.N += 1
            # Update the visit count and value of the edge from the parent node to the current node
            parent_node.edges[node.identifier]['visits'] += 1
            parent_node.edges[node.identifier]['value'] += view_score


    def rollout(self, dp, current_state, constraints):
        """
        Perform a Monte Carlo simulation from the current node and return the simulation result
        :return: Simulation result
        """
        while True:
            is_over, new_q = current_state.get_state_result()
            if is_over:
                break
            available_actions = current_state.get_available_actions(dp, constraints)
            action = Node.get_random_action(available_actions)  # Randomly select from the candidates to achieve the purpose of simulation
            current_state = current_state.get_next_state(action)

        return new_q


    def is_full_expand(self, current_state, dp,constraints):
        available_actions = set(current_state.get_available_actions(dp,constraints))

        tried_action_values = {identifier.split('_')[1] for identifier in self.edges.keys()}

        return available_actions.issubset(tried_action_values)

    def is_root_node(self):
        return self.parent


class MCGS:
    def __init__(self):
        self.root = None
        self.current_node = None

    def __str__(self):
        return "monte carlo graph search"

    def simulation(self, count, dp, history_score, good_view, constraints, current_view):

        for i in range(count):
            current_state = State()
            leaf_node = self.simulation_policy(dp, current_state, constraints)  # Select an unexplored node for simulation prediction
            new_query = leaf_node.rollout(dp, current_state, constraints)  # Play the game at a node using a random strategy
            leaf_node.update(new_query, current_state, dp, history_score, good_view, current_view)  # Update the result and backpropagate

    def simulation_policy(self, dp, current_state, constraints):
        current_node = self.current_node
        while True:
            is_over, _ = current_state.get_state_result()
            if is_over:
                break
            if current_node.is_full_expand(current_state, dp, constraints):  # Attack unexplored items
                _, current_node = current_node.select(current_state, dp, constraints)  # Root node selection is complete, continue from child node downwards
            else: 
                expand_node = current_node.expand(current_state, dp, constraints)  # If the current node has unexplored options, select and return it, waiting for simulation
                return expand_node
        leaf_node = current_node
        return leaf_node


    def start_exploring(self, history_score, good_view, dp, constraints, current_view):
        if not self.root:  # Establish the root node
            self.root = Node(None, "root", "root")
        self.current_node = self.root # Always start from the root node
        self.simulation(100, dp, history_score, good_view, constraints, current_view) 
        d = dict((k, v) for k, v in history_score.items() if v != -15)

