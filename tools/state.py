#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : Administrator
import numpy as np

from tools import Type

class State:

    def __init__(self,
                 query="mark [T] encoding x [X] y aggregate [AggFunction] [Y] color (Z) transform [TransForm] filter none group [G] bin x by [B] sort none topk none"):
        self.query = query
        self.edge_path = []

    def get_available_data_type(self, axis, char_type):
        x_ava_type = []
        y_ava_type = []
        if char_type == 'point':
            x_ava_type.append(Type.numerical)
            x_ava_type.append(Type.temporal)
            y_ava_type.append(Type.numerical)
        elif char_type == 'bar':
            x_ava_type.append(Type.categorical)
            x_ava_type.append(Type.temporal)
            x_ava_type.append(Type.numerical)
            y_ava_type.append(Type.numerical)
        elif char_type == 'line':
            x_ava_type.append(Type.temporal)
            x_ava_type.append(Type.categorical)
            x_ava_type.append(Type.numerical)
            y_ava_type.append(Type.numerical)
        elif char_type == 'arc':
            x_ava_type.append(Type.categorical)
            y_ava_type.append(Type.numerical)
        elif char_type == 'heatmap':
            x_ava_type.append(Type.numerical)
            x_ava_type.append(Type.temporal)
            y_ava_type.append(Type.numerical)
        elif char_type == 'box':
            x_ava_type.append(Type.categorical)
            x_ava_type.append(Type.temporal)
            y_ava_type.append(Type.numerical)
        return x_ava_type if axis == 'x' else y_ava_type



    def get_available_actions(self,dp,constraints):

        tokens = self.query.split(' ')
        if '[T]' in tokens:
            if constraints.get('[T]') is not None and constraints.get('[T]') != []:
                return constraints['[T]']
            mark = ['bar', 'line', 'point', 'arc']
            return mark
        if '[X]' in tokens:
            if constraints.get('[X]') is not None and constraints.get('[X]') != []:
                return constraints['[X]']

            x_ava_type_list = []
            x_ava_type = self.get_available_data_type('x',tokens[tokens.index("mark") + 1])
            x_all = dp.column_names.copy()
            for index,value in enumerate(x_all):
                if Type.getType(dp.column_types[index]) in x_ava_type:
                    x_ava_type_list.append(value)

            if x_ava_type_list == []:
                return ['*!STOP!*']
            else:
                return x_ava_type_list

        if '[TransForm]' in tokens:
            if constraints.get('[TransForm]') is not None and constraints.get('[TransForm]') != []:
                return constraints['[TransForm]']

            if any(constraints.get(key) for key in ['[AggFunction]', '[G]', '[B]']):
                return ['true']

            chart_type = tokens[tokens.index("mark") + 1]
            if chart_type == 'box' or chart_type == 'heatmap':
                TransForm = ['true'] 
            elif chart_type == 'point':
                TransForm = ['false', 'true']
            else:
                TransForm = ['true']
            
            return TransForm
        if '[AggFunction]' in tokens:
            if constraints.get('[AggFunction]') is not None and constraints.get('[AggFunction]') != []:
                return constraints['[AggFunction]']

            chart_type = tokens[tokens.index("mark") + 1]
            if chart_type == 'box' or chart_type == 'heatmap':
                aggregate = ['none']
            else:
                aggregate = ['count', 'sum', 'average','none']
            return aggregate
        if '[Y]' in tokens:
            if constraints.get('[Y]') is not None and constraints.get('[Y]') != []:
                return constraints['[Y]']
            x_axis = tokens[tokens.index("x") + 1]
            y_ava_type_list = []

            y_ava_type = self.get_available_data_type('y', tokens[tokens.index("mark") + 1])

            agg_type = tokens[tokens.index("aggregate") + 1]
            if agg_type in ['count', 'sum', 'average']:
                y_ava_type = [Type.numerical]

            chart_type = tokens[tokens.index("mark") + 1]
            y_all = dp.column_names.copy()
            for index,value in enumerate(y_all):
                if chart_type == 'heatmap' and value == x_axis:
                    continue
                if tokens[tokens.index('transform') + 1] == 'false' and value == x_axis:
                    continue
                if Type.getType(dp.column_types[index]) in y_ava_type:
                    y_ava_type_list.append(value)

            if y_ava_type_list == []:
                return ['*!STOP!*']
            else:
                return y_ava_type_list
        if '[F]' in tokens:
            group = ["none"]
            return group
        if '[G]' in tokens:
            if constraints.get('[G]') is not None and constraints.get('[G]') != []:
                return constraints['[G]']
            
            group_all = dp.column_names.copy()
            group = [tokens[tokens.index("x") + 1]]
            if tokens[tokens.index("aggregate") + 1] == 'none':
                group.append('none')
            return group
        if '[B]' in tokens:
            if constraints.get('[B]') is not None and constraints.get('[B]') != []:
                return constraints['[B]']
            group_type = tokens[tokens.index("group") + 1]
            if ',' in group_type:
                return ['none']

            x_cur_type = Type.getType(dp.column_types[dp.column_names.index(tokens[tokens.index("x") + 1])])
            y_cur_type = Type.getType(dp.column_types[dp.column_names.index(tokens[tokens.index("aggregate") + 2])])

            chart_type = tokens[tokens.index("mark") + 1]

            if x_cur_type == Type.numerical:
                if y_cur_type == Type.numerical:
                    group = ['none', 'auto', 'ZERO']  
                else:
                    group = ['none','ZERO']
            elif x_cur_type == Type.temporal:
                if y_cur_type == Type.numerical:
                    group = ['none', 'auto', 'date','day'] 
                else:
                    group = ['none','date','day']
            else:
                group = ['none']  
            return group
        if '[S]' in tokens:
            chart_type = tokens[tokens.index("mark") + 1]
            if chart_type == 'line':
                return ['none']
            group = ['none','asc','desc']  
            return group
        if '[K]' in tokens:
            group = ['none','5','10','20'] 
            return group

        return ''

    def get_state_result(self):
        if self.query.find('[') != -1:
            is_over = False
        else:
            is_over = True
        if self.query.find('*!STOP!*') != -1:
            is_over = True
        return is_over, self.query


    def get_next_state(self, action):
        q = self.query
        if q.find('[T]') != -1:
            if action == 'line':
                q = q.replace('[S]', 'none')
                q = q.replace('[T]', action)
            else:
                q = q.replace('[T]', action)
        elif q.find('[X]') != -1:
            q = q.replace('[X]', action)

        elif q.find('[TransForm]') != -1:
            if action == 'false':
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
            q = q.replace('[AggFunction]', action)
        elif q.find('[Y]') != -1:
            q = q.replace('[Y]', action)
        elif q.find('[F]') != -1:
            q = q.replace('[F]', action)
        elif q.find('[G]') != -1:
            if len(action.split(",")) > 1:
                q = q.replace('[B]', 'none')
            q = q.replace('[G]', action)
        elif q.find('[B]') != -1:

            if action == 'ZERO':
                q = q.replace('[S]', 'none')
                q = q.replace('[K]', 'none')
                q = q.replace('[B]', action)
            else:
                q = q.replace('[B]', action)
        elif q.find('[S]') != -1:
            q = q.replace('[S]', action)
        elif q.find('[K]') != -1:
            q = q.replace('[K]', action)

        self.query = q
        return self



