import math
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
from numpy import corrcoef
from .features import Type
from functools import reduce

class Chart(object):
    scatter = 0
    line = 1
    bar = 2
    pie = 3
    heatmap = 4
    box = 5
    chart = ['scatter','line','bar','pie', 'heatmap', 'box']

class View(object):
    """
    Attributes:
        table(Table): the table corresponding to this view.
        fx(Feature): the attributes of axis x.
        fy(Feature): the attributes of axis y.
        x_name(str): the name of axis x.
        y_name(str): the name of axis y.
        series_num(int): the number of classification.
        X(list): the data of axis x.
        Y(list): the data of axis y.
        chart(str): the type of the chart, including bar, line ,scatter ans pie.
        tuple_num(int): tuple_num in the corresponding table (the number of columns after transformation).
        score_l(float): the score of the chart in learning_to_rank method.
        M(float): M value in the paper.
        Q(float): Q value in the paper.
        score(float): the score of the chart in partial_order method.
    """
    def __init__(self,table,x_id,y_id,z_id,series_num,X,Y,chart):
        self.table = table
        self.fx = table.features[x_id]
        self.fy = table.features[y_id]
        self.x_name = self.fx.name
        self.y_name = self.fy.name
        self.z_id = z_id
        self.series_num = series_num
        self.X = X
        self.Y = Y
        self.chart = chart
        self.tuple_num = table.tuple_num
        self.score_l = 0 #learning_to_rank score
        self.M = self.Q = self.W = self.score = 0 # partial and div_ranking score
        self.getM()
        self.getQ()
        self.describe = self.output_describe()


#### function for learning to rank
    def getCorrelation_l(self,series_id):
        """
        Calculate correlation coefficient of X and Y, log(X) and Y, X and log(Y), log(X) and log(Y)
        to determine the relationship of X and Y such as linear, exponential, logarithm and power.
        (especially for learning_to_rank method)

        Args:
            series_id(int): the index of X and Y(list), determining correlation coefficient of which
                            two columns are to be calculated.
            
        Returns:
            result(float): For the correlation coefficient of X and Y, log(X) and Y, X and log(Y),
                           log(X) and log(Y), result is the max of the four correlation coefficient.
            
        """
        if self.fx.type == Type.categorical: # regard the corrcoef of categorical as 0
            return 0
        if self.fx.type == Type.temporal:
            data1 = [i for i in range(self.tuple_num // self.series_num)]
        else:
            data1 = self.X[series_id]
        data2 = self.Y[series_id]
        log_data1 = log_data2 = []
        if self.fx.type != Type.temporal and self.fx.min != '' and self.fx.min > 0:
            log_data1 = map(math.log, data1)
        try:
            if self.fy.min != '' and self.fy.min > 0:
                log_data2 = map(math.log, data2)
        except:
            pass
        result = 0
        # calculate and compare correlation
        # linear
        try:
            result = abs(corrcoef(data1, data2)[0][1])
        except Exception as e:
            result = 0
        else:
            pass

        # exponential
        if log_data2:
            try:
                r = abs(corrcoef(data1, log_data2)[0][1])
                if r > result:
                    result = r
            except Exception as e:
                result = 0
            else:
                pass

        # logarithm
        if log_data1:
            try:
                r = abs(corrcoef(log_data1, data2)[0][1])
                if r > result:
                    result = r
            except Exception as e:
                result = 0
            else:
                pass

        # power
        if log_data1 and log_data2:
            try:
                r = abs(corrcoef(log_data1, log_data2)[0][1])
                if r > result:
                    result = r
            except Exception as e:
                result = 0
            else:
                pass
        if not -1 <= result <= 1:
            result = 0
        return result

    def output_score(self):
        """
        For learning_to_rank method, get score of each chart and write to files

        Args:
            None
            
        Returns:
            The string which needs to be written in .score file
            
        """
        correlation = max([self.getCorrelation_l(i) for i in range(self.series_num)])
        if self.fx.min == '':
            self.fx.min = 0
        if self.fy.min == '':
            self.fy.min = 0
        if self.fx.type == Type.temporal:
            return '1 qid:1 1:' + str(self.fx.type) + ' 2:' + str(self.fy.type) + ' 3:' + str(
                self.tuple_num) + ' 4:' + str(self.tuple_num) + ' 5:0 6:' + str(self.fy.min) + ' 7:0 8:' + str(
                self.fy.max) + ' 9:' + str(self.fx.distinct) + ' 10:' + str(self.fy.distinct) + ' 11:' + str(
                self.fx.ratio) + ' 12:' + str(self.fy.ratio) + ' 13:' + str(correlation) + ' 14:' + str(self.chart)
        else:
            return '1 qid:1 1:' + str(self.fx.type) + ' 2:' + str(self.fy.type) + ' 3:' + str(
                self.tuple_num) + ' 4:' + str(self.tuple_num) + ' 5:' + str(self.fx.min) + ' 6:' + str(
                self.fy.min) + ' 7:' + str(self.fx.max) + ' 8:' + str(self.fy.max) + ' 9:' + str(
                self.fx.distinct) + ' 10:' + str(self.fy.distinct) + ' 11:' + str(self.fx.ratio) + ' 12:' + str(
                self.fy.ratio) + ' 13:' + str(correlation) + ' 14:' + str(self.chart)

#### function for partial order and diversified ranking 
    def getCorrelation(self,series_id):
        """
        Calculate correlation coefficient of X and Y, log(X) and Y, X and log(Y), log(X) and log(Y)
        to determine the relationship of X and Y such as linear, exponential, logarithm and power.
        (especially for partial order and diversified ranking methods)

        Args:
            series_id(int): the index of X and Y(list), determining correlation coefficient of which
                            two columns are to be calculated.
            
        Returns:
            result(float): For the correlation coefficient of X and Y, log(X) and Y, X and log(Y),
                           log(X) and log(Y), result is the max of the four correlation coefficient.
            
        """
        if self.fx.type == Type.temporal:
            data1 = [i for i in range(self.tuple_num // self.series_num)]
        else:
            if series_id < len(self.X):
                data1 = self.X[series_id]
        data2 = self.Y[series_id]
        log_data1 = log_data2 = []
        if self.fx.type != Type.temporal and self.fx.min != '' and self.fx.min > 0:
            log_data1 = map(math.log, data1)
        if self.fy.minmin != '' and self.fy.minmin > 0:
            log_data2 = map(math.log, data2)
        log_data2 = map(math.log, data2)
        result = 0
        # linear
        try:
            result = abs(corrcoef(data1, data2)[0][1])
        except Exception as e:
            result = 0
        

        # exponential
        if log_data2:
            try:
                r = abs(corrcoef(data1, log_data2)[0][1])
                if r > result:
                    result = r
            except Exception as e:
                pass

        # logarithm
        if log_data1:
            try:
                r = abs(corrcoef(log_data1, data2)[0][1])
                if r > result:
                    result = r
            except Exception as e:
                pass
                # print("3 ", e)
                # result = 0
            else:
                pass

        # power
        if log_data1 and log_data2:
            try:
                r = abs(corrcoef(log_data1, log_data2)[0][1])
                if r > result:
                    result = r
            except Exception as e:
                pass
                # print("4 ", e)
                # result = 0
            else:
                pass

        return result

    def getM(self):
        """
        Calculate M value in the paper

        Args:
            None
            
        Returns:
            None
            
        """
        if self.chart == Chart.pie:
            if self.tuple_num == 1:
                self.M = 0
            elif 2 <= self.tuple_num <= 10:
                sumY = sum(self.Y[0])
                self.M = reduce(lambda x, y: x + y, map(lambda y: -(1.0 * y / sumY) * math.log(1.0 * y / sumY),self.Y[0]))
            elif self.tuple_num > 10:
                sumY = sum(self.Y[0])
                self.M = reduce(lambda x, y: x + y, map(lambda y: -(1.0 * y / sumY) * math.log(1.0 * y / sumY), self.Y[0])) * 10.0 / (self.tuple_num)
        elif self.chart == Chart.bar:
            if self.tuple_num // self.series_num == 1:
                self.M = 0
            elif 2 <= self.tuple_num // self.series_num <= 20:
                self.M = 1
            else:
                self.M = 20 / (self.tuple_num // self.series_num)
        elif self.chart == Chart.scatter:
            if self.series_num == 1:
                self.M = self.getCorrelation(0)
            else:
                self.M = max([self.getCorrelation(i) for i in range(self.series_num)])
        else: #if self.chart == Chart.line
            if self.series_num == 1:
                if self.getCorrelation(0) > 0.3:
                    self.M = 1
                else:
                    self.M = 0
            else:
                if max([self.getCorrelation(i) for i in range(self.series_num)]) > 0.3:
                    self.M = 1
                else:
                    self.M = 0

    def getQ(self):
        """
        Calculate Q value in the paper

        Args:
            None
            
        Returns:
            None
            
        """
        #self.Q = 1
        #if self.chart == Chart.bar or self.chart == Chart.pie:
        self.Q = 1 - 1.0 * (self.tuple_num / self.series_num) / self.table.instance.tuple_num


#### function for all
    def output(self,order):
        """
            Encapsulate the value of several variables in variable data(ruturned value).

        Args:
            order(int): Not an important argument, only used in the assignment of data.
            
        Returns:
            data(str): A string including the value of several variables:
                       order1, order2, describe, x_name, y_name, chart, classify, x_data, y_data.
            
        """
        classify = str([])
        if self.series_num > 1:
            classify = str([v[0] for v in self.table.classes]).replace("u'", '\'').replace("'",'"')
        x_data = str(self.X)
        if self.fx.type == Type.numerical:
            x_data = str(self.X).replace("'", '').replace('"', '').replace('L', '')
        elif self.fx.type == Type.categorical:
            x_data = str(self.X).replace("u'", '\'').replace("'", '"')
        else:
            len_x = len(self.X)
            x_data = '["%s"]' % ''.join(list(reduce(lambda s1, s2: s1 + s2, ['","'.join(list(map(str, self.X[i]))) for i in range(len_x)]).replace("'",'"')))
        y_data = str(self.Y)
        if self.fy.type == Type.numerical:
            y_data = str(self.Y).replace("'", '').replace('"', '').replace('L', '')
        elif self.fy.type == Type.categorical:
            y_data = str(self.Y).replace("u'", '\'').replace("'", '"')
        else:
            len_y = len(self.Y)
            y_data = '["%s"]' % ''.join(list(reduce(lambda s1, s2: s1 + s2, ['","'.join(list(map(str, self.Y[i]))) for i in range(len_y)]).replace("'",'"')))
        data = '{"order1":' + str(order) + ',"order2":' + str(1) +  ',"describe":"' + self.table.describe + '","x_name":"' + self.fx.name + '","y_name":"' + self.fy.name + '","chart":"' + Chart.chart[self.chart] + '","classify":' + classify + ',"x_data":' + x_data + ',"y_data":' + y_data + '}'
        return data
        

    def output_describe(self):
        """
            Encapsulate the value of several variables in variable data(ruturned value).

        Args:
            order(int): Not an important argument, only used in the assignment of data.
            
        Returns:
            data(str): A string including the value of several variables:
                       order1, order2, describe, x_name, y_name, chart, classify, x_data, y_data.
            
        """
        classify = str([])
        if self.series_num > 1:
            classify = str([v[0] for v in self.table.classes]).replace("u'", '\'').replace("'",'"')
        #if self.fy.type == Type.numerical:
        #    y_data = y_data.replace('L', '')
        # data = '{"describe":"' + self.table.describe + '","x_name":"' + self.fx.name + '","y_name":"' + self.fy.name + '","chart":"' + Chart.chart[self.chart] + '}'
        data = f"chart: {Chart.chart[self.chart]} x_name: {self.fx.name} y_name: {self.fy.name} describe: {self.table.describe}"
        #data = 'score:' + str(round(self.score, 2)) + '\tM:' + str(round(self.M, 2)) + '\tQ:' + str(round(self.Q, 2)) + '\tW:' + str(round(self.W, 2)) + '{"order":' + str(order) + ',"describe":"' + self.table.describe + '","x_name":"' + self.fx.name + '","y_name":"' + self.fy.name + '","chart":"' + Chart.chart[self.chart] + '","classify":' + classify + ',"x_data":' + x_data + ',"y_data":' + y_data + '}'
        return data.lower()


