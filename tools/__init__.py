# coding:utf-8
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import datetime
from pprint import *
from pyecharts.charts import Bar, Line, Scatter, Pie, Grid, Page, HeatMap, Boxplot
from pyecharts import options as opts

from .instance import Instance
from .table_l import Table as Table_LTR  # table of Learning-to-rank model
from .view import Chart
from .features import Type

import re  # use regular expressions when recognizing the date type

from IPython.core.display import display, HTML

import numpy as np
import datetime

methods_of_import = ['none', 'mysql', 'csv']
methods_of_ranking = ['none', 'learn_to_rank', 'partial_order', 'diversified_ranking']


class default(object):
    def __init__(self):
        return


def is_vaild_datetime(old_date):
    try:
        datetime.datetime.strptime(old_date, '%Y-%m-%d %H:%M:%S')
        pd.to_datetime(old_date)
        return True
    except Exception as e:
        return False


class haichart(object):

    ##### initial function
    def __init__(self, *name):
        """
        Initialize the table infomation: name, istable, importmethod, rankmethod
        There are two ways of import method: mysql and csv.(only csv can use yet)
        Three ways of ranking: learn_to_rank(rank by machine learning)
                               partial_order(rank by rules of expert knowledge)
                               diversified_ranking(hybrid of the two methods)

        Args:
            name(str): The name of the table(s)

        Returns:
            None

        """
        if not name:
            self.name = 'haichart'  # if name is empty, set default name 'haichart'
        else:
            self.name = name
        self.is_table_info = False
        self.import_method = methods_of_import[0]  # = none
        self.rank_method = methods_of_ranking[0]  # = none
        self.eh_view = {}

    def table_info(self, name, column_info, *column_info2):
        """
        Input the table_info.

        Args:
            name(str): The name of the table(s)
            column_info(list): The name of each column
            column_info2(list or tuble or dict, usually list): The type of each column

        Returns:
            None
            
        """
        self.table_name = name
        self.column_names = []
        self.column_types = []
        if isinstance(column_info, list) and isinstance(column_info2[0], list):
            self.column_names = column_info
            self.column_types = column_info2[0]
        elif isinstance(column_info, dict):
            self.column_names = column_info.keys()
            self.column_types = column_info.values()
        elif isinstance(column_info, list) and isinstance(column_info[0], tuple):
            self.column_names = [i[0] for i in column_info]
            self.column_types = [i[1] for i in column_info]
        else:
            raise TypeError("unsupported argument types (%s, %s)" % (type(column_info), type(column_info2)))
        for idx, val in enumerate(self.column_types):
            if Type.getType(val.lower()) == 0:  # not a normal type
                raise Exception(
                    "doesnt support this column_type \' %s \' of column name \' %s \',please check Readme for specification " % (
                        val, self.column_names[idx]))
        self.is_table_info = True

    def error_throw(self, stage):
        """
        Find if there are errors at the beginning of each function in this file

        Args:
            stage(str): distinguish which function calls errow_throw:
                        rank: call errow_throw when executing the ranking function
                        output: call errow_throw when ececuting the output function

        Returns:
            None
            
        """
        if self.is_table_info == False:
            print("please enter table info by table_info()")
            sys.exit(0)
        if stage == 'rank':
            if self.import_method == 'none':
                self.error_output_import()
        elif stage == 'output':
            if self.import_method == 'none':
                self.error_output_import()
            else:
                if self.rank_method == 'none':
                    self.error_output_rank()

    def error_output_import(self):
        """
        Print import error information

        Args:
            None

        Returns:
            None
            
        """
        im_methods_string = ''
        for i in range(len(methods_of_import)):
            if i == 0:
                continue
            elif i != len(methods_of_import) - 1:
                im_methods_string += ('from_' + methods_of_import[i] + '() or ')
            else:  # i == len(methods_of_import)
                im_methods_string += ('from_' + methods_of_import[i] + '()')
        print("please import by " + im_methods_string)
        sys.exit(0)

    def error_output_rank(self):
        """
        Print rank error information

        Args:
            None

        Returns:
            None
            
        """
        rank_method_string = ''
        for i in range(len(methods_of_ranking)):
            if i == 0:
                continue
            elif i != len(methods_of_ranking) - 1:
                rank_method_string += (methods_of_ranking[i] + '() or ')
            else:
                rank_method_string += (methods_of_ranking[i] + '()')
        print(" " + rank_method_string)
        sys.exit(0)

    ##### data import function

    def from_csv_new(self, path):
        """
        Read the csv file and process its contents, replacing spaces in column names with "-".

        Args:
            path(str): the path of the csv file

        Returns:
            None
        """
        self.csv_path = path

        try:
            # Read CSV file using pandas
            self.csv_dataframe = pd.read_csv(self.csv_path, header=0, keep_default_na=False).dropna(axis=0, how='any')

            # Replace spaces in column names with "-"
            self.csv_dataframe.columns = [col.replace(' ', '-').replace('_', '-') for col in self.csv_dataframe.columns]

            # Extract column names and types
            column_names = self.csv_dataframe.columns.tolist()
            column_types = [self._determine_column_type(self.csv_dataframe[col]) for col in column_names]

            # Process the table information
            table_name = path.rsplit('/', 1)[-1][:-4]
            self.table_info(table_name, column_names, column_types)
            self.import_method = 'csv'  # Set the import method

        except FileNotFoundError:
            print("Error: File not found -", path)
        except pd.errors.EmptyDataError:
            print("Error: No data - File is empty")
        except Exception as e:
            print("An error occurred:", e)

    def _determine_column_type(self, series):
        """
        Determine the type of a column in the DataFrame.

        Args:
            series (pd.Series): A pandas Series representing a column in the DataFrame.

        Returns:
            str: The determined type of the column.
        """
        col_type = series.dtype.name
        if col_type.startswith(('int', 'float')):
            if series.apply(lambda x: x == 0 or (1000 < x < 2100)).all():
                return 'year'
            return col_type
        elif col_type == 'object':
            if series.apply(lambda x: isinstance(x, str) and re.match(r'\d+[/-]\d+[/-]\d+', x)).any():
                return 'date'
            return 'varchar'
        return col_type

    def from_csv(self, path):
        """
        read the csv file

        Args:
            path(str): the path of the csv file

        Returns:
            None
            
        """
        self.csv_path = path

        try:
            fh = open(self.csv_path, "r", encoding='utf-8')
            a = fh.readline()
            a = a[:-1]  # remove '\n'
            # x = a.replace('$', '').split(',')  # x stores the name of each column

            column_names = a.split(',')
            column_names = [name.replace(' ', '-').replace('_', '-') for name in column_names]
            x = column_names  # x stores the name of each column
        except IOError:
            print("Error: no such file or directory")

        fh.close()

        test = pd.DataFrame(pd.read_csv(self.csv_path, engine='c',names=column_names, skiprows=1)).dropna(axis=0, how='any')
        types = [0 for i in range(len(test.dtypes))]

        # type transformation
        for i in range(len(test.dtypes)):
            if test.dtypes[i].name[0:3] == 'int' or test.dtypes[i].name[0:5] == 'float':
                if (x[i][0] == "'" or x[i][0] == '"'):  
                    x[i] = x[i].replace('\'', '').replace('"', '')
                for j in test[x[i]]:
                    # if not (j == 0 or (j > 1000 and j < 2100)):
                    if (not ((j > 1900 and j < 2100))) or isinstance(j, float):
                        types[i] = test.dtypes[i].name[0:5]
                        break
                    else:
                        types[i] = 'year'
            elif test.dtypes[i].name[0:6] == 'object' or test.dtypes[i].name[0:6] == 'bool':
                if (x[i][0] == "'" or x[i][0] == '"'):  
                    x[i] = x[i].replace('\'', '').replace('"', '')
                for j in test[x[i]]:
                    my_re = re.compile(r'[A-Za-z]', re.S)
                    res = re.findall(my_re, str(j))
                    # if j != 0 and (not(re.search(r'\d+[/-]\d+[/-]\d+', j)) or len(res)!=0):

                    if not is_vaild_datetime(j) and (j != 0 or j == False) and (not (re.search(
                            r'^((((1[6-9]|[2-9]\d)\d{2})(\/|\-)(0?[13578]|1[02])(\/|\-)(0?[1-9]|[12]\d|3[01]))|(((1[6-9]|[2-9]\d)\d{2})(\/|\-)(0?[13456789]|1[012])(\/|\-)(0?[1-9]|[12]\d|30))|(((1[6-9]|[2-9]\d)\d{2})(\/|\-)0?2(\/|\-)(0?[1-9]|1\d|2[0-8]))|(((1[6-9]|[2-9]\d)(0[48]|[2468][048]|[13579][26])|((16|[2468][048]|[3579][26])00))-0?2-29-))$',
                            str(j))) or len(res) != 0):
                        types[i] = 'varchar'
                        break
                    else:
                        if is_vaild_datetime(j):
                            types[i] = 'datetime'
                        else:
                            types[i] = 'date'

        raw_csv = pd.read_csv(self.csv_path, header=0, keep_default_na=False, engine='c')
        self.csv_dataframe = pd.DataFrame(raw_csv).dropna(axis=0, how='any')
        name = path.rsplit('/', 1)[-1][:-4]
        for index in range(len(x)):
            x[index] = x[index].replace('$', '')
        self.table_info(name.replace('$', ''), x, types)
        self.import_method = methods_of_import[2]  # = 'csv'

        self.show_csv_info()

    def csv_handle(self, instance):
        """
        format the data according to the type

        Args:
            instance(Instance): the object of class Instance

        Returns:
            the instance object with the infomation(names, types, etc.)
            
        """
        table_origin = self.csv_dataframe
        in_column_num = len(self.column_names)
        in_column_name = self.column_names
        in_column_type = self.column_types

        instance.column_num = instance.tables[0].column_num = in_column_num
        for i in range(instance.column_num):
            instance.tables[0].names.append(in_column_name[i])
            instance.tables[0].types.append(
                Type.getType(in_column_type[i].lower()))  
        instance.tables[0].origins = [i for i in range(instance.tables[0].column_num)]

        instance.tuple_num = instance.tables[0].tuple_num = table_origin.shape[0]  # the number of rows
        for i in range(instance.tables[0].column_num):
            if instance.tables[0].types[
                i] == 3:  # if there is date type column in csv,convert into datetime format
                col_name = table_origin.columns[i]
                col_type = self.column_types[i]
                self.csv_handle_changedate(col_name, col_type)

        # change table column name with table_info column_names (for date type columns)
        try:
            for i in range(len(table_origin.columns)):
                table_origin.rename(columns={table_origin.columns[i]: in_column_name[i]}, inplace=True)
        except Exception as e:
            print(e)

        instance.tables[
            0].D = table_origin.values.tolist()  #  dataframe to list type and feed to D(where to store all the table info )
        return instance

    def csv_handle_changedate(self, col_name, col_type):
        """
        deal with date type data, wrap to datetime format

        Args:
            col_name(str): the name of columns
            col_type(str): the type of columns

        Returns:
            None
            
        """
        table = self.csv_dataframe
        if col_type == 'date':
            table[col_name] = pd.to_datetime(table[col_name]).dt.date
        elif col_type == 'datetime':
            table[col_name] = pd.to_datetime(table[col_name]).dt.to_pydatetime()
        elif col_type == 'year':
            # table[col_name] = pd.to_datetime(table[col_name].apply(lambda x: str(x)+'/1/1')).dt.date
            table[col_name] = pd.to_datetime(table[col_name].apply(lambda x: str(x))).dt.date

    def show_csv_info(self):
        """
        print out csv info

        Args:

        Returns:
            None
            
        """
        # print()
        # display(HTML(self.csv_dataframe.head(10).to_html()))

    ##### ranking function
    def rank_generate_all_views(self,instance):
        """
        initialize before ranking 


        Args:
            instance(Instance): The object of class Instance

        Returns:
            instance with tables added
            
        """
        if len(instance.tables[0].D) == 0:
            print ('no data in table')
            sys.exit(0)
        # print(instance.table_num, instance.view_num)
        instance.addTables(instance.tables[0].dealWithTable()) # the first deal with is to transform the table into several small ones
        # print(instance.table_num, instance.view_num)
        begin_id = 1
        while begin_id < instance.table_num:
            instance.tables[begin_id].dealWithTable() # to generate views
            begin_id += 1
        if instance.view_num == 0:
            print ('no chart generated')
            # sys.exit(0)
        # print(instance.table_num, instance.view_num)
        return instance


    def learning_to_rank(self):
        """
        use Learn_to_rank method to rank the charts

        Args:
            None
            
        Returns:
            None
            
        """
        self.error_throw('rank')

        instance = Instance(self.table_name)
        instance.addTable(Table_LTR(instance, False, '', ''))
        if self.import_method == 'mysql':
            instance = self.mysql_handle(instance)
        elif self.import_method == 'csv':
            instance = self.csv_handle(instance)  

        self.rank_learning(instance)

        self.rank_method = methods_of_ranking[1] # = 'learn_to_rank'
    


    def rank_learning(self, instance):
        """
        inner function of learning_to_rank

        Args:
            instance(Instance): The object of class Instance.
            
        Returns:
            None
        """
        instance = self.rank_generate_all_views(instance)
        instance.getScore_learning_to_rank()
        self.instance = instance


    def rank_partial(self, instance):
        """
        inner function of partial_order and diversified_ranking

        Args:
            instance(Instance): The object of class Instance.
            
        Returns:
            None
            
        """
        instance = self.rank_generate_all_views(instance)

        # mark bar data hollywoods_stories encoding x film y aggregate none year transform group theme
        instance.getM()
        instance.getW()
        instance.getScore()

        self.instance = instance

    ##### output function : list, print, single_json, multiple_jsons, single_html, multiple_htmls, 6 in total.
    def to_list(self):
        """
        export as list type

        Args:
            None
            
        Returns:
            the export list
            
        """
        # self.error_throw('output')

        if self.rank_method == methods_of_ranking[3]:  # 'diversified_ranking'
            export_list = self.output_div('list')
        else:
            export_list = self.output('list')
        return export_list

    def to_print_out(self):
        """
        print out to cmd

        Args:
            None
            
        Returns:
            None
            
        """
        self.error_throw('output')

        if self.rank_method == methods_of_ranking[3]:  # 'diversified_ranking'
            self.output_div('print')
        else:
            self.output('print')

    def to_single_json(self):
        """
        create a single json file

        Args:
            None
            
        Returns:
            None
            
        """
        self.error_throw('output')

        if self.rank_method == methods_of_ranking[3]:  # 'diversified_ranking'
            self.output_div('single_json')
        else:
            self.output('single_json')

    def to_multiple_jsons(self):
        """
        create multiple json files

        Args:
            None
            
        Returns:
            None
            
        """
        self.error_throw('output')

        if self.rank_method == methods_of_ranking[3]:  # 'diversified_ranking'
            self.output_div('multiple_jsons')
        else:
            self.output('multiple_jsons')

    def to_single_html(self, dict_sorted, tname):
        """
        convert to html by pyecharts and output to single html file

        Args:
            None
            
        Returns:
            None
            
        """

        self.output('single_html', dict_sorted, tname)

    def to_multiple_htmls(self):
        """
        convert to html by pyecharts and output to multiple html files

        Args:
            None
            
        Returns:
            None
            
        """
        self.error_throw('output')

        
        self.output('multiple_htmls')

    def output(self, output_method, dict_sorted, tname):
        """
        output function of partial_order and learning_to_rank for all kinds of output

        Args:
            output_method(str): output method:
                                list: to list
                                print: print to console
                                single_json/multiple_jsons: single/multiple json file(s)
                                single_html/multiple_htmls: single/multiple html file(s)

        Returns:
            None

        """
        instance = self.instance
        order1 = order2 = 1
        old_view = ''
        export_dict = {}

        path2 = os.getcwd() + '/html/'
        if not os.path.exists(path2):
            os.mkdir(path2)
        page = Page()
        if output_method == 'single_html':  
            self.page = Page()
            num = 1
            for list in dict_sorted:
                view = list[1]
                if old_view:
                    order2 = 1
                    order1 += 1
                old_view = view
                self.html_output(order1, view, 'single', tname, num)
                num = num + 1
            self.page.render('./html/' + tname + '_all' + '.html')
        elif output_method == 'list':
            for i in range(instance.view_num):
                view = instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos]
                if old_view:
                    order2 = 1
                    order1 += 1
                export_dict[view.output_describe()] = view
                # export_list.append(view.output(order1))
                old_view = view
            return export_dict

    def output_list(self, output_method):
        """
        output function of partial_order and learning_to_rank for all kinds of output

        Args:
            output_method(str): output method:
                                list: to list
                                print: print to console
                                single_json/multiple_jsons: single/multiple json file(s)
                                single_html/multiple_htmls: single/multiple html file(s)
            
        Returns:
            None
            
        """
        instance = self.instance
        export_list = []
        export_dict = {}
        order1 = order2 = 1
        old_view = ''
        if output_method == 'list':
            for i in range(instance.view_num):
                view = instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos]
                if old_view:
                    order2 = 1
                    order1 += 1
                export_dict[view.output_describe()] = view
                # export_list.append(view.output(order1))
                old_view = view
            return export_dict
        elif output_method == 'print':
            for i in range(instance.view_num):
                view = instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos]
                if old_view:
                    order2 = 1
                    order1 += 1
                pprint (view.output(order1))
                old_view = view
            return
        elif output_method == 'single_json' or output_method == 'multiple_jsons':
            path2 = os.getcwd() + '/json/'
            if not os.path.exists(path2):
                os.mkdir(path2)
            if output_method == 'single_json':
                f = open(path2 + self.table_name + '.json','w')
                for i in range(instance.view_num):
                    view = instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos]
                    if old_view:
                        order2 = 1
                        order1 += 1
                    f.write(view.output(order1) + '\n')
                    old_view = view
                f.close() # Notice that f.close() is out of the loop to create only one file
            else: # if output_method == 'multiple_jsons'
                for i in range(instance.view_num):
                    view = instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos]
                    if old_view:
                        order2 = 1
                        order1 += 1
                    f = open(path2 + self.table_name + str(order1) + '.json','w')
                    f.write(view.output(order1))
                    f.close() # Notice that f.close() is in the loop to create multiple files
                    old_view = view
            return
        elif output_method == 'single_html' or output_method == 'multiple_htmls':
            path2 = os.getcwd() + '/html/'
            if not os.path.exists(path2):
                os.mkdir(path2)
            page = Page()
            if output_method == 'single_html':
                self.page = Page()
                for i in range(instance.view_num):
                    view = instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos]
                    if old_view:
                        order2 = 1
                        order1 += 1
                    old_view = view
                    self.html_output(order1, view, 'single')
                self.page.render('./html/' + self.table_name + '_all' + '.html')
            else: # if output_method == 'multiple_htmls'
                path3 = os.getcwd() + '/html/' + self.table_name
                if not os.path.exists(path3):
                    os.mkdir(path3)
                for i in range(instance.view_num):
                    view = instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos]
                    if old_view:
                        order2 = 1
                        order1 += 1
                    old_view = view
                    self.html_output(order1, view, 'multiple')
            return

    def html_output(self, order, view, mode, tname, num):
        """
        output function of html

        Args:
            order(int): diversified_ranking use different order
            view(View): view object
            mode(str): single or multiple
            
        Returns:
            None
            
        """
        try:
            data = {}
            data['order'] = order
            data['chartname'] = str(num) + "、" + tname
            data['describe'] = view.table.describe
            data['x_name'] = view.fx.name
            data['y_name'] = view.fy.name
            data['chart'] = Chart.chart[view.chart]
            data['classify'] = [v[0] for v in view.table.classes]
            data['x_data'] = view.X
            data['y_data'] = view.Y
            data['title_top'] = 5
            [chart, filename] = self.html_handle(data, view.table.D)
        except Exception as e:
            print(f"error{e}-------------")
            traceback.print_exc()  # 打印详细的错误堆栈信息
            sys.exit()

        # if data['chart'] == 'pie':
        #     pos_top = '30%'
        #     pos_bottom = '10%'
        # else:
        #     pos_top = '25%'  
        #     pos_bottom = '20%'

        grid = Grid()

        percent = '20%'
        # grid.add(chart, grid_opts=opts.GridOpts(pos_bottom=pos_bottom, pos_top=pos_top, pos_left=percent, pos_right=percent))
        grid.add(chart, grid_opts=opts.GridOpts(pos_bottom="20%", pos_top="25%", pos_left=percent, pos_right=percent))
        # grid.add(chart, grid_opts=opts.GridOpts(pos_bottom='20%', pos_top='20%', pos_left='10%'))
        if mode == 'single':
            self.page.add(grid)  # the grid is added in the same page
        elif mode == 'multiple':
            grid.render('./html/' + self.table_name + '/' + filename)  # the grid is added in a new file

    # * 画图看这里
    def html_handle(self, data, table_data):
        """
        convert function to html by pyecharts

        Args:
            data(dict): the data info
            
        Returns:
            chart: chart generated by pyecharts: Bar, Pie, Line or Scatter
            filename: html file name
            
        """

        filename = self.table_name + str(data['order']) + '.html'
        margin = str(data['title_top']) + '%'

        # common_axis_label_opts = opts.LabelOpts(font_size=20)

        my_font_size = 22
        common_text_style = opts.TextStyleOpts(font_size=my_font_size)  

        common_label_opts = opts.LabelOpts(font_size=my_font_size)
        
        chart_subtitle = ""
        if data['describe'] != '':
            chart_subtitle = "Operation: " + data['describe'].lower()
        else:
            chart_subtitle = "Operation: none"

        # subtitle_style_opts = opts.TextStyleOpts(color="grey", font_size=13, font_family="Arial")

        data['chartname'] = data['chartname'].split('、')[-1] if '、' in data['chartname'] else data['chartname']
        
        subtitle_style_opts = opts.TextStyleOpts(font_size=24, font_weight="bold")
        common_title_opts = opts.TitleOpts(
            # title=data['chartname'],
            subtitle=chart_subtitle,
            pos_left='center',
            pos_top=margin,
            # pos_top="-10%",
            title_textstyle_opts=common_text_style,
            subtitle_textstyle_opts=subtitle_style_opts
            # pos_left="5%",  
            # pos_top="1%"  
            # pos_top="10%"
            # subtitle_textstyle_opts=subtitle_style_opts
        )

        # common_legend_opts = opts.LegendOpts(textstyle_opts=common_text_style)

        common_axis_opts = opts.AxisOpts(
            axislabel_opts=common_label_opts,
            name_textstyle_opts=common_text_style
        )

        # tool_percent = '30%'
        common_toolbox_opts = opts.ToolboxOpts(
            item_gap=5,
            item_size = 17,
            # pos_top="1%",  
            # pos_top="5%",
            # pos_bottom="90%",
            pos_right="20%",  
            pos_left="75%",  
            # pos_top=tool_percent,  
            # pos_bottom=tool_percent,  
            feature={
                "dataZoom": {"show": True, "title": {
                            "zoom": "Zoom In/Out",
                            "back": "Reset Zoom"
                            }},
                "dataView": {"show": True, "title": "Data View", "lang": ["Data View", "Close", "Refresh"]},
                "magicType": {"type": ["line", "bar", "stack"], "title": {
                            "line": "Switch to Line Chart",
                            "bar": "Switch to Bar Chart",
                            "stack": "Switch to Stacked Chart",
                            "tiled": "Switch to Tiled Chart"
                            }},  
                # "restore": {"show": True, "title": "Restore"},
                "saveAsImage": {"show": True, "title": "Save"},  
            }
        )

        color_flag = False
        if "," in chart_subtitle or data['chart'] == 'pie' or (data['chart'] == 'scatter' and "none" not in chart_subtitle):
            color_flag = True

        common_legend_opts = opts.LegendOpts(
            textstyle_opts=common_text_style, 
            is_show=color_flag,
            # pos_top = margin
            )

        if data['chart'] == 'bar':
            chart = (Bar().set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(
                    title_opts=common_title_opts,
                    xaxis_opts=opts.AxisOpts(name=data['x_name'],
                                            #  ,axislabel_opts=common_axis_label_opts
                                            axislabel_opts=common_label_opts,
                                            name_textstyle_opts=common_text_style
                                             ),
                    yaxis_opts=opts.AxisOpts(
                        name=data['y_name'],
                        splitline_opts=opts.SplitLineOpts(is_show=True),
                        axislabel_opts=common_label_opts,
                        name_textstyle_opts=common_text_style
                        # ,axislabel_opts=common_axis_label_opts
                    ),
                    # legend_opts=opts.LegendOpts(is_show=True,pos_left='center'),  
                    legend_opts=common_legend_opts,
                    toolbox_opts=common_toolbox_opts 
                ))

        elif data['chart'] == 'pie':
            chart = (Pie().set_global_opts(
                    title_opts=common_title_opts,
                    legend_opts=common_legend_opts,
                    toolbox_opts=common_toolbox_opts, 
                    )
                    # .set_series_opts(label_opts=opts.LabelOpts(font_size=my_font_size))
                )
            
        elif data['chart'] == 'line':
            chart = (Line().set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(
                title_opts=common_title_opts,
                xaxis_opts=opts.AxisOpts(name=data['x_name'],axislabel_opts=common_label_opts,name_textstyle_opts=common_text_style),
                yaxis_opts=opts.AxisOpts(name=data['y_name'], 
                                         splitline_opts=opts.SplitLineOpts(is_show=True),
                                         axislabel_opts=common_label_opts,
                                         name_textstyle_opts=common_text_style
                                         )
                
                ,
                    # legend_opts=opts.LegendOpts(is_show=True),
                    legend_opts=common_legend_opts,
                    toolbox_opts=common_toolbox_opts  
                ))
        elif data['chart'] == 'scatter':
            chart = (Scatter().set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(
                title_opts=common_title_opts,
                xaxis_opts=opts.AxisOpts(type_='value', name=data['x_name'],
                                         splitline_opts=opts.SplitLineOpts(is_show=True),
                                         axislabel_opts=common_label_opts,
                                         name_textstyle_opts=common_text_style
                                         ),
                yaxis_opts=opts.AxisOpts(type_='value', name=data['y_name'],
                                         splitline_opts=opts.SplitLineOpts(is_show=True),
                                         axislabel_opts=common_label_opts,
                                         name_textstyle_opts=common_text_style
                                         )
                                         ,
                    # legend_opts=opts.LegendOpts(is_show=True),
                    legend_opts=common_legend_opts,
                    toolbox_opts=common_toolbox_opts  
                    
                    
                    ))
        elif data['chart'] == 'heatmap':
            chart = (HeatMap().set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(
                title_opts=common_title_opts,
                xaxis_opts=opts.AxisOpts(name=data['x_name']),
                yaxis_opts=opts.AxisOpts(name=data['y_name'], splitline_opts=opts.SplitLineOpts(is_show=True))))
        elif data['chart'] == 'box':
            chart = (Boxplot().set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(
                title_opts=common_title_opts,
                xaxis_opts=opts.AxisOpts(name=data['x_name']),
                yaxis_opts=opts.AxisOpts(name=data['y_name'], splitline_opts=opts.SplitLineOpts(is_show=True))))
        else:
            print("not valid chart")


        # chart.set_global_opts(toolbox_opts=common_toolbox_opts)

        if not data["classify"]:  
            attr = data["x_data"][0]  
            val = data["y_data"][0]  
            if data['chart'] == 'bar':
                chart.add_xaxis(attr).add_yaxis("", val, label_opts=opts.LabelOpts(is_show=False))
            elif data['chart'] == 'box':
                x_attr = []
                val_list = []
                for box_data in table_data:
                    x_attr.append(box_data[0])
                    val_list.append(box_data[1])
                chart.add_xaxis(x_attr).add_yaxis("", val_list, label_opts=opts.LabelOpts(is_show=False))
            elif data['chart'] == 'line':
                chart.add_xaxis(attr).add_yaxis("", val, label_opts=opts.LabelOpts(is_show=False))
            elif data['chart'] == 'pie':
                data_pair = [list(z) for z in zip(attr, val)]
                series_name = data['y_name']
                label_opts = opts.LabelOpts(font_size=my_font_size)
                chart.add(series_name=series_name, 
                data_pair=data_pair,
                label_opts=label_opts)
            elif data['chart'] == 'scatter':
                if isinstance(attr[0], str):
                    attr = [x for x in attr if x != '']
                    attr = list(map(float, attr))
                if is_vaild_datetime(str(attr[0])):
                    attr = [x for x in attr if x != '']
                    attr = list(map(str, attr))
                if isinstance(val[0], str):
                    val = [x for x in val if x != '']
                    val = list(map(float, val))
                chart.add_xaxis(attr).add_yaxis("", val, label_opts=opts.LabelOpts(is_show=False))
                # .set_global_opts(
                # title_opts=opts.TitleOpts(title=data['chartname'], subtitle=data['describe'], pos_left='center',
                #                           pos_top=margin),
                # xaxis_opts=opts.AxisOpts(name=data['x_name']),
                # yaxis_opts=opts.AxisOpts(name=data['y_name'], splitline_opts=opts.SplitLineOpts(is_show=True)))
            elif data['chart'] == 'heatmap':
                x_axis_set = set()
                y_axis_set = set()
                data_dict = {}
                for i, line in enumerate(table_data):
                    if line[-2] in data_dict:
                        y_dict = data_dict.get(line[-2])
                        if line[-1] in y_dict:
                            cur_num = y_dict.get(line[-1])
                            data_dict.get(line[-2])[line[-1]] = str(int(cur_num) + int(line[-3]))
                        else:
                            data_dict.get(line[-2])[line[-1]] = line[-3]
                    else:  
                        data_dict[line[-2]] = {line[-1]: line[-3]}
                    x_axis_set.add(line[-2])
                    y_axis_set.add(line[-1])

                x_axis_list = list(x_axis_set)
                y_axis_list = list(y_axis_set)
                if '~' in x_axis_list[0]:
                    x_axis_list.sort(key=lambda d: int(d.split('~')[0]))
                    y_axis_list.sort(key=lambda d: int(d.split('~')[0]))
                elif 'th' in x_axis_list[0]:
                    x_axis_list.sort(key=lambda d: int(d.replace('th', '')))
                    y_axis_list.sort(key=lambda d: int(d.split('~')[0]))
                else:
                    x_axis_list.sort()
                    y_axis_list.sort()

                max_num = 0
                heatmap_data = []
                for x_index, x_column in enumerate(x_axis_list):
                    for y_index, y_column in enumerate(y_axis_list):
                        num_sum = data_dict.get(x_column).get(y_column)
                        if num_sum is not None:
                            num_sum = int(num_sum)
                            if num_sum > max_num:
                                max_num = num_sum
                            heatmap_data.append([x_index, y_index, num_sum])

                chart.add_xaxis(x_axis_list).add_yaxis(
                    "",
                    y_axis_list,
                    heatmap_data,
                    label_opts=opts.LabelOpts(is_show=True, position="inside"),
                ).set_global_opts(
                    visualmap_opts=opts.VisualMapOpts(max_=max_num),
                    title_opts=opts.TitleOpts(title=data['chartname'], subtitle=data['describe'], pos_left='center',
                                              pos_top=margin),
                    xaxis_opts=opts.AxisOpts(name=data['x_name']),
                    yaxis_opts=opts.AxisOpts(name=data['y_name']))


        else: 

            attr = data["x_data"][0] 
            for i in range(len(data["classify"])):  
                val = data["y_data"][i]  
                name = (
                    data["classify"][i][0] if type(data["classify"][i]) == type(('a', 'b')) else data["classify"][i])
                if i == 0:
                    if data['chart'] != 'pie' and data['chart'] != 'scatter':
                        chart.add_xaxis(attr)
                if data['chart'] == 'bar':
                    chart.add_yaxis(name, val, stack="stack1", label_opts=opts.LabelOpts(is_show=False))
                elif data['chart'] == 'line':
                    chart.add_yaxis(name, val, label_opts=opts.LabelOpts(is_show=False))
                elif data['chart'] == 'pie':
                    chart.add("", [list(z) for z in zip(attr, val)])
                elif data['chart'] == 'scatter':
                    attr_scatter = data["x_data"][i]
                    if isinstance(attr_scatter[0], str):  
                        attr_scatter = [x for x in attr_scatter if x != '']
                        attr_scatter = list(map(float, attr_scatter))
                    if isinstance(val[0], str):
                        val = [x for x in val if x != '']
                        val = list(map(float, val))
                    chart.add_xaxis(attr_scatter).add_yaxis(name, val, label_opts=opts.LabelOpts(is_show=False))
        return chart, filename



    def show_visualizations(self, number=-1):
        """
        show the charts in jupyter notebook.

        Args:
            number(int): the index of chart to be shown in jupyter notebook.
                         If number == -1, show all the charts in jupyter notebook.
            
        Returns:
            page(Page()): an object of class Page in pyecharts, containing the chart(s)
                          to be shown in jupyter notebook.
            
        """
        instance = self.instance
        if number > instance.view_num:
            print("In function show_visualizations: Error, input number greater than the view numbers.")
            return Page()
        if number != -1:
            begin = number - 1
            end = number
        else:
            begin = 0
            end = instance.view_num
        page = Page()
        for order in range(begin, end):
            
            view = instance.tables[instance.views[order].table_pos].views[instance.views[order].view_pos]
            data = {}
            data['order'] = order
            data['chartname'] = instance.table_name
            data['describe'] = view.table.describe
            data['x_name'] = view.fx.name
            data['y_name'] = view.fy.name
            data['chart'] = Chart.chart[view.chart]
            data['classify'] = [v[0] for v in view.table.classes]
            data['x_data'] = view.X
            data['y_data'] = view.Y
            data['title_top'] = 5

            margin = str(data['title_top']) + '%'

            if data['chart'] == 'bar':
                chart = (Bar().set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                    .set_global_opts(
                    title_opts=opts.TitleOpts(title=data['chartname'], subtitle=data['describe'], pos_left='center',
                                              pos_top=margin),
                    xaxis_opts=opts.AxisOpts(name=data['x_name']),
                    yaxis_opts=opts.AxisOpts(name=data['y_name'], splitline_opts=opts.SplitLineOpts(is_show=True))))
            elif data['chart'] == 'pie':
                chart = (Pie().set_global_opts(
                    title_opts=opts.TitleOpts(title=data['chartname'], subtitle=data['describe'], pos_left='center',
                                              pos_top=margin)))
            elif data['chart'] == 'line':
                chart = (Line().set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                    .set_global_opts(
                    title_opts=opts.TitleOpts(title=data['chartname'], subtitle=data['describe'], pos_left='center',
                                              pos_top=margin),
                    xaxis_opts=opts.AxisOpts(name=data['x_name']),
                    yaxis_opts=opts.AxisOpts(name=data['y_name'], splitline_opts=opts.SplitLineOpts(is_show=True))))
            elif data['chart'] == 'scatter':
                chart = (Scatter().set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                    .set_global_opts(
                    title_opts=opts.TitleOpts(title=data['chartname'], subtitle=data['describe'], pos_left='center',
                                              pos_top=margin),
                    xaxis_opts=opts.AxisOpts(type_='value', name=data['x_name'],
                                             splitline_opts=opts.SplitLineOpts(is_show=True)),
                    yaxis_opts=opts.AxisOpts(type_='value', name=data['y_name'],
                                             splitline_opts=opts.SplitLineOpts(is_show=True))))
            else:
                print("not valid chart")

            if not data["classify"]:
                attr = data["x_data"][0]
                val = data["y_data"][0]
                if data['chart'] == 'bar':
                    chart.add_xaxis(attr).add_yaxis("", val, label_opts=opts.LabelOpts(is_show=False))
                elif data['chart'] == 'line':
                    chart.add_xaxis(attr).add_yaxis("", val, label_opts=opts.LabelOpts(is_show=False))
                elif data['chart'] == 'pie':
                    chart.add("", [list(z) for z in zip(attr, val)])
                elif data['chart'] == 'scatter':
                    if isinstance(attr[0], str):
                        attr = [x for x in attr if x != '']
                        attr = list(map(float, attr))
                    if isinstance(val[0], str):
                        val = [x for x in val if x != '']
                        val = list(map(float, val))
                    chart.add_xaxis(attr).add_yaxis("", val, label_opts=opts.LabelOpts(is_show=False))
                page.add(chart)
            else:
                attr = data["x_data"][0]
                for i in range(len(data["classify"])):
                    val = data["y_data"][i]
                    name = (
                        data["classify"][i][0] if type(data["classify"][i]) == type(('a', 'b')) else data["classify"][
                            i])
                    if i == 0:
                        if data['chart'] != 'pie' and data['chart'] != 'scatter':
                            chart.add_xaxis(attr)
                    if data['chart'] == 'bar':
                        chart.add_yaxis(name, val, stack="stack1", label_opts=opts.LabelOpts(is_show=False))
                    elif data['chart'] == 'line':
                        chart.add_yaxis(name, val, label_opts=opts.LabelOpts(is_show=False))
                    elif data['chart'] == 'pie':
                        chart.add("", [list(z) for z in zip(attr, val)])
                    elif data['chart'] == 'scatter':
                        attr_scatter = data["x_data"][i]
                        if isinstance(attr_scatter[0], str):
                            attr_scatter = [x for x in attr_scatter if x != '']
                            attr_scatter = list(map(float, attr_scatter))
                        if isinstance(val[0], str):
                            val = [x for x in val if x != '']
                            val = list(map(float, val))
                        chart.add_xaxis(attr_scatter).add_yaxis(name, val, label_opts=opts.LabelOpts(is_show=False))
                page.add(chart)
        return page
