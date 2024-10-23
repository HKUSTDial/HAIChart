from datetime import datetime
import json
import math
import os
import re
from flask import Flask, jsonify, request, render_template, send_from_directory
import time

import pandas as pd
import tools
from mcgs import MCGS, Node
from tools import Type

app = Flask(__name__)

constraints = []

UPLOAD_FOLDER = 'datasets'
HTML_FOLDER = 'html'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HTML_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')


constraints = {
    '[T]': [],
    '[X]': [],
    '[Y]': [],
    '[AggFunction]': [],
    '[G]': [],
    '[TransForm]': [],
    '[B]': []
}

haichart = tools.haichart("mcgs")
mcgs = MCGS()
history_score = {}  
dict_sorted = []
current_view = []
good_view = {}
curr_filename = ""
curr_hints = []



def generate_suggestions(top_guidance):
    suggestions = []

    for category, guidances in top_guidance.items():
        for guidance, info in guidances:
            guidance_key = guidance.split("_")[0]
            guidance_label = guidance.split("_")[1]

            # ['bar', 'line', 'point', 'arc', 'heatmap', 'box']
            suggestion_text = ""
            if guidance_key == '[T]':
                if guidance_label.lower() == "bar":
                    suggestion_text = f"Explore distributions with a 'bar' chart."
                elif guidance_label.lower() == "arc":
                    suggestion_text = f"View proportions in a 'pie' chart."
                elif guidance_label.lower() == "point":
                    suggestion_text = f"Discover patterns with a 'scatter' plot."
                else:
                    suggestion_text = f"Try a '{guidance_label.lower()}' chart for a new view."
            elif guidance_key == '[X]':
                    # suggestion_text = f"Explore '{guidance_label}' trends or distribution over categories or time."
                    suggestion_text = f"Explore '{guidance_label}' across categories or time."
            elif guidance_key == '[Y]':
                    suggestion_text = f"Compare '{guidance_label}' to different categories."
            elif guidance_key == '[AggFunction]':
                suggestion_text = f"Summarize data using '{guidance_label}'."
            elif guidance_key == '[G]':
                suggestion_text = f"Break down the data by '{guidance_label}'."
            elif guidance_key == '[TransForm]':
                if guidance_label.lower() == "true":
                    suggestion_text = f"Enable data transformations to potentially reveal hidden patterns."
                elif guidance_label.lower() == "false":
                    suggestion_text = f"Disable data transformations for a more direct interpretation of your data."
            elif guidance_key == '[B]':
                suggestion_text = f"Bin data based on '{guidance_label}'."

            suggestions.append((guidance, suggestion_text))

    return suggestions

def generate_suggestions_by_hints(top_guidance):
    suggestions = []

    for guidance in top_guidance:
        guidance_key = guidance.split("_")[0]
        guidance_label = guidance.split("_")[1]

        # ['bar', 'line', 'point', 'arc', 'heatmap', 'box']
        suggestion_text = ""
        if guidance_key == '[T]':
            if guidance_label.lower() == "bar":
                suggestion_text = f"Explore distributions with a 'bar' chart."
            elif guidance_label.lower() == "arc":
                suggestion_text = f"View proportions in a 'pie' chart."
            elif guidance_label.lower() == "point":
                suggestion_text = f"Discover patterns with a 'scatter' plot."
            else:
                suggestion_text = f"Try a '{guidance_label.lower()}' chart for a new view."
        elif guidance_key == '[X]':
                # suggestion_text = f"Explore '{guidance_label}' trends or distribution over categories or time."
                suggestion_text = f"Explore '{guidance_label}' across categories or time."
        elif guidance_key == '[Y]':
                suggestion_text = f"Compare '{guidance_label}' to different categories."
        elif guidance_key == '[AggFunction]':
            suggestion_text = f"Summarize data using '{guidance_label}'."
        elif guidance_key == '[G]':
            suggestion_text = f"Break down the data by '{guidance_label}'."
        elif guidance_key == '[TransForm]':
            if guidance_label.lower() == "true":
                suggestion_text = f"Enable data transformations to potentially reveal hidden patterns."
            elif guidance_label.lower() == "false":
                suggestion_text = f"Disable data transformations for a more direct interpretation of your data."
        elif guidance_key == '[B]':
            suggestion_text = f"Bin data based on '{guidance_label}'."

        suggestions.append((guidance, suggestion_text))

    return suggestions


def mcgs_according_to_constraints():
    global curr_filename,eh,dict_sorted,curr_hints 
    # current_view.append(formatting_query(new_query))
    current_view = []
    mcgs.start_exploring(history_score, good_view, haichart, constraints,
                         current_view)
    
    
    if all(len(lst) == 0 for lst in constraints.values()):
        dict_sorted = sorted(good_view.items(),
                                              key=lambda good_view: good_view[1].score_l, reverse=True)
    else:
        def filter_queries(constraints, queries):
            filtered_queries = []

            for query in queries:
                query_tokens = query.split()
                
                if constraints.get('[T]') and (query_tokens[query_tokens.index('M') + 1] not in constraints['[T]']):
                    continue
                if constraints.get('[X]') and (query_tokens[query_tokens.index('E') + 2] not in constraints['[X]']):
                    continue
                if constraints.get('[Y]') and (query_tokens[query_tokens.index('A') + 2] not in constraints['[Y]']):
                    continue
                if constraints.get('[AggFunction]') and (query_tokens[query_tokens.index('A') + 1] not in constraints['[AggFunction]']):
                    continue
                see_group = constraints.get('[G]')
                if see_group:
                    query_groups = query_tokens[query_tokens.index('G') + 1].split(",")
                    if not any(q_g in see_group for q_g in query_groups):
                        continue  

                if constraints.get('[B]') and (query_tokens[query_tokens.index('B') + 3] not in constraints['[B]']):
                    continue

                filtered_queries.append(query)

            return filtered_queries

        matched_keys = []

        matched_keys = filter_queries(constraints, good_view.keys())

        try:
            current_results = {key: good_view[key] for key in set(matched_keys + current_view)}
        except Exception as e:
            print(f"error: {e}-------------")


        dict_sorted = sorted(current_results.items(), key=lambda item: item[1].score_l, reverse=True)



    t_name = curr_filename.split('.')[0]
    haichart.to_single_html(dict_sorted, t_name)


    def convert_node_identifier(node_identifier):
        key, value = node_identifier.split("_", 1)

        if key == '[T]':
            if value.lower() == 'arc':
                return "chart: pie"
            elif value.lower() == 'point':
                return "chart: scatter"
            else:
                return f"chart: {value}"
        elif key == '[X]':
            return f"x_name: {value}"
        elif key == '[Y]':
            return f"y_name: {value}"
        elif key == '[AggFunction]':
            agg_mapping = {"count": "cnt", "sum": "sum", "average": "avg", "none": ""}
            agg = agg_mapping.get(value, "")
            return f"{agg}("
        elif key == '[G]':
            return f"group by {value}"
        elif key == '[B]':
            return f"by {value}"
        else:
            return f"unknown_key: {value}"


    def find_y_name_pattern(text, target):
        pattern = rf"y_name:\s*((cnt|sum|avg)\({target}\)|{target})"
        
        match = re.search(pattern, text)
        
        return bool(match)

    new_idf_data = {}
    for node_identifier, node in mcgs.root.global_node_storage.items():
        if "none" in node_identifier or '[transform]' in node_identifier.lower() or "*!STOP!*" in node_identifier.lower():
            continue
        
        category, item = node_identifier.split('_')

        if item in constraints[category]:
            print(f"{node_identifier} It's already in constraints")
            continue


        new_idf = convert_node_identifier(node_identifier).lower()
        view_count = 0
        score_sum = 0
        for key,view in good_view.items():
            key = view.describe
            if "y_name" in new_idf:
                if not find_y_name_pattern(key,new_idf.split(" ")[-1]):
                    continue
            else:
                if new_idf not in key:
                    continue
            
            if node_identifier not in new_idf_data:
                new_idf_data[node_identifier] = {'view_count': 0, 'score_sum': 0, 'views': []}

            new_idf_data[node_identifier]['view_count'] += 1
            new_idf_data[node_identifier]['score_sum'] += view.score  # 假设view.score是一个数值
            new_idf_data[node_identifier]['views'].append(view)

    

    view_frequencies = {}
    for node_identifier, node_data in new_idf_data.items():
        for view in node_data['views']:
            view_key = view.describe  
            if view_key not in view_frequencies:
                view_frequencies[view_key] = 0
            view_frequencies[view_key] += 1

    total_hints = len(new_idf_data)
    idf_values = {key: math.log(total_hints / freq) for key, freq in view_frequencies.items()}

    for node_identifier, node_data in new_idf_data.items():
        adjusted_score_sum = 0
        for view in node_data['views']:
            view_key = view.describe
            idf_value = idf_values[view_key]
            adjusted_score_sum += view.score * idf_value
        node_data['adjusted_score_sum'] = adjusted_score_sum


    # Budget maximum coverage algorithm
    def select_k_hints(hints, budget, k):
        """
        Selects k hints to maximize the total score while keeping the total number of charts under the budget.

        :param hints: Dictionary of hints with their chart count and total score.
        :param budget: Maximum number of charts allowed.
        :param k: Number of hints to select.
        :return: List of selected hints.
        """
        global curr_hints
        # Filter hints that exceed the budget
        valid_hints = [(hint, details) for hint, details in hints.items() if details['view_count'] <= budget]

        # Sort hints based on score-to-chart ratio and total score
        sorted_hints = sorted(valid_hints, key=lambda x: (x[1]['adjusted_score_sum'] / x[1]['view_count'], x[1]['adjusted_score_sum']), reverse=True)
        curr_hints = sorted_hints
        

        # Select top k hints based on the sorted order while considering the budget
        selected_hints = []
        field_name_counts = {}
        total_charts = 0
        for hint, details in sorted_hints:
            
            # Extract the field name from the hint
            field_name = hint.split(']_')[1]
            
            # Skip the hint if the field name has been selected twice
            if field_name_counts.get(field_name, 0) >= 2:
                continue

            if len(selected_hints) < k and total_charts + details['view_count'] <= budget:
                selected_hints.append(hint)
                total_charts += details['view_count']
                # Update the count for the selected field name
                field_name_counts[field_name] = field_name_counts.get(field_name, 0) + 1


        return selected_hints

    budget = 200  # Maximum number of charts
    k = 9       # Number of hints to select

    selected_hints = select_k_hints(new_idf_data, budget, k)
    

    suggestions = generate_suggestions_by_hints(selected_hints)

    return suggestions


import time
import random

logname = 'mutl-test.txt'

def record_event(log_name,event_type, payload):
    timestamp = int(time.time() * 1000)  
    iso_time = datetime.fromtimestamp(timestamp / 1000).isoformat() + 'Z'  

    payload = payload.replace("'",'"').replace('"','""')

    record = f'{timestamp},{iso_time},{event_type},"{payload}"\n'

    # if not os.path.isfile(log_name):
    #     with open(log_name, 'w') as file:
    #         file.write('timestamp,ISOString,type,payload\n')  

    # with open(log_name, 'a') as file:
    #     file.write(record)

user_log_name = ""

@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    global curr_filename,constraints,history_score,mcgs,dict_sorted,current_view,good_view,haichart,user_log_name  # 声明 curr_filename 为全局变量

    mcgs = MCGS()
    Node.global_node_storage = {}
    good_view = {}
    history_score = {}
    curr_filename = ""

    constraints = {
        '[T]': [],
        '[X]': [],
        '[Y]': [],
        '[AggFunction]': [],
        '[G]': [],
        '[TransForm]': [],
        '[B]': []
    }

    file = request.files.get('file')

    if file:
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)  
        # data cleaning
        # df = pd.read_csv(file_path)

        # df = df.dropna()

        # df.to_csv(file_path, index=False)

        # if not os.path.exists(file_path):
    else:
        data = request.get_json()
        dataset_name = data['datasetName']
        filename = dataset_name + ".csv"
        file_path = os.path.join("example_datasets", filename)

    start_time = time.time()  # 记录开始时间
    start_time_formatted = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')  

    # user_log_name = f"multi_turn_log/user_log_{start_time}_{filename.split('.')[0]}.csv"  
    
    # record_event(user_log_name,'DATASET_REQUEST', json.dumps({"name": filename.split('.')[0]}))

    # with open(logname, 'a') as file:
    #     file.write(f"-----------time{start_time_formatted}---dataset{file_path}---------------------\n")


    df = pd.read_csv(file_path)
    sample_data = df.head(10).values.tolist() 


    haichart.from_csv(file_path)
    print(file_path)
    haichart.learning_to_rank()
    haichart.eh_view = haichart.output_list("list")
    curr_filename = filename

    suggestions = mcgs_according_to_constraints()

    # querys = [item[0] for item in dict_sorted]  
    # record_event(user_log_name,'RESULT_RECEIVE', json.dumps(querys))
    # record_event(user_log_name,'RESULT_HINTS', json.dumps(suggestions))

    columns_info = [{'name': name, 'type': dtype} for name, dtype in zip(haichart.column_names, haichart.column_types)]

    # print(columns_info)

    visualization_html = curr_filename.split('.')[0] + '_all.html'  
    recommendations = suggestions
    
    return jsonify({
        'fileName': filename,
        'constraints': constraints,
        'visualization': os.path.join(HTML_FOLDER, visualization_html),
        'recommendations': recommendations,
        'columnsInfo': columns_info,  
        'sampleData': sample_data,  
    })

@app.route('/html/<path:filename>')
def serve_static_html(filename):
    return send_from_directory(HTML_FOLDER, filename)



@app.route('/update-constraints', methods=['POST'])
def update_constraints():
    global curr_filename,constraints,user_log_name,good_view,curr_hints  
    data = request.get_json()

    fieldName = data.get('fieldName')
    user_selected_hints = []
    if fieldName:
        # print(f"user click fieldName: {fieldName}" )
        # print(curr_hints)
        for hint_key, _ in curr_hints:
            if fieldName in hint_key:
                action = hint_key
                user_selected_hints.append(hint_key)
                user_value = generate_suggestions_by_hints(user_selected_hints)
                break
    else:
        action = data.get('key')
    
    c_key = action.split('_')[0]
    c_value = action.split('_')[1]
    
    # with open(logname, 'a') as file:
    #     file.write(f"-------------------The user selected the constraint{action}------------------------------------\n")

    # constraints = {
    #     '[T]': [],
    #     '[X]': [],
    #     '[Y]': ['destcity'],
    #     '[AggFunction]': [],
    #     '[G]': [],
    #     '[TransForm]': [],
    #     '[B]': []
    # }

    
    test_vail = False
    if constraints[c_key] == [] and c_value not in constraints[c_key]:
        constraints[c_key].append(c_value)
    
        for key,view in good_view.items():
            querys = key.split(' ')
            chart_type = querys[querys.index("M")+1]
            x_axis = querys[querys.index("E")+2]
            method = querys[querys.index("A")+1] 
            y_axis = querys[querys.index("A") + 2]
            group_by_who = querys[querys.index("G") + 1]
            bin_by_who = querys[querys.index("B") + 3]

            groups = group_by_who.split(",")

            is_valid_view = True  
            for constraint_key, constraint_values in constraints.items():
                if constraint_values:  
                    if constraint_key == '[T]':  
                        if chart_type not in constraint_values:
                            is_valid_view = False
                            continue  
                    elif constraint_key == '[X]':
                        if x_axis not in constraint_values:
                            is_valid_view = False
                            continue
                    elif constraint_key == '[Y]':
                        if y_axis not in constraint_values:
                            is_valid_view = False
                            continue
                    elif constraint_key == '[AggFunction]':
                        if method not in constraint_values:
                            is_valid_view = False
                            continue
                    elif constraint_key == '[G]':
                        if len(groups) == 2:
                            if groups[0] not in constraint_values and groups[1] not in constraint_values:
                                is_valid_view = False
                                continue
                        else:
                            if group_by_who not in constraint_values:
                                is_valid_view = False
                                continue
                    elif constraint_key == '[B]':
                        if bin_by_who not in constraint_values:
                            is_valid_view = False
                            continue
            if is_valid_view:
                test_vail = True
                break
        
    if test_vail == False:
        # constraints[c_key].remove(c_value)
        constraints['[T]'] = []
        constraints['[X]'] = []
        constraints['[Y]'] = []
        constraints['[AggFunction]'] = []
        constraints['[G]'] = []
        constraints['[B]'] = []
        constraints[c_key].append(c_value)

    suggestions = mcgs_according_to_constraints()

    # record_event(user_log_name,'SPEC_ACTION', json.dumps({"action": f"ADD_{action}"}))
    # querys = [item[0] for item in dict_sorted]  #
    # record_event(user_log_name,'RESULT_RECEIVE', json.dumps(querys))
    # record_event(user_log_name,'RESULT_HINTS', json.dumps(suggestions))



    visualization_html = curr_filename.split('.')[0] + '_all.html' 
    recommendations = suggestions


    if fieldName:
        return jsonify({
            'hintKey':hint_key,
            'value': user_value[0][1].replace("'",""),
            'constraints': constraints,
            'visualization': os.path.join(HTML_FOLDER, visualization_html),
            'recommendations': recommendations
        })

    else:
        return jsonify({
            'constraints': constraints,
            'visualization': os.path.join(HTML_FOLDER, visualization_html),
            'recommendations': recommendations
        })

@app.route('/remove-constraint', methods=['POST'])
def remove_constraint():
    data = request.get_json()
    constraint_key = data.get('key')  
    constraint_value = data.get('value')  

    # with open(logname, 'a') as file:
    #     file.write(f"-------------------The user deleted the constraint{data}------------------------------------\n")
     

    if constraint_key in constraints:
        if constraint_value in constraints[constraint_key]:
            constraints[constraint_key].remove(constraint_value)  

    suggestions = mcgs_according_to_constraints()  
    visualization_html = curr_filename.split('.')[0] + '_all.html'  
    recommendations = suggestions

    # record_event(user_log_name,'SPEC_ACTION', json.dumps({"action": f"DEL_{data}"}))
    # querys = [item[0] for item in dict_sorted]  # 获取dict_sorted中的第一个查询键
    # record_event(user_log_name,'RESULT_RECEIVE', json.dumps(querys))
    # record_event(user_log_name,'RESULT_HINTS', json.dumps(suggestions))

    return jsonify({
        'constraints': constraints,
        'visualization': os.path.join(HTML_FOLDER, visualization_html),
        'recommendations': recommendations
    })



if __name__ == '__main__':
    
    test_model = True

    if test_model :
        timestamp = int(time.time())
        random.seed(timestamp)

        random_number = random.randint(1000, 9999)
        print(random_number)
        app.run(debug=False,port=random_number)
    else:
        app.run(host='0.0.0.0', debug=False, port=8080)

    
