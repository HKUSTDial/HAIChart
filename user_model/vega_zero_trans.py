import pickle
import pandas as pd
import random
random.seed(15)
from tools import Chart


def rand_pick(seq, probabilities):
    x = random.uniform(0, 1)
    cumprob = 0.0
    for item, item_pro in zip(seq, probabilities):
        cumprob += item_pro
        if x < cumprob:
            break
    return item


def vega_zero_trans_config(view):
    f = open('user_model/configs_dict.pkl', 'rb')
    configs_dict = pickle.load(f)

    f = open('user_model/purchase_dict.pkl', 'rb')
    purchase_dict = pickle.load(f)

    vega_type = Chart.chart[view.chart]


    config_list = configs_dict.get(vega_type)

    vega_zero_configs = []
    v_rec_list = []
    users_session_list = []
    data_type = ["none", "categorical", "numeric", "datetime"]
    for c in config_list:
        if "xsrc" in c and "ysrc" in c:
            cs = c.split('"')
            vega_x_type = data_type[view.fx.type]
            vega_y_type = data_type[view.fy.type]
            xsrc = cs[cs.index("xsrc") + 1]
            ysrc = cs[cs.index("ysrc") + 1]
            if xsrc == vega_x_type and ysrc == vega_y_type:
                cs1 = c.replace("}", "").replace('"', '').replace(':', '').split("'")
                cur_id = cs1[cs1.index("id") + 1]
                v_rec_list.append(cur_id)
                purchase_history = purchase_dict.get(cur_id)
                purchase_num = 0
                value_list = [0, 1]
                if purchase_history is not None and purchase_history >= 5:
                    purchase_num = 1
                elif purchase_history is not None and 1 <= purchase_history < 5:
                    cur_prob = purchase_history / 5.0
                    purchase_num = rand_pick(value_list, [1.0 - cur_prob, cur_prob])
                else:
                    purchase_num = rand_pick(value_list, [0.95, 0.05])

                vega_zero_configs.append(c)
                sessionDict = {'click': str(cur_id),
                               'rec_list': v_rec_list,
                               'purchase': purchase_num,
                               'dis_reward': 1.0
                               }
                users_session_list.append(sessionDict)

    for us in users_session_list:
        us['rec_list'] = v_rec_list
    # print("finish")
    return_list = [users_session_list]
    return return_list
