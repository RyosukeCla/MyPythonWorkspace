import numpy as np
import pandas as pd

def to_tensor(pd_data):
    print('Hi')
    dayMin = pd_data.day.min()
    dayMax = pd_data.day.max()
    stations = pd_data.id.unique()
    column_num = len(pd_data.columns)
    
    result = np.empty((0, len(stations), 8), float)
    for day in range(dayMin, dayMax):
        mat = np.empty((0, column_num), float)
        data_on_day = pd_data[pd_data.day == day] # その日のデータのみ
        for stat_id in stations:
            # もし、あるstationのデータがあったら
            station_data = data_on_day[data_on_day.id == stat_id]
            if not station_data.empty:
                mat = np.append(mat, station_data.as_matrix(), axis = 0)
            else:
                mat = np.append(mat, np.zeros((1, column_num), float), axis = 0)
        result = np.append(result, mat.reshape(1, len(stations), column_num), axis = 0)
    return result
