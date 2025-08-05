import itertools as it
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

time_frequency = 60 * 24
chunk_size = 10

def datetime_range(start, end, delta):
    current = start
    while current <= end:
        yield current
        current += delta

def moving_window(x, length, step=1):
    streams = it.tee(x, length)
    return zip(*[it.islice(stream, i, None, step) for stream, i in zip(streams, it.count(step=step))])

def choose_target_generate_fllist_LA(sheroaks_crime):
    crime_type = 8
    neighborhood_type = 113
    start_time_so = '2018-01-01'
    end_time_so = '2018-12-31'
    format_string = '%Y-%m-%d'
    start_time_so = datetime.strptime(start_time_so, format_string)
    end_time_so = datetime.strptime(end_time_so, format_string)
    time_list_so = [dt.strftime('%Y-%m-%d') for dt in datetime_range(start_time_so, end_time_so, timedelta(minutes=time_frequency))]
    x_ = list(moving_window(time_list_so, chunk_size))
    final_list_so = []
    label_list_so = []
    for i in range(0, len(x_)):
           feature_time_frame = x_[i][:chunk_size-1]
           feature_list = []
           for index_fea in range(0, len(feature_time_frame) - 1):
               start_so = feature_time_frame[index_fea]
               end_so = feature_time_frame[index_fea + 1]
               df_so_middle = sheroaks_crime.loc[(sheroaks_crime['date_occ'] >= start_so) & (sheroaks_crime['date_occ'] < end_so)]
               crime_record = np.zeros((neighborhood_type, crime_type))
               for index, row in df_so_middle.iterrows():
                  #if int(row['neighborhood_id']) in [0, 3, 10, 15, 4, 87, 93, 110, 97, 108]:
                  crime_record[int(row["neighborhood_id"])][int(row["crime_type_id"])] = 1
               feature_list.append(crime_record)
           final_list_so.append(feature_list)

           label_time_frame = x_[i][chunk_size-2:]
           label_time_slots = sheroaks_crime.loc[(sheroaks_crime['date_occ'] >= label_time_frame[0]) & (sheroaks_crime['date_occ'] < label_time_frame[1])]
           crime_record = np.zeros((neighborhood_type, crime_type))
           for index_label, row_label in label_time_slots.iterrows():
                  #if int(row_label['neighborhood_id']) in [0, 3, 10, 15, 4, 87, 93, 110, 97, 108]:
                     crime_record[int(row_label["neighborhood_id"])][int(row_label["crime_type_id"])] = 1
           label_list_so.append(crime_record)

    print("the shape of feature list is {}, and the shape of label list is {} ".format(np.shape(final_list_so), np.shape(label_list_so)))
    return final_list_so, label_list_so


def choose_target_generate_fllist_CHI(sheroaks_crime):
    crime_type = 8
    neighborhood_type = 77
    start_time_so = '1/1/2015'
    end_time_so = '12/31/2015'
    format_string = '%m/%d/%Y'
    start_time_so = datetime.strptime(start_time_so, format_string)
    end_time_so = datetime.strptime(end_time_so, format_string)
    time_list_so = [dt.strftime('%m/%d/%Y') for dt in
                    datetime_range(start_time_so, end_time_so, timedelta(minutes=time_frequency))]
    x_ = list(moving_window(time_list_so, chunk_size))
    # Ensure date column in DataFrame is in datetime format
    sheroaks_crime['date'] = pd.to_datetime(sheroaks_crime['date'], format='%m/%d/%Y', errors='coerce')

    final_list_so = []
    label_list_so = []
    for i in range(0, len(x_)):
        feature_time_frame = x_[i][:chunk_size - 1]
        feature_list = []
        for index_fea in range(0, len(feature_time_frame) - 1):
            start_so = feature_time_frame[index_fea]
            end_so = feature_time_frame[index_fea + 1]
            df_so_middle = sheroaks_crime.loc[(sheroaks_crime['date'] >= start_so) & (sheroaks_crime['date'] < end_so)]
            crime_record = np.zeros((neighborhood_type, crime_type))
            for index, row in df_so_middle.iterrows():
                # if int(row['neighborhood_id']) in [0, 3, 10, 15, 4, 87, 93, 110, 97, 108]:
                crime_record[int(row["neighborhood_id"])][int(row["crime_type_id"])] = 1
            feature_list.append(crime_record)
        final_list_so.append(feature_list)

        label_time_frame = x_[i][chunk_size - 2:]
        label_time_slots = sheroaks_crime.loc[
            (sheroaks_crime['date'] >= label_time_frame[0]) & (sheroaks_crime['date'] < label_time_frame[1])]
        crime_record = np.zeros((neighborhood_type, crime_type))
        for index_label, row_label in label_time_slots.iterrows():
            # if int(row_label['neighborhood_id']) in [0, 3, 10, 15, 4, 87, 93, 110, 97, 108]:
            crime_record[int(row_label["neighborhood_id"])][int(row_label["crime_type_id"])] = 1
        label_list_so.append(crime_record)

    print("the shape of feature list is {}, and the shape of label list is {} ".format(np.shape(final_list_so),
                                                                                       np.shape(label_list_so)))
    return final_list_so, label_list_so