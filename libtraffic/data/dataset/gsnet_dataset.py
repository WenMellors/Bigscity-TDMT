import datetime
import pickle


import numpy as np


from libtraffic.data.dataset import TrafficStateGridDataset, TrafficStateCPTDataset


class GSNetDataset(TrafficStateCPTDataset, TrafficStateGridDataset):
    def __init__(self, config):
        super(GSNetDataset, self).__init__(config)

        # NOTE: DOES NOT take into account time-of-day and day-of-week rows
        self.num_of_target_time_feature = self.config.get('num_of_target_time_feature', 0)
        self.grid_in_channel = len(self.config.get('data_col', []))
        if self.add_time_in_day:
            self.num_of_target_time_feature += 24
            self.grid_in_channel += 24
        if self.add_day_in_week:
            self.num_of_target_time_feature += 7
            self.grid_in_channel += 7

    def _load_rel(self):
        pass

    def _load_dyna(self, filename):
        # dynamic data must be 4D in this model
        return super(GSNetDataset, self)._load_grid_4d(filename)

    def _get_external_array(self, ts, ext_data=None, previous_ext=False):
        # one-hot encoding that differs from ordinary datasets
        ts_count = len(ts)
        data_list = []

        if self.add_time_in_day:
            time_indices = ((ts - ts.astype("datetime64[D]")) / np.timedelta64(1, "h")).astype("int")
            curr = np.zeros((ts_count, 24))
            # [ts_count, 24]
            curr[np.arange(0, ts_count), time_indices] = 1
            data_list.append(curr)
        if self.add_day_in_week:
            week_indices = []
            for day in ts.astype("datetime64[D]"):
                week_indices.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            curr = np.zeros((ts_count, 7))
            curr[np.arange(0, ts_count), week_indices] = 1
            data_list.append(curr)

        if ext_data is not None:
            indexs = []
            for ts_ in ts:
                if previous_ext:
                    ts_index = self.idx_of_ext_timesolts[ts_ - self.offset_frame]
                else:
                    ts_index = self.idx_of_ext_timesolts[ts_]
                indexs.append(ts_index)
            select_data = ext_data[indexs]
            data_list.append(select_data)

        if len(data_list) > 0:
            data = np.concatenate(data_list, axis=1)
        else:
            data = np.zeros((len(ts), 0))

        return data

    def get_data_feature(self):
        d = super(TrafficStateCPTDataset, self).get_data_feature()

        for k in ['risk_mask', 'road_adj', 'risk_adj', 'poi_adj', 'grid_node_map']:
            with open(self.data_path + 'tmp/' + k + '.pkl', 'rb') as f:
                d[k] = pickle.load(f)

        d['num_of_target_time_feature'] = self.num_of_target_time_feature

        lp = self.len_period * (self.pad_forward_period + self.pad_back_period + 1)
        lt = self.len_trend * (self.pad_forward_trend + self.pad_back_trend + 1)

        d['len_closeness'] = self.len_closeness
        d['len_period'] = lp
        d['len_trend'] = lt

        d['add_time_in_day'] = self.add_time_in_day
        d['add_day_in_week'] = self.add_day_in_week

        # what rows should be considered in transformation from grid input to the graph one, referred by indices
        data_col = self.config.get('data_col', [])
        for k in ['graph_input', 'target_time']:
            d[f'{k}_indices'] = []
            for n in self.config.get(f'{k}_col', []):
                # let ValueErrors raise
                d[f'{k}_indices'].append(data_col.index(n))

        d['risk_thresholds'] = self.config.get('risk_thresholds', [])
        d['risk_weights'] = self.config.get('risk_weights', [])
        for k in ['risk_thresholds', 'risk_weights']:
            d[k] = self.config.get(k, [])
            if d[k] != sorted(d[k]):
                raise ValueError(f'Dataset config item {k} is not a sorted list')
        if len(d['risk_thresholds']) != len(d['risk_weights']) - 1:
            raise ValueError('Mask loss risk thresholds must be one element shorter than risk weights')

        return d
