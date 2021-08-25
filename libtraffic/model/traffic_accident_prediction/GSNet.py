import torch
import torch.nn as nn
import torch.nn.functional as F


from libtraffic.model.abstract_traffic_state_model import AbstractTrafficStateModel


class GCNLayer(nn.Module):
    def __init__(self, num_of_features, num_of_filter):
        """
        One layer of GCN

        Arguments:
            num_of_features {int} -- the dimension of node feature
            num_of_filter {int} -- the number of graph filters
        """
        super(GCNLayer, self).__init__()
        self.gcn_layer = nn.Sequential(
            nn.Linear(in_features=num_of_features,
                      out_features=num_of_filter),
            nn.ReLU()
        )

    def forward(self, input_, adj):
        """
        Arguments:
            input {Tensor} -- signal matrix,shape (batch_size,N,T*D)
            adj {np.array} -- adjacent matrix，shape (N,N)

        Returns:
            {Tensor} -- output,shape (batch_size,N,num_of_filter)
        """
        batch_size, _, _ = input_.shape
        adj = adj.to(input_.device).repeat(batch_size, 1, 1)
        input_ = torch.bmm(adj, input_)
        output = self.gcn_layer(input_)
        return output


class STGeoModule(nn.Module):
    def __init__(self, grid_in_channel, num_of_gru_layers,
                 seq_len,
                 gru_hidden_size, num_of_target_time_feature):
        """
        Arguments:
            grid_in_channel {int} -- the number of grid data feature (batch_size,T,D,W,H),grid_in_channel=D
            num_of_gru_layers {int} -- the number of GRU layers
            seq_len {int} -- the time length of input
            gru_hidden_size {int} -- the hidden size of GRU
            num_of_target_time_feature {int} -- the number of target time feature, 24(hour)+7(week)+1(holiday)=32
        """
        super(STGeoModule, self).__init__()
        self.grid_conv = nn.Sequential(
            nn.Conv2d(in_channels=grid_in_channel, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=grid_in_channel, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.grid_gru = nn.GRU(grid_in_channel, gru_hidden_size, num_of_gru_layers, batch_first=True)

        self.grid_att_fc1 = nn.Linear(in_features=gru_hidden_size, out_features=1)
        self.grid_att_fc2 = nn.Linear(in_features=num_of_target_time_feature, out_features=seq_len)
        self.grid_att_bias = nn.Parameter(torch.zeros(1))
        self.grid_att_softmax = nn.Softmax(dim=-1)

    def forward(self, grid_input, target_time_feature):
        """
        Arguments:
            grid_input {Tensor} -- grid input，shape：(batch_size,seq_len,D,W,H)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size,num_target_time_feature)
        Returns:
            {Tensor} -- shape：(batch_size,hidden_size,W,H)
        """
        batch_size, T, D, W, H = grid_input.shape

        grid_input = grid_input.view(-1, D, W, H)
        conv_output = self.grid_conv(grid_input)

        conv_output = conv_output.view(batch_size, -1, D, W, H) \
                                 .permute(0, 3, 4, 1, 2) \
                                 .contiguous() \
                                 .view(-1, T, D)
        gru_output, _ = self.grid_gru(conv_output)

        grid_target_time = torch.unsqueeze(target_time_feature, 1).repeat(1, W*H, 1).view(batch_size*W*H, -1)
        grid_att_fc1_output = torch.squeeze(self.grid_att_fc1(gru_output))
        grid_att_fc2_output = self.grid_att_fc2(grid_target_time)
        grid_att_score = self.grid_att_softmax(F.relu(grid_att_fc1_output+grid_att_fc2_output+self.grid_att_bias))
        grid_att_score = grid_att_score.view(batch_size*W*H, -1, 1)
        grid_output = torch.sum(gru_output * grid_att_score, dim=1)

        grid_output = grid_output.view(batch_size, W, H, -1).permute(0, 3, 1, 2).contiguous()

        return grid_output


class STSemModule(nn.Module):
    def __init__(self, num_of_graph_feature, nums_of_graph_filters,
                 seq_len,
                 num_of_gru_layers, gru_hidden_size,
                 num_of_target_time_feature, north_south_map, west_east_map):
        """
        Arguments:
            num_of_graph_feature {int} -- the number of graph node feature,
                                          (batch_size,seq_len,D,N),num_of_graph_feature=D
            nums_of_graph_filters {list} -- the number of GCN output feature
            seq_len {int} -- the time length of input
            num_of_gru_layers {int} -- the number of GRU layers
            gru_hidden_size {int} -- the hidden size of GRU
            num_of_target_time_feature {int} -- the number of target time feature, 24(hour)+7(week)+1(holiday)=32
            north_south_map {int} -- the weight of grid data
            west_east_map {int} -- the height of grid data

        """
        super(STSemModule, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.road_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.road_gcn.append(GCNLayer(num_of_graph_feature, num_of_filter))
            else:
                self.road_gcn.append(GCNLayer(nums_of_graph_filters[idx-1], num_of_filter))

        self.risk_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.risk_gcn.append(GCNLayer(num_of_graph_feature, num_of_filter))
            else:
                self.risk_gcn.append(GCNLayer(nums_of_graph_filters[idx-1], num_of_filter))

        self.poi_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.poi_gcn.append(GCNLayer(num_of_graph_feature, num_of_filter))
            else:
                self.poi_gcn.append(GCNLayer(nums_of_graph_filters[idx-1], num_of_filter))

        self.graph_gru = nn.GRU(num_of_filter, gru_hidden_size, num_of_gru_layers, batch_first=True)

        self.graph_att_fc1 = nn.Linear(in_features=gru_hidden_size, out_features=1)
        self.graph_att_fc2 = nn.Linear(in_features=num_of_target_time_feature, out_features=seq_len)
        self.graph_att_bias = nn.Parameter(torch.zeros(1))
        self.graph_att_softmax = nn.Softmax(dim=-1)

    def forward(self, graph_feature,
                road_adj, risk_adj, poi_adj,
                target_time_feature, grid_node_map):
        """
        Arguments:
            graph_feature {Tensor} -- Graph signal matrix，(batch_size,T,D1,N)
            road_adj {np.array} -- road adjacent matrix，shape：(N,N)
            risk_adj {np.array} -- risk adjacent matrix，shape：(N,N)
            poi_adj {np.array} -- poi adjacent matrix，shape：(N,N)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size,num_target_time_feature)
            grid_node_map {np.array} -- map graph data to grid data,shape (W*H,N)
        Returns:
            {Tensor} -- shape：(batch_size,pre_len,north_south_map,west_east_map)
        """
        batch_size, T, D1, N = graph_feature.shape

        road_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
        for gcn_layer in self.road_gcn:
            road_graph_output = gcn_layer(road_graph_output, road_adj)

        risk_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
        for gcn_layer in self.risk_gcn:
            risk_graph_output = gcn_layer(risk_graph_output, risk_adj)

        graph_output = road_graph_output + risk_graph_output

        if poi_adj is not None:
            poi_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
            for gcn_layer in self.poi_gcn:
                poi_graph_output = gcn_layer(poi_graph_output, poi_adj)
            graph_output += poi_graph_output

        graph_output = graph_output.view(batch_size, T, N, -1) \
            .permute(0, 2, 1, 3) \
            .contiguous() \
            .view(batch_size*N, T, -1)
        graph_output, _ = self.graph_gru(graph_output)

        graph_target_time = torch.unsqueeze(target_time_feature, 1).repeat(1, N, 1).view(batch_size*N, -1)
        graph_att_fc1_output = torch.squeeze(self.graph_att_fc1(graph_output))
        graph_att_fc2_output = self.graph_att_fc2(graph_target_time)
        graph_att_score = self.graph_att_softmax(F.relu(graph_att_fc1_output+graph_att_fc2_output+self.graph_att_bias))
        graph_att_score = graph_att_score.view(batch_size*N, -1, 1)
        graph_output = torch.sum(graph_output * graph_att_score, dim=1)
        graph_output = graph_output.view(batch_size, N, -1).contiguous()

        grid_node_map_tmp = grid_node_map \
            .to(graph_feature.device) \
            .repeat(batch_size, 1, 1)
        graph_output = torch.bmm(grid_node_map_tmp, graph_output) \
            .permute(0, 2, 1) \
            .view(batch_size, -1, self.north_south_map, self.west_east_map)
        return graph_output


class _GSNet(nn.Module):
    def __init__(self,
                 grid_in_channel, num_of_gru_layers,
                 seq_len, pre_len,
                 gru_hidden_size,
                 num_of_target_time_feature, num_of_graph_feature, nums_of_graph_filters,
                 north_south_map, west_east_map):
        """
        GSNet main module.

        Arguments:
            grid_in_channel {int} -- the number of grid data feature (batch_size,T,D,W,H),grid_in_channel=D
            num_of_gru_layers {int} -- the number of GRU layers
            seq_len {int} -- the time length of input
            pre_len {int} -- the time length of prediction
            gru_hidden_size {int} -- the hidden size of GRU
            num_of_target_time_feature {int} -- the number of target time feature，为24(hour)+7(week)+1(holiday)=32
            num_of_graph_feature {int} -- the number of graph node feature，(batch_size,seq_len,D,N),
                                          num_of_graph_feature=D
            nums_of_graph_filters {list} -- the number of GCN output feature
            north_south_map {int} -- the weight of grid data
            west_east_map {int} -- the height of grid data
        """

        super(_GSNet, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map

        self.st_geo_module = STGeoModule(grid_in_channel, num_of_gru_layers,
                                         seq_len,
                                         gru_hidden_size, num_of_target_time_feature)

        self.st_sem_module = STSemModule(num_of_graph_feature, nums_of_graph_filters,
                                         seq_len,
                                         num_of_gru_layers, gru_hidden_size,
                                         num_of_target_time_feature,
                                         north_south_map, west_east_map)

        fusion_channel = 16
        self.grid_weight = nn.Conv2d(in_channels=gru_hidden_size, out_channels=fusion_channel, kernel_size=1)
        self.graph_weight = nn.Conv2d(in_channels=gru_hidden_size, out_channels=fusion_channel, kernel_size=1)
        self.output_layer = nn.Linear(fusion_channel*north_south_map*west_east_map,
                                      pre_len*north_south_map*west_east_map)

    def forward(self,
                grid_input,
                target_time_feature, graph_feature,
                road_adj, risk_adj, poi_adj,
                grid_node_map):
        """
        Arguments:
            grid_input {Tensor} -- grid input，shape：(batch_size,T,D,W,H)
            graph_feature {Tensor} -- Graph signal matrix，(batch_size,T,D1,N)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size,num_target_time_feature)
            road_adj {np.array} -- road adjacent matrix，shape：(N,N)
            risk_adj {np.array} -- risk adjacent matrix，shape：(N,N)
            poi_adj {np.array} -- poi adjacent matrix，shape：(N,N)
            grid_node_map {np.array} -- map graph data to grid data,shape (W*H,N)

        Returns:
            {Tensor} -- shape：(batch_size,pre_len,north_south_map,west_east_map)
        """
        batch_size, _, _, _, _ = grid_input.shape

        grid_output = self.st_geo_module(grid_input, target_time_feature)
        graph_output = self.st_sem_module(graph_feature,
                                          road_adj, risk_adj, poi_adj,
                                          target_time_feature, grid_node_map)

        grid_output = self.grid_weight(grid_output)
        graph_output = self.graph_weight(graph_output)
        fusion_output = (grid_output + graph_output).view(batch_size, -1)
        final_output = self.output_layer(fusion_output) \
                           .view(batch_size, -1, self.north_south_map, self.west_east_map)
        return final_output


class GSNet(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super(GSNet, self).__init__(config, data_feature)
        self.device = config.get('device', 'cpu')

        self.graph_input_indices = data_feature.get('graph_input_indices', [])
        self.grid_in_channel = data_feature.get('feature_dim', 0)
        self.target_time_indices = data_feature.get('target_time_indices', [])
        # currently examined feature dimension index
        curr_idx = data_feature.get('feature_dim', 0)
        # always in the beginning of external dimensions
        if data_feature.get('add_time_in_day', False):
            self.target_time_indices.extend(range(curr_idx, curr_idx+24))
            curr_idx += 24
            self.grid_in_channel += 24
        # always right after dimensions on time of day
        if data_feature.get('add_day_in_week', False):
            self.target_time_indices.extend(range(curr_idx, curr_idx+7))
            curr_idx += 7
            self.grid_in_channel += 7

        self.num_of_gru_layers = config.get('num_of_gru_layers', 5)
        self.seq_len = config.get(
                'seq_len',
                data_feature.get('len_closeness', 0) +
                data_feature.get('len_period', 0) +
                data_feature.get('len_trend', 0)
        )
        self.pre_len = config.get('pred_len', 1)
        self.gru_hidden_size = config.get('gru_hidden_size', 256)
        self.num_of_target_time_feature = data_feature.get('num_of_target_time_feature', 0)
        self.num_of_graph_feature = len(self.graph_input_indices)
        self.nums_of_graph_filters = config.get('gcn_nums_filters', 64)
        self.north_south_map = data_feature.get('len_column', 20)  # N-S/W-E grid count
        self.west_east_map = data_feature.get('len_row', 20)

        self.risk_mask = data_feature.get('risk_mask', torch.Tensor(size=(self.north_south_map, self.west_east_map)))
        self.road_adj = data_feature.get('road_adj', torch.Tensor(size=(self.north_south_map, self.west_east_map)))
        self.risk_adj = data_feature.get('risk_adj', torch.Tensor(size=(self.north_south_map, self.west_east_map)))
        self.poi_adj = data_feature.get('poi_adj', torch.Tensor(size=(self.north_south_map, self.west_east_map)))
        self.grid_node_map = data_feature.get('grid_node_map',
                                              torch.Tensor(size=(self.north_south_map, self.west_east_map)))

        self.dtype = config.get('dtype', torch.float32)
        self.risk_mask = torch.from_numpy(self.risk_mask).to(device=self.device, dtype=self.dtype)
        self.road_adj = torch.from_numpy(self.road_adj).to(device=self.device, dtype=self.dtype)
        self.risk_adj = torch.from_numpy(self.risk_adj).to(device=self.device, dtype=self.dtype)
        self.poi_adj = torch.from_numpy(self.poi_adj).to(device=self.device, dtype=self.dtype)
        self.grid_node_map = torch.from_numpy(self.grid_node_map).to(device=self.device, dtype=self.dtype)

        self.risk_mask.requires_grad = False
        self.road_adj.requires_grad = False
        self.risk_adj.requires_grad = False
        self.poi_adj.requires_grad = False
        self.grid_node_map.requires_grad = False

        self.risk_thresholds = data_feature.get('risk_thresholds', [])
        self.risk_weights = data_feature.get('risk_thresholds', [])

        self.gsnet = _GSNet(
                grid_in_channel=self.grid_in_channel,
                seq_len=self.seq_len, pre_len=self.pre_len,
                num_of_gru_layers=self.num_of_gru_layers, gru_hidden_size=self.gru_hidden_size,
                nums_of_graph_filters=self.nums_of_graph_filters, num_of_graph_feature=self.num_of_graph_feature,
                num_of_target_time_feature=self.num_of_target_time_feature,
                north_south_map=self.north_south_map, west_east_map=self.west_east_map)

    def forward(self, batch):
        batch_size = batch['X'].shape[0]

        # [batch_size, seq_len, input_dim, num_cols, num_rows]
        grid_input = torch.cat([
            # [batch_size, seq_len, num_rols, num_cols, ...] ->
            # [batch_size, seq_len, ..., num_cols, num_rows]
            batch['X'].permute(0, 1, 4, 3, 2),
            # [batch_size, seq_len, ext_dim] ->
            # [batch_size, seq_len, ext_dim, num_cols, num_rows]
            batch['X_ext'].unsqueeze(-1).unsqueeze(-1) \
                          .repeat(1, 1, 1, self.west_east_map, self.north_south_map)
                     ], dim=2)

        # [batch_size, seq_len, input_dim, num_cols, num_rows] ->
        # [batch_size, seq_len, len(graph_input_indices), num_cols*num_rows] ->
        # [batch_size, seq_len, len(graph_input_indices), num_graph_nodes]
        graph_input = grid_input[:, :, self.graph_input_indices, :, :] \
            .reshape(batch_size,
                     self.seq_len,
                     len(self.graph_input_indices),
                     self.west_east_map*self.north_south_map) \
            .matmul(self.grid_node_map)

        # time features are supposed to be only dependent on time and indicate current time slot only
        # [batch_size, len(target_time_indices)]
        target_time_feature = grid_input[:, 0, self.target_time_indices, 0, 0]

        # [batch_size, pred_len, num_cols, num_rows]
        result = self.gsnet.forward(
                 grid_input=grid_input,
                 target_time_feature=target_time_feature,
                 graph_feature=graph_input,
                 road_adj=self.road_adj,
                 risk_adj=self.risk_adj,
                 poi_adj=self.poi_adj,
                 grid_node_map=self.grid_node_map
        )

        # [batch_size, pred_len, num_cols, num_rows] ->
        # [batch_size, num_rows, num_cols, pred_len] ->
        # [batch_size, output_dim, num_rows, num_cols, pred_len]
        return result.permute(0, 3, 2, 1).unsqueeze(1).contiguous()

    def calculate_loss(self, batch):
        # [batch_size, output_dim, num_cols, num_rows, pred_len]
        y_pred = self.forward(batch)
        # [batch_size, pred_len, num_rows, num_cols, feature_dim] ->
        # [batch_size, pred_len, num_rows, num_cols, output_dim] ->
        # [batch_size, output_dim, num_rows, num_cols, pred_len]
        y_true = batch['y'][..., :1].permute(0, 4, 2, 3, 1)

        risk_mask = self.risk_mask / self.risk_mask.mean()
        # [batch_size, output_dim, num_cols, num_rows, pred_len]
        loss = (y_true - y_pred).mul(risk_mask).pow(2)
        weight = torch.zeros(y_true.shape).to(self.device)
        for i in range(len(self.risk_thresholds)):
            if i == 0:
                weight[y_pred < self.risk_thresholds[i]] = self.risk_weights[i]
            else:
                weight[(y_pred > self.risk_thresholds[i-1]) &
                       (y_pred <= self.risk_thresholds[i])] = self.risk_weights[i]

        return loss.mul(weight).mean()
