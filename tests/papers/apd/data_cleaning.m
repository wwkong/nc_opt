% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

%% Global params
users_name = 'reviewerID';
items_name = 'asin';
ratings_name = 'overall';

%% Clean the relevant JSON data in the ./data folder.
[data, fname] = get_amazon_data_matrix('Patio_Lawn_and_Garden_5.json', 'patio', users_name, items_name, ratings_name);
save(fname, 'data');

[data, fname] = get_amazon_data_matrix('Musical_Instruments_5.json', 'music', users_name, items_name, ratings_name);
save(fname, 'data');

%% Helper functions.
function [mat, outname] = get_amazon_data_matrix(fname, fname_base, users_name, items_name, ratings_name)
required_fields = {users_name, items_name, ratings_name};
dt = multiline_json_to_data_table(fname, required_fields);
mat = make_recommender_matrix(dt, users_name, items_name, ratings_name);
[n, m] = size(mat);
outname = [fname_base, '_', num2str(n), 'u_', num2str(m), 'm.mat'];
end