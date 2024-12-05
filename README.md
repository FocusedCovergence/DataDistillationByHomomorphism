Main process is main_process.py

set_seed.py is for seting seed for experiment reproductivity as in the paper

add_mask.py is for adding mask for imported dataset

filter_top_features.py is for filter the top m features

reduce_features.py is for create a temparory graph dataset same as the original dataset but with filtered features (keep features with select from above)

create_homo_graph.py is for creating the distilled new 

images.py and spectral_pro.py are for drawing plots for comparison between original graph data and distilled graph data

get_top_features.py is a removed file from process, not used