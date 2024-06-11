# Attack4 TrajMIA

## steps:
1. If targeting GETNext. Generate the graph files by running **"Python build_graph.py"** and save them to ./dataset_4sq/lira/attack_target/, also, save the index of locations to ./dataset_4sq/lira/dictionary/dict.pt in our example, use ./dataset_4sq/lira/attack_target/graph_train.csv to generate the graphs
2. Train the target model and query by running **"Python traintarget.py"**
3. Train the shadow models and query by running **"Python trainshadow.py"**
4. Show the attack result by running **"Python draw_result.py"**

## Notes
1. We have uploaded an example dataset in the repo, if you need to use your customized dataset, please follow our paper for the data preprocessing and place the shadow training dataset in ./dataset_4sq/lira/data folder
