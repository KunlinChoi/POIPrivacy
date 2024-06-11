# Attack3 LocMIA

## steps:
1. If targeting GETNext. Generate the graph files and save them to ./dataset_4sq/lira/attack_target/, also, save the index of locations to ./dataset_4sq/lira/locationdictionary/Best.pt (we already included an example file there) in our example, use ./dataset_4sq/lira/attack_target/graph_train.csv to generate the graphs by running **"Python build_graph.py"**
2. Train the target model by running **"Python train_target.py"**, it will generate a target model: "Target.pt" in "./dataset_4sq/lira/models/4sq/". All resulting models for each epoch can be found in ./runs_4sq folder. If need to replace the target model, you can select the resulting model from ./runs_4sq and replace the "Target.pt"
3. Train the shadow models by running **"Python train_shadow.py"**
4. Query the target model by running **"Python attack3_targetmodel.py"**
5. Query the shadow models by running **"Python attack3_shadowmodel.py"**
6. Show the attack result by running **"Python draw_result.py"**

## Notes
1. We have uploaded an example dataset in the repo, if you need to use your customized dataset, please follow our paper for the data preprocessing and place the shadow training dataset in ./dataset_4sq/lira/data folder
