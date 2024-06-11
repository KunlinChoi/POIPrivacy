# Attack1 LocExtract

## Steps:
1. Generate training files by running "python build_graph.py"
2. Train a victim model by running "python train_target.py"
3. Select trained models (ends in .pt) in "./run" to "./models" and name it "best.pt", if it's GETNEXT, put the graph_A and graph_X files here as well
4. Put the training file into "./Training_data", this will be used to get user_id and for evaluation comparisons
5. Run the attack by "Python attack1_soft.py"
6. Analysis the extraction and see the result by running "Python result_soft.py"  The result will be printed in the format of 
   acc1,acc3,acc5, total

## Note:
This attack example used the GETNext model, please refer to the model_batch.py file for details. Replace the victim model training process if needed.