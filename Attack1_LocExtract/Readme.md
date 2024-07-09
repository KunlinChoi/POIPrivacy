# Attack1 LocExtract

## Steps:
1. Generate training files by running ```python build_graph.py```
2. Train a victim model by running ```python train_target.py```
3. Move a selected saved victim model (ends with .pt) from ./runs to ./model and name it "Best.pt", if it's GETNEXT, put the graph_A and graph_X files to the same folder as well.
4. Put the training file into "./Training_data", this will be used to get user_id and for evaluation comparisons
5. Run the attack by ```python attack1_soft.py```
6. Analysis the extraction and see the result by running ```python result_soft.py```  The result will be printed in the format of 
   acc1,acc3,acc5, total

## Notes:
1. This attack example used the GETNext model, please refer to the model_batch.py file for details. Replace the victim model training process if needed.
2. Please refer to  [Main README](../README.md) for the required packages.

