# Attack2 TrajExtract

## Steps

1. Generate training files by running "python build_graph.py"
2. Train a victim model by running "python train_target.py"
3. Move a selected saved victim model (ends with .pt) from ./runs to ./model and name it "Best.pt", if it's GETNEXT, put the graph_A and graph_X files here as well
4. Generate ground truth file by running "python attack2_generate_gt.py"
5. Perform extraction "python attack2_extraction.py"
6. Show the result "python matchextraction.py"