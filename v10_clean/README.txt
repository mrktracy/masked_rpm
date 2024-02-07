Uses Python 3.10

"main_cln.py" contains the main training loop

"evaluate_cln.py" defines the evaluation function for validation

"models_cln.py" defines the model

"pos_embed_cln.py" defines function for getting 2-D positional encodings

"datasets_cln.py" defines dataset for training and (slightly different) dataset for validation. The difference is that in training, any panel position can be masked, whereas in validation the last panel position is always masked

"display_results_cln.py" contains extra code for displaying guesses made and saved to a results folder during training