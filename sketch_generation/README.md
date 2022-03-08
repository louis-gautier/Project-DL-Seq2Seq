Here are some instructions to run the files of our repository:

* `data_exploration.py` allows you to randomly visualize some drawings of any dataset. Modify the values of `data_type` to change the dataset name (multi-class datasets are named with the name of each class they contained separated with underscore). You can set the `part` variable to either "train" or "test".
* `drawing_completion.py` completes a drawing composed of one line. You can either draw one line thanks to the interactive tool in `interactive-drawing` and reference the automatically downloaded csv file or take a random first line from the test set by setting `first_line_csv` to `None`.
* `evaluate_precision.py` allows to compute the average loss of a model on 1000 uniformly chosen drawings of the train set and the test set. Don't forget to set `GRU` to `True` if your saved model uses GRU instead of LSTM.
* `infer_cond.py` allows to perform conditional generation on the input image called `conditioned_on` (you can change it based the image id you can get by using `data_exploration.py`).
* `infer_uncond.py` allows to perform unconditional generation.
* `interpolation.py` plots linear interpolation between two input images on the model generated on the `data_type` dataset.
* `main.py` allows to try model. The syntax for executing it is `python main.py <cond_gen> <data_type> <w_kl> <n_layers> <GRU>`. cond_gen should be set to 1 for training a conditional generation model. `GRU` should be set to 1 to use GRUs instead of LSTMs.
* `pca_exploration.py` allows to explore the latent space reduced to 2 dimensions with PCA.
* `sketch_analogies.py` allows to perform the vectorial operation $z_1+z_2-z_3$ in the latent space.
* `uncond_various_t.py` plots unconditionally generated images with different temperatures.
* `data_load.py`, `eval_skrnn.py` and `model.py` cannot be executed but are used in every other file.