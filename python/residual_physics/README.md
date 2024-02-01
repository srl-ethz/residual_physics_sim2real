### Training options for resiudal physics framework

- seed [Int]: set random seed for training. default
- epochs [Int]: total training epochs. Default: 100
- learning_rate [Float]: optimizer learning rate. Default: 5e-6
- optimizer: the optimizer to use when training neural network. Only support Adam for now. You can easily add other optimizers
- start_frame [Int] :The starting timestamp we select for each trajectory in the training set
- end_frame [Int]: The last timestamp we use for each trajectory in the traininig set.
- training_set [List[Int]]: Training set trajectory index we use for training
- validate_set [List[Int]]: Validation set trajectory index. No validation will be performed if not provided.
- cuda [Int | String]: The cuda device used for training. CPU will be used if not provided.
- normalize [Bool]: Perform normalization on inputs and outputs for neural network or not. Default: False
- Scale [Int]: Scaling the training loss after each epoch. Default: 1.
- data_type [String]: To select for which dataset to use
- weight_decay [float]: Weight decay for optimizer. Default: 0.0
- validate_physics [Bool]: Perform validation by running diffpd simulations. However, this is very time consuming.
- validate_epochs [Int]: This will be set only when `validate_physics` = True, so that we only perform validation each `validate_epochs`.
- fit [String] = "forces" | "SITL": [forces]: train the network by supervised learning on residual forces. [SITL]: train the network with "Solver-in-the-loop" formulation, such that we don't need residual forces.
- model [String] = "MLP" | "skip_connection": [MLP]: vanilla MLP with layer normalization. [skip_connection]: MLP with skip connection structure.
- tolerance [Int]: After `tolerance` epochs, if the validation error is not improved, we stop training.

If we use skip connection architecture:
- num_mlp_blocks [Int]: How many MLP blocks in the network.
- hidden_size [Int]: Hidden_size in each MLP block.
- num_block_layer [Int]: The number of layers for each MLP.