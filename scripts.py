python train.py --parametrization SubTB --parametrization.weighing equal --n_iterations 20000 --batch_size 16 --env HyperGrid --env.ndim 2 --env.height 64 --validate_with_training_examples --validation_samples 200000 --wandb test_subtb2  &> /dev/null &
python train.py --parametrization SubTB --parametrization.weighing geometric --parametrization.lamda 0.9 --n_iterations 20000 --batch_size 16 --env HyperGrid --env.ndim 2 --env.height 64 --validate_with_training_examples --validation_samples 200000 --wandb test_subtb2  &> /dev/null &
python train.py --parametrization SubTB --parametrization.weighing TB --n_iterations 20000 --batch_size 16 --env HyperGrid --env.ndim 2 --env.height 64 --validate_with_training_examples --validation_samples 200000 --wandb test_subtb2  &> /dev/null &
python train.py --parametrization SubTB --parametrization.weighing DB --n_iterations 20000 --batch_size 16 --env HyperGrid --env.ndim 2 --env.height 64 --validate_with_training_examples --validation_samples 200000 --wandb test_subtb2  &> /dev/null &
python train.py --parametrization TB --n_iterations 20000 --batch_size 16 --env HyperGrid --env.ndim 2 --env.height 64 --validate_with_training_examples --validation_samples 200000 --wandb test_subtb2  &> /dev/null &


python train.py --parametrization SubTB --parametrization.weighing geometric --parametrization.lamda 0.9 --n_iterations 20000 --batch_size 16 --env HyperGrid --env.ndim 2 --env.height 64 --validate_with_training_examples --validation_samples 200000 --wandb test_subtb2  &> /dev/null &


python train.py --parametrization SubTB --parametrization.weighing geometric --parametrization.lamda 0.9 --n_iterations 20000 --batch_size 16 --env HyperGrid --env.ndim 2 --env.height 64 --validate_with_training_examples --validation_samples 200001 --wandb test_subtb3  &> /dev/null &
python train.py --parametrization TB --n_iterations 20000 --batch_size 16 --env HyperGrid --env.ndim 2 --env.height 64 --validate_with_training_examples --validation_samples 200000 --wandb test_subtb3  &> /dev/null &
