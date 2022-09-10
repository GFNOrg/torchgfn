python four_kls.py  --validate_with_training_examples --ndim 2 --height 64 --R0 0.001 --batch_size 64 --n_iterations 15000  --seed 15 --mode tb --uniform_pb --wandb vi_vs_rws_vs_tb  &> /dev/null &
python four_kls.py  --validate_with_training_examples --ndim 2 --height 64 --R0 0.001 --batch_size 64 --n_iterations 15000  --seed 15 --mode forward_kl --uniform_pb --wandb vi_vs_rws_vs_tb &> /dev/null &
python four_kls.py  --validate_with_training_examples --ndim 2 --height 64 --R0 0.001 --batch_size 64 --n_iterations 15000  --seed 16 --mode tb --uniform_pb --wandb vi_vs_rws_vs_tb  &> /dev/null &
python four_kls.py  --validate_with_training_examples --ndim 2 --height 64 --R0 0.001 --batch_size 64 --n_iterations 15000  --seed 16 --mode forward_kl --uniform_pb --wandb vi_vs_rws_vs_tb &> /dev/null &
python four_kls.py  --validate_with_training_examples --ndim 2 --height 64 --R0 0.001 --batch_size 64 --n_iterations 15000  --seed 17 --mode tb --uniform_pb --wandb vi_vs_rws_vs_tb  &> /dev/null &
python four_kls.py  --validate_with_training_examples --ndim 2 --height 64 --R0 0.001 --batch_size 64 --n_iterations 15000  --seed 17 --mode forward_kl --uniform_pb --wandb vi_vs_rws_vs_tb &> /dev/null &


