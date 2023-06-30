# Example training scripts

```bash
python train.py --env hypergrid --env.ndim 4 --env.height 8 --n_iterations 100000 --loss trajectory-balance
python train.py --env discrete-ebm --env.ndim 4 --env.alpha 0.5 --n_iterations 10000 --batch_size 64 --sampler.temperature 2.
python train.py --env hypergrid --env.ndim 2 --env.height 64 --n_iterations 100000 --loss detailed-balance --logit_PB.module_name Uniform --optim adam --optim.lr 1e-3 --batch_size 64
python train.py --env hypergrid --env.ndim 4 --env.height 8 --env.R0 0.01 --loss flowmatching --optim adam --optim.lr 1e-4
```
