# Setting up Multinode

Assuming you have installed torchgfn as normal into a conda environment:

```
cd /path/to/torchgfn
conda create -n torchgfn python=3.10
conda activate torchgfn
pip install .
```

You can modify your environment to be compatible with OneCCL using

```
bash install_multinode_dependencies
```


