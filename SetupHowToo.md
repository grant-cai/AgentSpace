# Environment Set Up

1. If you do not have Conda download it

2. Create a new conda environment in terminal using:
```
conda create -n <environment_name> python=3.13

```

3. Activate the conda environment using:
```
conda activate <environment_name>
```

4. 4. Download dependencies by running the following command in the same directory as `environment.yml`:
```
conda env update -f environment.yml
```