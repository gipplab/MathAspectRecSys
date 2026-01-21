# Repository is for SIGIR Submission "Aspect-Aware Content-Based Recommendations for Mathematical Research Papers"

## Important step to Obtain data and Reproducible results

- Create a virtual environment

- Install requirements from requirements.txt

Now you are ready to obtain the data and reproduce results

## Loading Datasets

The steps to obtain datasets are provided in detaild 
[Data](#data) : Contains SilverRiM and GoldRiM dataset, steps on how to obtain them and required scripts.

Load the dataset using steps in "data/"

## Reproducing Results

### Figure 3
python src/data_stats/n_gram_ovelap.py

### Figure 4
python src/data_stats/cosScores_ovelap.py
(currently, this includes already generated scores, to get the actual scores, you will have to run src/eval/createIndexes.py)

### Figure 5
(for demonstration only, created using powepoint)

### Figure 6
(for demonstration only, created using powepoint)

### Table 1
assuming if you have already loaded dataframe from "data/"
python src/data_stats/print_data_stats.py

### Table 2
src/get_eval_goldRiM_silverRiM.sh

### Table 3
src/get_eval_pwc.sh

### Figure 7

### Figure 8

## License 

CC-BY-SA 4.0. This defines the license for the whole dataset, which also contains non-copyrighted bibliographic metadata and reference data derived from I4OSC (CC0).