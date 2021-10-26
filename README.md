# FastDeepNAS
Git repository for code used to produce the results reported in our paper "Fast Deep Neural Architecture Search for Wearable Activity Recognition by Early Prediction of Converged Performance (https://dl.acm.org/doi/10.1145/3460421.3478813)

To run the code in the notebooks or main.py, first install the requirements using pip:

    pip install -r requirements.txt
        
then, you will also need to install the custom OpenAI gym environment gym_nas_pt as follows:

    pip install -e ./gym_nas_pt_env
        
both of these commands should be run from the root of the cloned repo.

The directories MLP* and Baseline* contain notebooks to replicate the NAS runs reported in the paper, as well as the intermediate results from our experiments.
