# Knowledge Distillation via Smoothed Models and Adversarial Robustness

This project came to pass as a continuation of the nemcovsky et al.(..) paper. 
The smoothed model is a non-deterministic, cumbersome model which achieved significant performance boost for perturbations.
In our project we explore the possibility of using Knowledge distillation via the smoothed model, to create a student model, that is deterministic 
and can perform on perturbed data similar to the smoothed model or even better. 



## Get smoothed model outputs data
In order to run knowledge distillation trainig we need to get the smoothed model outputs.
There are two types of aggregated outputs that can be retrived with the following command lines:

 ### Smoothe Prediction
 
 ```
run_attack.py -- --seed 42 --arch wideresnet --width 4 --layers 28 --batch-size 256 --cpni  --attack pgd --attack_k 10 --alpha 0.006 --smooth mcpredict --m_forward 512 --resume trained_models/cpni/CPNI_wide4_offd_decay_1e-3_time_2020-03-14_16-58-12/model_best.pth.tar --save results_w_csv.txt --gpus 0 
```
 ### Soft smoothe Prediction
  ```
run_attack.py -- --seed 42 --arch wideresnet --width 4 --layers 28 --batch-size 256 --cpni  --attack pgd --attack_k 10 --alpha 0.006 --smooth mcpredict --m_forward 512 --resume trained_models/cpni/CPNI_wide4_offd_decay_1e-3_time_2020-03-14_16-58-12/model_best.pth.tar --save results_w_csv.txt --gpus 0 
```

--resume variable should contain the path of the model_best.pth.tar file

## Run Knowledge distillation training
To run knowledge distillation trainig use the following command lines:
 ### Training
 
 ```
./knowledge_distillation/run.py -- --learning-rate 0.0001 --loss CrossEntropy --opt ADAM  --hist_data True --distill_weight 0.5
```
There are two types of aggregated outputs for the teacher model:
  - Smoothe Prediction
    - To use this model outputs add this variable: --hist_data true  
  - Soft smoothe Prediction
    - To use this model outputs add this variable: --soft_data true  

 
 --hist_data True...
 
 ### Adversarial training
 ```
./knowledge_distillation/run.py -- --adv_training True  --learning-rate 0.0001 -- --loss CrossEntropy --opt ADAM  --hist_data True --adv_training True --distill_weight 0.75 --perturb_distill_weight 0.25 
```


**Results on CIFAR10**

|Method | Clean accuracy| PGD-10 accuracy|
|--- |---|---|
|Smooth prediction smoothing |  88.68| 63.67|
|Smooth soft prediction smoothing | 88.53| 63.48|
|KD prediction smoothing | 79.9| 46|
|KD soft prediction smoothing | 82.13| 45.83|



