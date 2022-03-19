# Knowledge Distillation via Smoothed Models and Adversarial Robustness

This project came to pass as a continuation of the nemcovsky et al.(..) paper. 
The smoothed model is a non-deterministic, cumbersome model which achieved significant performance boost for perturbations.
In our project we explore the possibility of using Knowledge distillation via the smoothed model, to create a student model, that is deterministic 
and can perform on perturbed data similar to the smoothed model or even better. 



## Get smoothed model outputs data
In order to run knowledge distillation trainig we need to get the smoothed model outputs.
There are two types of aggregated outputs that can be retrived with the following command lines:

 ### Smoothed Prediction
 
 ```
ipython run_attack.py -- --seed 42 --arch wideresnet --width 4 --layers 28 --batch-size 256 --cpni  --attack pgd --attack_k 10 --alpha 0.006 --smooth mcpredict --m_forward 512 --resume trained_models/cpni/CPNI_wide4_offd_decay_1e-3_time_2020-03-14_16-58-12/model_best.pth.tar --save results_w_csv.txt --gpus 0 
```
 ### Soft smoothed Prediction
  ```
ipython run_attack.py -- --seed 42 --arch wideresnet --width 4 --layers 28 --batch-size 256 --cpni  --attack pgd --attack_k 10 --alpha 0.006 --smooth mcpredict --m_forward 512 --resume trained_models/cpni/CPNI_wide4_offd_decay_1e-3_time_2020-03-14_16-58-12/model_best.pth.tar --save results_w_csv.txt --gpus 0 
```

--resume variable should contain the path of the model_best.pth.tar file

## Run Knowledge distillation training
To run knowledge distillation trainig use the following command lines:
 ### Training
 
 ```
ipython ./knowledge_distillation/run.py -- --learning-rate 0.0001 --loss CrossEntropy --opt ADAM  --hist_data True --distill_weight 0.5
```
There are two types of aggregated outputs for the teacher model:
  - Smoothe Prediction
    - To use this model outputs add this variable: --hist_data true  
  - Soft smoothe Prediction
    - To use this model outputs add this variable: --soft_data true  

 

 
 ### Adversarial training
 ```
ipython ./knowledge_distillation/run.py -- --adv_training True  --learning-rate 0.0001 -- --loss CrossEntropy --opt ADAM  --hist_data True --adv_training True --distill_weight 0.75 --perturb_distill_weight 0.25 
```


### Results on CIFAR10

|Method | Clean accuracy| PGD-10 accuracy|
|--- |---|---|
|Smooth prediction smoothing |  88.68| 63.67|
|Smooth soft prediction smoothing | 88.53| 63.48|
|KD prediction smoothing | 79.9| 46|
|KD soft prediction smoothing | 82.13| 45.83|

## Transfer Attack
To run transfer attack the following parameters need to be incorporated:
<li>--transfer-attack</li>
<li>--attack-path [path to the model that the attack will be create on]</li>

#### Use the following command to run a transfer attack - target model is CNI or Smoothed CNI: 
```
ipython run_attack.py -- --seed 42  --arch wideresnet --width 4 --layers 28  --batch-size 256 --cpni  --attack pgd --attack_k 10 --alpha 0.006 --smooth mcpredict --m_forward 512 --eps 8 --noise_sd 0 --transfer-attack --attack-path knowledge_distillation/kd_models/student_20220227-175932.pt --resume trained_models/cpni/CPNI_wide4_offd_decay_1e-3_time_2020-03-14_16-58-12/model_best.pth.tar --save results_transfer_attack_cni --experiment-name transfer_attack_student_cni_eps8 --gpus 0 
```

Parameters we experimented on:
<li>--eps - values we used: 2, 8, 30</li>
<li>--noise-sd - this param determines if the model is smoothed cni [0.25] or cni [0]</li>

#### Use the following command to run a transfer attack - target model is KD student: 
```
ipython ./knowledge_distillation/run.py -- --learning-rate 0.0001 --loss CrossEntropy --opt ADAM  --hist_data True --distill_weight 0.75 --perturb_distill_weight 0.25 --load-student-model True --resume-path knowledge_distillation/kd_models/student_20220227-175932.pt --transfer-attack --epsilon 8 --noise_sd 0.25 --attack-path trained_models/cpni/CPNI_wide4_offd_decay_1e-3_time_2020-03-14_16-58-12/model_best.pth.tar --experiment-name transfer_attack_scni_sudent_eps8_
```


### Results of Transfer Attack
Results for transfer attack using epsilon = 8/255.

|Attack Model| Target Model| Clean Accuracy| PGD-10 Accuracy|
|--- |--- |--- |---|
|CNI| CNI| 88.72| 63.67|
|Smoothed CNI| CNI| 88.67| 88.82|
|KD student| CNI| 88.76| 63.4|
|CNI| Smoothed CNI| 42.6| 42.62|
|Smoothed CNI| Smoothed CNI| 42.81| 36.58|
|KD student| Smoothed CNI|42.74 |32.95 |
|CNI| KD student| 79.92| 69.3|
|Smoothed CNI| KD student| 79.92| 77.97|
|KD student| KD student| 79.92| 46.06|

These result show that using Knowledge Distillation the student model was abel to learn a successful 
Black-Box attack on the CNI model. Read our project report to learn more.


## TODO:
<ol>
 <li> Add link to experiment excel </li>
 <li> Add link to PDF </li>
 <li> Add graphs </li>
 <li> Add refrences </li>
 <li> Add system requierments </li>
 </ol>




