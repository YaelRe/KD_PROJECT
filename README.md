# Knowledge Distillation via Smoothed Models and Adversarial Robustness

This project came to pass as a continuation of the Nemcovsky et al.[1] research. 
In thier project they showed that combining smoothing along with randomization approaches and adversarial training can improve the robustness to adversarial attacks.

In our project we explored the possibility of using Knowledge distillation via the smoothed model from [1], to create a student model, that is deterministic 
and can perform on perturbed data similar to the smoothed model. We used the soft targets KD method that was developed by Hinton et al. [2] and
combined it with the adversarial training proposed by Madry et al. [3]

In addition we explored transfer-based attacks, we wanted to see if the student model had learned a weakness in Nemcovsky et al. [1] model
that could be exploited to create a black-box attack. To test this hypothesis we generated the PGD attack on the best performing student model, the one that was trained on soft prediction smoothing output, and then tested the attack on both the CNI and Smoothed CNI models.

## Get smoothed model outputs data
In order to run knowledge distillation trainig we need to get the smoothed model outputs.

There are two types of aggregated outputs that can be retrived with the following command lines:

 ### Smoothed Prediction
 
 ```
ipython run_attack.py -- --seed 42 --arch wideresnet --width 4 --layers 28 --batch-size 256 --cpni --smooth mcpredict --m_forward 512 --resume trained_models/cpni/CPNI_wide4_offd_decay_1e-3_time_2020-03-14_16-58-12/model_best.pth.tar 
```
 ### Soft smoothed Prediction
  ```
ipython run_attack.py -- --seed 42 --arch wideresnet --width 4 --layers 28 --batch-size 256 --cpni --smooth mcpredict --m_forward 512 --resume trained_models/cpni/CPNI_wide4_offd_decay_1e-3_time_2020-03-14_16-58-12/model_best.pth.tar
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
    - To use this model outputs add this variable: --soft_data true and remove --hist_data true 

 

 
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
ipython run_attack.py -- --seed 42  --arch wideresnet --width 4 --layers 28  --batch-size 256 --cpni  --attack pgd --attack_k 10 --alpha 0.006 --smooth mcpredict --m_forward 512 --eps 8 --noise_sd 0.25 --transfer-attack --transfer-attack-noise 0 --resume trained_models/cpni/CPNI_wide4_offd_decay_1e-3_time_2020-03-14_16-58-12/model_best.pth.tar --save results_transfer_attack_cni --experiment-name transfer_attack_cni_scni_eps8_ --gpus 0 | tee results_transfer_attack_mc.txt 
```

Parameters we experimented on:
<li>--eps - values we used: 2, 8, 30</li>
<li>--noise-sd - this param determines if the model is smoothed cni [0.25] or cni [0]</li>
<li>--transfer-attack-noise - makes the attack model a smoothed cni model </li>

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
Black-Box attack on the CNI and Smoothed models. Read our project paper to learn more.

## References
[1] [Smoothed Inference for Improving Adversarial Robustness](https://arxiv.org/pdf/1911.07198.pdf)\
[2] [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)\
[3] [Towards Deep Learning Models Resistant to Adversarial Attacks](https://openreview.net/forum?id=rJzIBfZAb)

## TODO:
<ol>
 <li> Add link to experiment excel </li>
 <li> Add link to PDF </li>
 <li> Add graphs </li>
 <li> Add references </li>
 <li> Add system requirements </li>
 </ol>




