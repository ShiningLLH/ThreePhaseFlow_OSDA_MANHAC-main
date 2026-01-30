# Cross-Domain Open-Set State Monitoring for Three-Phase Flow by MANHAC

Source code of MANHAC on oil-gas-water three-phase flow dataset.
The dataset is obtained through multiphase flow experiment at Tianjin Key Laboratory
of Process Measurement and Control at Tianjin University.

The details of the data and model can be found in    
 [L. H. Li, et al. Multigrained Adversarial Network With Hierarchical Attribute
 Causality: Cross-Domain Open-Set State Monitoring for Three-Phase Flow, TII, 2026.]
(https://doi.org/10.1109/TII.2025.3649061)


#### Fast execution in command line:  
python3 MANHAC_main.py  


#### Results Example:  
Target Accuracy_known: (3.3333%)  
Target Accuracy_outlier: (99.8333%)  
Overall_Accuracy: (41.9333%)  
class 0: accuracy = 0.0000  
class 1: accuracy = 6.6667  
class 2: accuracy = 3.3333  
Epoch 0 | Loss: 14.1265 | Att_Loss: 10.2917 | t: 0.7965

Target Accuracy_known: (16.5556%)  
Target Accuracy_outlier: (95.6667%)  
Overall_Accuracy: (48.2000%)  
class 0: accuracy = 2.0000  
class 1: accuracy = 19.0000  
class 2: accuracy = 28.6667  
Epoch 1 | Loss: 12.8091 | Att_Loss: 9.1397 | t: 0.7875

......

#### All rights reserved, citing the following papers are required for reference:   
[1] L. H. Li, et al. Multigrained Adversarial Network With Hierarchical Attribute
 Causality: Cross-Domain Open-Set State Monitoring for Three-Phase Flow,
IEEE Trans. Ind. Informat., 2026.