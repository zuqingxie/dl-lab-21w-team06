# Introdution
This project started as a deep learning laboratory from university Stuttgart with institute ISS (Institut für Signalverarbeitung und Systemtheorie), which contains two classification missions. They are **Diabetic Retinopathy Detection** and 
**Human Activity Recognition**. The detail descriptions of the requirements could be read in `DL_Lab_21W_script.pdf` in Documents
# Team info 
   team number: 06
  <br />
  team member:
- **Zuqing Xie (st146813)**
- **Chongzhe Zhang (st171393)**


# How to run the code
Go to the home path (you can choose other path, but we would like the use home path as example):

    cd ~
Git clone the file with the branch master:

    git clone -b master https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team06.git

**batch.sh** for running 1. project is under the path: *~/dl-lab-21w-team06/diabetic_retinopathy/batch.sh*</br>
**batch.sh** for running 2. project is under the path: *~/dl-lab-21w-team06/human_activity_recognition/batch.sh*</br>
**config.gin** file for configure the parameter of the 1.project is under the path: *~/dl-lab-21w-team06/diabetic_retinopathy/configs/config.gin*</br>
**config.gin** file for configure the parameter of the 2.project is under the path: *~/dl-lab-21w-team06/human_activity_recognition/configs/config.gin*</br>
## run project 1:  Diabetic Retinopathy Detection

### Training and evaluating:

We defined some models and can be seperated into two parts. First parts are custom models like `vgg_like`,`CNN_team06`,`resnet_simple`. And second parts are transfer learning models like: `mobilenet`, `inceptionV3`,  `inception_resnet_v2`,  `xception`. We can also set the corresponding parameters in the `config.gin`
- Uncomment the corresponding line in **batch.sh**. And choose a model that you want to train. 


      # Training with different models: vgg_like, CNN_team06, resnet_simple
      # Or transfer learning model: mobilenet, inceptionV3,  inception_resnet_v2,  xception
      python3 main.py -train=True -model_name="vgg_like"
      # ...

- Run it!


      sbatch ~/dl-lab-21w-team06/diabetic_retinopathy/batch.sh



### Evaluation without training:
- Uncomment the corresponding line in **batch.sh**. And set the `eval_folder_name` to the name of the folder that you want to evaluate.
      
  
      # Evaluate the given latest Checkponit without training 
      python3 main.py -train=False -eval_folder_name="run_xxxx-xx-xxxxx-xx-xx-xxxxxx"

- Run it!


      sbatch ~/dl-lab-21w-team06/diabetic_retinopathy/batch.sh

### Ensemble learning:

- Uncomment the corresponding line in **batch.sh**.


      # Run ensemble learning
      python3 ensemble.py

- Run it!


      sbatch ~/dl-lab-21w-team06/diabetic_retinopathy/batch.sh

### Check the result of the training:
During the training a newest experiments folder will be generated under the folder ~/dl-lab-21w-team06/experiments.
like: folder `run1_xxxx-xx-xxxxx-xx-xx-xxxxxx`. In this folder the gin configuration will be saved as config_operative.gin.
In the folder logs the running log was saved as `run.log`. Additionally in the ckpts folder will contain all the saved checkpoints and an `eval` folder.
In the eval folder will have a `confusionmatrix.png` picture based on the latest checkpoints.


### Result of Diabetic Retinopathy Detection:


#### The image after the preprocessing:
.<div align=center><img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team06/blob/master/images/preprocess.jpg" width="550" height="370" />
.<div align=center>***Image1: preprocessed image example***

.<div align=left>
#### Image from the visualization:
.<div align=center><img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team06/blob/master/images/visualizaiton.png" width="550" height="500" />
.<div align=center>***Image2: Visualization***

.<div align=left>
#### Evaluate the performance of the model based on the confusion matrix.
.<div align=center><img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team06/blob/master/images/confusionmatrix_1.png" width="300" height="300" />
.<div align=center>***Image3: Confusionmatrix***

.<div align=left>
#### The result of the Transfer learning: 

.<div align=center>

| |model             |sensitivity|specificity|precision|recall|F1-score |F0.9-score |Test accuracy(%)|
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|1    |Inception_Resnet_V2|            0.76          |   0.54    |     0.73       |0.76 |   **0.751**    | **0.742**        |  **67.96**    |
|2    |Inception_V3              |    0.7    |       0.54      |  0.71     |      0.7       |    0.708   |   0.709  |    64.08    |
|3    |Xception             |      0.7  | 0.54    | 0.714  |  0.7     | 0.708  |   0.709   |    64.07    |  
|4    |Mobilenet        |    0.79  |      0.3    |     0.65  |   0.79    |   0.718  | 0.716     |       61.16  |

.<div align=center>***Table1: Evaluation the best performance of the Transfer learning models***


.<div align=left>
#### Optomization result of VGG_like model:

.<div align=center>

| |block | filters| neurons in FC layer |dropout rate |test accuracy(%)|
|:---|:---|:---|:---|:---|:---|
 |   1     | 4     | 18    | 32   |0.4   | 89.3 |
 |   2     | 4     | 16    | 64   | 0.5  | 84.5|
 |   3     | 4     | 16   | 48   | 0.5  | 82.5 |
 |   4     | 5     | 16   | 48   | 0.45  | 81.6 |
 |   5     | 4     | 14   | 32   | 0.5  | 80.6 |

.<div align=center>***Table2: Optomization result of VGG_like model***




.<div align=left>
#### Optomization result of Resnet model :

.<div align=center>


| |filters | neurons in FC layer| dropout rate | test accuracy(\%) |
|:---|:---|:---|:---|:---|
 |   1    | 10  |8    | 0.4  |74.8 |
 |   2   |10   |8     | 0.5  | 74.8 |
  |  3   |6   | 8    | 0.4  |73.8 |
  |  4     | 4    | 4    | 0.5  | 72.8 |
  |  5    | 4    | 4    | 0.45  | 71.8 |
  |  6    | 6    | 16   | 0.45  |71.8 |
  |  7    | 10    | 16   | 0.45 | 70.9 |

.<div align=center>***Table3:  Optomization result of Resnet model***

.<div align=left>
#### Optomization result of CNN_team06 model :

.<div align=center>

| |filters |kernel size | strides |dropout rate | test accuracy(\%)|
|:---|:---|:---|:---|:---|:---|
|    1     | (12,24,48,96) | (5,5,3,3) | (1,1,1,1) | 0.5  |83.5 |
 |   2    | (10,20,40,80) | (5,3,3,3) | (2,2,1,1) | 0.5  | 79.6 |
|    3    | (14,28,56,112) | (5,3,3,3) |(1,1,1,1) | 0.4   | 77.7 |
|    4    | (8,16,32,64) | (5,5,3,3)| (2,1,1,1) | 0.5  | 74.8 |
 |   5    | (14,28,56,112)|(5,5,3,3) |(2,2,1,1) | 0.4  | 74.8 |

.<div align=center>***Table4:  Optomization result of CNN_team06 model***
.<div align=left>

### Conclusion of the 1.project:

1) For the classification of diabetic retinopathy, the simple VGG-like model has better performance because of the small amount of data. This is because more complex models contain more parameters, which can easily lead to overfitting. After adjusting the parameters, the VGG-like model can achieve a test accuracy of 89.3%, which proves the good generalization ability.
2) For ensemble learning. It didn't performance better as the best single model. the reason could be the lack number of the models and the performance capability among models maintain a huge gap. For example VGG_like model it has accuracy about 89%, but for Resnet model it has only about 75% accuracy. To choose a proper weight vector would be very random and difficult. 
3) Transfer learning models performance worse than other models. They couldn't transfer what they learned to our project very well, although they have similar features between two domains.

### What we can do more:
It could be even better if we tune the weight vector with optimization method for our ensemble learning.

.<div align=left>

****
## run project 2:  Human Activity Recognition:
### Training and evaluating:
####  1. Train the pure `gru`, `lstm` or `simple_rnn` model:

    
- Choose a pure RNN  model. you need to open the config.gin file with your editor( eg. vim )
  

      vim ~/dl-lab-21w-team06/human_activity_recognition/configs/config.gin

please set the value of rnn.rnn_name = "gru" | "lstm" | "simple_rnn".

- Uncomment the corresponding line in **batch.sh**.


      # Training the rnn model, the type of the model pls choose it in the config.gin first
      python3 main.py -train=True -device_name='GPU-Server' -model_name='rnn' -ensemble_learning=False
      # ...

- Run it!
    

      sbatch ~/dl-lab-21w-team06/human_activity_recognition/batch.sh

#### 2. Train the mixed RNN models:

- Uncomment the corresponding line in **batch.sh**.


      # Training the rnn_mix model
      python3 main.py -train=True -device_name='GPU-Server' -model_name='rnn_mix' -ensemble_learning=False
      # ...

- Run it!
    
    
      sbatch ~/dl-lab-21w-team06/human_activity_recognition/batch.sh


#### 3. Ensemble learning models:

- Uncomment the corresponding part in **batch.sh**.


      # Run ensemble learning
      python3 main.py -train=False -device_name='GPU-Server' -ensemble_learning=True
      # ...




- Run it!


      sbatch ~/dl-lab-21w-team06/human_activity_recognition/batch.sh
* #### If you want to customize the ensemble learning with different models or different weights vector:
  

      1) Setting the weights_list in file ~/dl-lab-21w-team06/human_activity_recognition/configs/config_ensemble.gin. the sum of all the number in list should be 1. And with the length of the Model that you choose
    
      eg.:  Ensemble.weight_list = [0.3, 0.2, ...]
  
      2) Setting the model parameters and checkpoints paths of every model in file ~/dl-lab-21w-team06/human_activity_recognition/ensemble.py
      eg.:  ...
            self.model_1 = rnn(window_size=300, lstm_units=112, dropout_rate=0.4, dense_units=64, rnn_name='lstm',
                                num_rnn=4, activation='tanh')
            self.model_2 = rnn(window_size=300, lstm_units=128, dropout_rate=0.4, dense_units=32, rnn_name='gru', 
                                num_rnn=5, activation='tanh')
            ...
            self.model = [self.model_1, self.model_2, ...]
            self.run_paths_ckpt_1 = "~/dl-lab-21w-team06/experiments_test/run2_2022-02-11T17-15-09-480006/ckpts"
            self.run_paths_ckpt_2 = "~/dl-lab-21w-team06/experiments_test/run2_2022-02-11T18-11-49-619451/ckpts"
            ...
            self.run_paths_ckpt = [self.run_paths_ckpt_1, self.run_paths_ckpt_2, ...]
            ...

      3) Run it!
            sbatch ~/dl-lab-21w-team06/human_activity_recognition/batch.sh


### Check the results of the training:
During the training a newest experiments folder will be generated under the folder ~/dl-lab-21w-team06/experiments.
like: folder `run2_xxxx-xx-xxxxx-xx-xx-xxxxxx`. In this folder the gin configuration will be saved as config_operative.gin.
In the folder logs the running log was saved as `run.log`. Additionally in the ckpts folder will contain all the saved checkpoints and an `eval` folder.
in the eval folder will have a `confusionmatrix.png` picture based on the latest checkpoints.


### Evaluate the **specific number** of the checkponit:
After all the training we will automatically evaluate the latest checkpoint. This part of programm provide us a method to evaluate the specific number of checkpoint.


- Uncomment the corresponding line in **batch.sh**. We need to set`eval_folder_name`and checkpoint index `index_ckpt` that we want to evaluate


      # evaluate a specific Checkponit eg.:
      python3 main.py -train=False -device_name='GPU-Server' -ensemble_learning=False -eval_folder_name="run2_2022-02-07T22-20-23-685508" -index_ckpt=2

- Run it!


      sbatch ~/dl-lab-21w-team06/human_activity_recognition/batch.sh


### Results of Human Activity Recognition:



#### Model structure:
.<div align=center><img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team06/blob/master/images/model_structure.png" width="710" height="400" />
.<div align=center>***Image4: Multimodels LSTM, GRU, SimpleRNN and RNN_mix***

.<div align=left>
#### Evaluation:
.<div align=center><img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team06/blob/master/images/confusionmatrix.png" width="550" height="500" />
.<div align=center>***Image5: Confusionmatrix with accuracy:90.74, sensitivity: 0.928, specificity: 0.957, f1_score:0.908***
.<div align=center><img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team06/blob/master/images/labeled_rawdata.png" width="700" height="700" />
.<div align=center>***Image6: Labeled RawData***

.<div align=left>
We use weighted F1-score to evaluate the performance of the model which take the number of the label into account. 

.<div align=center><img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team06/blob/master/images/f1_score.png" width="500" height="140" />

.<div align=left>
#### Best performance of each model:
The result in the real case is not stable enough, especially for the `simple_rnn` and `rnn_mix`. we choose the best result that we met as the following table.



|Rank |Type             |Window_len|Window_shift|RNN_units|Densse_units|Num_RNN|Drop_rate|Test_acc%|Val_acc%|F1_Score|
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|1    |Ensemble learning|          |             |       |             |       |         |   **94.19**      |        |    **0.92**    |
|2    |GRU              |    300    |       150      |  112     |      96       |    5   |     0.4    |    93.63     |   **94.23**     |  0.919      |
|3    |LSTM             |      300  |      150    |  112     |       96      |   5    |    0.4     |    90.08     |    93.08    |    0.901    |
|4    |SimpleRNN        |    250  |      100    |     96  |       112      |     6  |   0.3      |       90.07  |    91.45    |    0.889    |
|5    |RNN_MIX          |    250    |     150     |    112   |      112       |   5    |      0.5   |     88.72    |    84.01    |     0.878   |


.<div align=center>***Table5: Best performance of each model***

.<div align=left>
#### Conclusion of the 2. project:
1) Weighted Ensemble learning performance better others. But it is hard to set the weight vector. 
2) GRU model is better as LSTM and SimpleRNN.
3) RNN_MIX performances not good as expected.
4) Balanced Input pipeline sometime cause the instable of training.

#### What can we do more:
1) Using the wandb to tune the weight list of the ensemble learning. Which could get a much better result.
2) It can be more intuitive and scientific to compare the difference between the Balanced and Inbalanced input dataset.
