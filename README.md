# Deep-Learning-and-Practice (DLP)

This repo contains the lab code in the 2022 Deep-Learning-and-Practice course.
For the detail of each lab please refer to its report.

## Lab02

Implement the back propagation from scratch and train the network to classify the linear and non-linear data.

-Linear data
>accuracy: 99%\
<img src="img/lab02_linear_result.png" alt = "linear" title = "linear_result" height=50% width=50%>


-Non-linear XOR data
>accuracy:99%\
<img src="img/lab02_nonlinear_result.png" alt = "nonlinear" title = "nonlinear_result" height=50% width=50%>

## Lab03

Mastering the 2048 game by reinforce-learning and N-tuple network.
>Score in 10000 times playing the 2048 game after training:
<img src="img/lab03_result.png" alt = "2048" title = "2048_result" height=50% width=50%>

## Lab04-1
Implement the EEGNet and DeepConvNet to classify the BCI dataset. In this lab, different activation function were tested for the preformance.

-EEGNet
>accuracy: 87.4% with LeakyReLU\
<img src="img/lab04-1_EEG.png" alt = "EEGNet" title = "EEGNet_result" height=50% width=50%>


-DeepConvNet
>accuracy:81.76 with ELU\
<img src="img/lab04-1_Deep.png" alt = "DeepConvNet" title = "DeepConvNet_result" height=50% width=50%>

## Lab04-2
Trained the classifier with Resnet18 and Resnet50 to detect the Diabetic Retinopathy.

-Resnet18
>accuracy: 82.47% with pretrained weight\
<img src="img/lab04-2_resnet18.png" alt = "resnet18" title = "resnet18_result" height=60% width=60%>

-Resnet50
>accuracy: 83.15% with pretrained weight\
<img src="img/lab04-2_resnet50.png" alt = "resnet50" title = "resnet50_result" height=60% width=60%>

## Lab05
Inplement the conditional variational autoencoder(VAE) to predict the motion frame of robot end effector.

-Ground truth
><img src="img/lab05_true.gif" alt = "vae" title = "vae_ture" height=40% width=40%>

-Predict result
> psnr : 23.6 dB\
<img src="img/lab05_pred.gif" alt = "vae" title = "vae_pred" height=40% width=40%>

## Lab06
Master the LunarLander game with reinforce-learning, using DQN, DDPG, DDQN and TD3.

-Result
>LunarLander score of each RL model\
<img src="img/lab06_result.png" alt = "lunar" title = "lunar_result" height=100% width=100%>

## Lab07
Implenet the conditional generative adversarial network(GAN) to generate the specific synthetic images.

-Generated images: (from only words such as 'red cylinder, blue cube... etc')
><img src="img/lab07_result.png" alt = "GAN" title = "GAN_result" height=70% width=70%>


## Final
We dicussed the to pretrained a classifiaction model.
Three pretrained weights were generated from autoencoder, K-means and simCLR seperately. Then the weights were used to fine-tune and detects on covid-19 detection task. We discuss that the variety of the pretraining dataset is more important to the performance of down-stream task.

-Result
><img src="img/final.png" alt = "final" title = "final_result" height=70% width=70%>
