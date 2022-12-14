# Homework

# Part 1: Adversarial examples (1 point)

In this section you will explore the effect of the epsilon and step size variables on the generation of adversarial examples. To do so, run the file named punto_1.py and test the generation of an adversarial example from the carro.jpg image. Then, experiment with the value of the epsilon and step size that are declared at the beginning of the file. In that sense, 9 experiments with 3 values of epsilon and 3 values of norm should be performed to obtain the full score (the idea is that you take a small value, a medium value and a large value for each parameter). In your report attach a subplot similar to the one shown as an example and explain why with the higher values of epsilon and stepsize the presence of noise is clearly seen in the generated image.


![My Image](figura_git-hub-tarea.png)

# Part 2: Adversarial attacks (2 points)

This point consists of attacking a pre-trained classifier with the CIFAR10 dataset. The idea is to carry out 9 experiments varying both the algorithm used (pgd or apgd) and different hyperparameters of the attacks. To do so, run the file punto_2.py, modifying the values declared in the first lines. In your report, insert a table with the results obtained and briefly describe the observed trends.(Note: Be careful with the parameters you decide to use for experimentation, remember that the idea is to generate adversarial examples that are similar to the original images. An attack with 0.0 accuracy may indicate that the adversarial examples are images with pure noise, which is not what we want. We recommend that the number of iterations you use be at least 40).




# Part 3: MM attack (2 points)

In this section we will explore a code of the Minimum Margin Attack algorithm. To do so, look at the file punto_3.py. In this case describe what the code is doing between the lines 147 and 150, more specifically, what information they contain and to which pseudocode variables the 'a' and 'b' parameters of the mentioned lines correspond. In your report answer these questions. Additionally, on line 162 of the same file there is a TODO that you must solve in order to run the code and produce an adversarial example using the MMA algorithm. 



# Bonus (0.5 points): 

Briefly answer the following questions in your report. 
What parameter of neural network training is related to the step size used in adversarial attacks?
What is the objective of the loss function proposed by the authors of APGD in an untargeted attack?



