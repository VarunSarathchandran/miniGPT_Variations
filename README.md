
This repository contains files related to the project titled:   
Exploring Variations of Transformer Architecture in miniGPT Design.

Course Name: Machine learning, a Bayesian Perspective (EE4685)  
Course Intructor: Dr. ir. Justin Dawels

Abstract: This project offers insight into the transformer architecture via a mini GPT implementation. We start with a base model built by Andrej Karpathy in https://github.com/karpathy/ng-video-lecture, which essentially implements the "Attention is All You Need" paper. We then implement different types of transformer architectures, attention types and position representations on the base model. Two different attention mechanisms- dot product attention and additive attention, and two different transformer blocks- the traditional Transformer block (as found in “Attention is All You Need” paper), and Weighted Transformer are presented. A powerful method of position representation- Relative Position Representation(RPR) is also implemented. The user can switch between different types of attention and transformer blocks, as well as toggle RPR on/off to study its effects on the loss and generated text.

Instructions to run the code:
1. Clone the repository
2. Create a virtual environment on VS Code
3. Install the requirements using -
4. pip install -r requirements.txt
5. Run the python script with desired architecture and hyperparameters
6. If desired, download a model checkpoint at https://drive.google.com/file/d/1ibkj-czO6VuwzOqgtPeiwtBQvubae54H/view?usp=share_link.	
Note: The checkpoint contains a model pre-trained on Wikipedia and fine-tuned on Shakespeare. The default architecture parameters and hyperparamters in the code have been used(Dot product attention, Traditional Transformer block, with RPR).			
CAUTION: the checkpoint cannot be loaded if the architecture parameters or the hyperparameters are changed.

Details of the Files in the Repository:  
1.Datasets: Two different datasets are included- ‘Input.txt’: Shakespeare text and ‘Enwik8.txt’: 100mb of Wikipedia  
2. Generated Data: The code writes a .txt file with 1000 generated tokens into this folder.  
3. Saved Checkpoints: Every 100 iterations, the code saves a checkpoint with model weights, loss and optimizer dictionary  
4. Tokenizers: Two different tokenizers, Regex and Basic are included, although only Regex is used.   
5. miniGPT_Variations: Python script with code  
6. requirements.txt: required packages to be installed.  

Instructions on running custom experiments:
1. Choose the desired architecture variations in the “Experiment Parameters”. Simply change the variable to the desired parameter. Ex->  
      Attention options -> attention=”additive” / attention=”dot product”  
      Block options -> block= “weighted”/ block= “traditional”  
	    RPR-> RPR=True / RPR=False  
      (note: changing include_tokenizer to False will revert to the character level model)
The user may choose whatever combination they wish. Note that RPR cannot be used with additive attention. The model will force attention to dot product  if RPR is True.
2. The model saves checkpoints every eval_interval number of iterations in the Saved Checkpoints folder.
3. The model will generate a plot of the losses and also store a generated text file in the GeneratedData folder.
4. To load a checkpoint: load_checkpoint= False, and enter the checkpoint name.

Brief Summary of Results:
1. Performance metrics used: Training and Validation loss.
2. Additive attention does not outperform dot-product attention
3. The weighted transformer block marginally outperforms the traditional block, but incurs larger training cost.
4. Relative position representation is a powerful tool to encode pairwise relationships between tokens- reduces loss and spelling errors.
5. Experiments on pre-training on Wikipedia and fine tuning on shakespeare were conducted. Pre-training allowed lowering loss via a large model, while preventing overfitting.
6. Experiments were conducted on Chinese text. 

Reflection Section:
​​Through the course of this project, we have been able to gain insight into transformers and attention mechanisms. Watching Andrej Karapthy's video lectures were also immensly helpful. His initial lectures were about simpler LLMs and then the GPT model is presented as a final lesson. Attention mechansims, and masking are very intuitevly introduced as a solution to a problem previous LLM models had. 
Performing a literature review helped understand what changes have been introduced in the transformer architecture. Implementing them was also a lesson in learning how to convert paper -> code.
Next time, we would be more precise with experiments desgined to pitch different transformer architectures with each other. For example, we currently dont perform a hyperparameter search- and that means that a model has not been truly evaluated in its hyperparameter space. We also do not use a test set to report final loss values from the models. Although we did realise that we must use a test set half way, it would have been too expensive to rerun the experiments while holding out a test set. (we used Lambda labs, an online GPU service which was billed by the hour.)
