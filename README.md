This is the output i got:-
Question:- List the technical skills in the resume.
Instruction:- I worked on AWS, Terraform, and CI/CD pipelines in Jenkins.
Output:- I'm a big fan of the AWS CLI, and I've been using it for a while now.

Simple output as i used gpt2 and also ran it through .to('cpu') and not .to('cuda') as my system didn't have GPU when checked using torch.

This is the training info, ran it through cpu didn't se bf16 or fp16 because of system. If using cloud then we can use GPU configured from there. 
{'train_runtime': 149.4407, 'train_samples_per_second': 0.241, 'train_steps_per_second': 0.04, 'train_loss': 4.061666170756022, 'epoch': 3.0}
