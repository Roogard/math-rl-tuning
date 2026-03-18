# Math RL Tuning


## Intro

In the past couple years, many different people, teams, and labs have been trying to see how accurate LLM's can get at math problems. When I saw this, I got really curious about how the process worked. What did the training look like? How can you verify if a math answer is correct? Because of this, I decided to try to make it myself to learn how it works. From the beginning, the plan was to take an LLM, and train it through supervised fine-tuning (SFT) and group relative policy optimization (GRPO). Also as a side note, before this project, I didn't have any experience with agentic engineering, so I wanted to use the project to also learn alot about how to effectively use it. 

## Tech

To do this, I used VSCode and Claude Code to generate a large amount of this project. However, I still coded soem things by hand, as well as made all the decisions, set up the structure, ran the training, and did a large part of the debugging. 

For running SFT and GRPO, I ended up using Google Colab to train the models. This ended up being pretty inefficient, since I had to push to github every time I wanted to make one small change. However, I have the google student plan, so I really wanted to use the free credits.

Also, compute and budget was a large influencer on how this project turned out. I am a broke college TA, so I don't really have access to GPU clusters. I however did spend $70 on google colab credits due to training and testing models. Today, I would have used a lot less money since I understand what's going on in the project a lot more now. 

## Design Decisions (Success and failure included)

This project went through a lot of phases. It started out on 3 different colab notebooks, with all the code inside those notebooks. It was very inefficient, mostly AI generated, and a huge mess. Eventually, I had enough and split all the code into a github repository instead, which ended up being far more efficient. 

After this, my next answer was verifying LLM answers to see if they found the right answer. At first, I had reward functions for if the right answer was in boxed, and if the right answer was there at all. However, this ended up being a terrible design. Beyond this, the actual parser that tried to find if the right answer was in boxed was broken for a while, so I only ever ended up getting the partial answer. Due to this, for a couple days I threw out any kind of training and just ensured that pretty much any answer was capable of being found. The library math_verify helped a lot with this.

After this, I did a lot of different trainings. However, they all relied on using SFT on an instruct model. For 90% of the project, I did all my trainings on instruct models instead of base models. Here, I learned the very valuable lesson to not perform SFT on instruct models, because it normally degrades performance instead of improving it. After switching to a base model, it went from a decrease of 10% in performance after SFT training to an increase in 20% after SFT.

Also, GRPO got better after adding negative reward for getting math questions incorrect, added more gradients to train from.

These are only a small part of the issues I ran into, so if you have any other questions or just want to hear more, email me or interview me or something.

## Results

Final accuracies:
Base Model: 35% accuracy
SFT Model: 55% accuracy
GRPO Model: 


Reward Graph for GRPO training:




## Possible Next Steps

If I had the chance to try this project again, I'd change everything. First, I'd completely skip SFT and just start with an instruct model. Then, I would completely change how I structure the prompts. Currently, agent harnesses seem very interesting, and a way I could get more reasoning into LLM answers, so I'd change the architecture and try to compare the two. 



## License

MIT
