


For all training 1.X, I used an instruct model. Hindsight this was pretty dumb. It could've worked if I just did GRPO and not SFT, but I really wanted to try to make SFT work so I could get more experience.

Because of this, after more money than I want to admit on trying to make it work, I learned that trying to fine tune an instruct model actually made it worse, which is why the SFT never improved on the base model.

Trainint 2.X on actually uses a base model, which IMO is far more interesting. 