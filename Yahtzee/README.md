# Yahtzee Reinforcement Learning Bot

## Overview

When I was in college, a few of my roommates and I played Yahtzee all the time. I’m not exactly sure why, but it probably has something to do with the mixture of strategy, luck (and therefore surprises), and casual gameplay that allowed for conversation concurrent with playing. So naturally, I wanted to create a bot that could beat them at the game – and that’s exactly what I did… except it probably isn’t good enough to beat them consistently. We’ll get into that later.

---

## How to play Yahtzee (via Wikipedia)

The objective of the game is to score points by rolling five dice to make certain combinations. The dice can be rolled up to three times in a turn to try to make various scoring combinations and dice must remain in the box. A game consists of thirteen rounds. After each round, the player chooses which scoring category is to be used for that round. Once a category has been used in the game, it cannot be used again. The scoring categories have varying point values, some of which are fixed values and others for which the score depends on the value of the dice. A Yahtzee is five-of-a-kind and scores 50 points, the highest of any category. The winner is the player who scores the most points.

---

## My strategy

I decided to use Dueling DQN as my algorithm mostly because of its relative simplicity for 1. Me to understand and 2. Me to implement. Dueling DQN essentially separates the q-value from traditional Q-learning into the (v)alue stream and the (a)dvantage stream. This means that it takes the value of the current state and then assesses different steps to take from there. Let’s say there are 2 steps it could take. The algorithm then calculates the advantage of taking each step and evaluates them against one another (hence the dueling). It determines the q-value by adding the value state to the action advantage minus the average action advantage.

However, this gets complex because of how many actions are possible at each step in Yahtzee. This is one possible explanation of model performance.

---

## My Hyperparameters

Here are the hyperparameters I chose and a brief explanation of each:

1. **Episodes: 50,000**
   a. This number was chosen initially because of computational limits. Because Yahtzee is so complicated of a game with so many steps, it took over 9 hours to train. I only have so many Google Colab compute units…

2. **Epsilon decay: 0.9999**
   a. Epsilon is the ratio of steps that are random, with the first steps being 100% random and the final ones being 0.0001% random. I chose this rate by trial and error to get proper training without over-fitting or forgetting.

3. **Gamma: 0.99**
   a. Gamma is the measure of how much the model should discount future value of actions. Because Yahtzee is a game that requires foresight with each play (one might take a higher score on 6s with risk of a 0 on 4 of a kind in order to get the upper bonus, for example), gamma had to be high so that the model values future points just as much as current ones.

4. **Batch size: 256**
   a. I’m not sure the impact this would have made on the actual model, I just kept it here so as to not hinder training time/performance.

5. **Warm-Up Period: 10,000 episodes**
   a. Yahtzee is complex, so I wanted to give the model ample time to familiarize itself with the game.

6. **Memory capacity: 300,000 steps**
   a. This means that the model remembers 300,000 steps when it learns. It is the same way our brain works – we don’t remember the first time we ever did addition, but we remember how to do it because we’ve done it more recently.

7. **Alpha: 0.7**
   a. This is the model’s risk tolerance. I wanted to make it higher than usual because Yahtzee is a game in which one should sometimes take risks, perhaps to get a Yahtzee.

---

## Rewards

Figuring out the reward values was one of the most challenging parts of this. In one of my first versions, I only gave a reward for the final score. But because of the complexity of Yahtzee, that didn’t work. Perhaps with more episodes it could have, but I’m skeptical.

Eventually, I settled on a system that uses the expected value of each category according to optimal play. Now this is cheating a little bit because it stands on the shoulders of those who have already written mathematical algorithms for Yahtzee. But alas, this is not a math thesis, this is a guy with a hobby.

### Early-game negative rewards

* **-0.1** for zeroing Yahtzee or 1s
* **-2** for choosing Chance too early
* Penalties for zeroing anything in the upper section (bonus is important)

### Additional rewards

I also added a few rewards for the more “rare” categories like small and large straights, full house, and Yahtzee. Finally, I added the biggest rewards of course for the final scores.
