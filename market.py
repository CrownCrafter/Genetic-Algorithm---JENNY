import pandas as pd
import numpy as np

def moving_average_crossover_strategy(data, short_window, long_window):
   data['short_mavg'] = data['Close'].rolling(window=short_window).mean()
   data['long_mavg'] = data['Close'].rolling(window=long_window).mean()
   data['signal'] = 0
   data['signal'][short_window:] = np.where(data['short_mavg'][short_window:] > data['long_mavg'][short_window:], 1, 0)
   data['positions'] = data['signal'].diff()
   return data

def calculate_fitness(data):
   data['returns'] = data['Close'].pct_change()
   data['strategy_returns'] = data['returns'] * data['positions'].shift(1)
   total_return = data['strategy_returns'].sum()
   return total_return

import random

def initialize_population(pop_size, short_window_range, long_window_range):
   population = []
   for _ in range(pop_size):
       short_window = random.randint(*short_window_range)
       long_window = random.randint(*long_window_range)
       population.append((short_window, long_window))
   return population

def evaluate_population(data, population):
   fitness_scores = []
   for short_window, long_window in population:
       strategy_data = moving_average_crossover_strategy(data.copy(), short_window, long_window)
       fitness = calculate_fitness(strategy_data)
       fitness_scores.append((fitness, (short_window, long_window)))
   return fitness_scores

def select_top_strategies(fitness_scores, num_top_strategies):
   fitness_scores.sort(reverse=True, key=lambda x: x[0])
   return [strategy for _, strategy in fitness_scores[:num_top_strategies]]

def crossover(parent1, parent2):
   child1 = (parent1[0], parent2[1])
   child2 = (parent2[0], parent1[1])
   return child1, child2

def mutate(strategy, mutation_rate, short_window_range, long_window_range):
   if random.random() < mutation_rate:
       return (random.randint(*short_window_range), strategy[1])
   elif random.random() < mutation_rate:
       return (strategy[0], random.randint(*long_window_range))
   else:
       return strategy
   
def evolve_population(data, population, num_generations, num_top_strategies, mutation_rate, short_window_range, long_window_range):
   for _ in range(num_generations):
       fitness_scores = evaluate_population(data, population)
       top_strategies = select_top_strategies(fitness_scores, num_top_strategies)
       new_population = top_strategies.copy()
       while len(new_population) < len(population):
           parent1, parent2 = random.sample(top_strategies, 2)
           child1, child2 = crossover(parent1, parent2)
           new_population.append(mutate(child1, mutation_rate, short_window_range, long_window_range))
           if len(new_population) < len(population):
               new_population.append(mutate(child2, mutation_rate, short_window_range, long_window_range))
       population = new_population
   return population

import yfinance as yf

# Fetch historical data for a given stock
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# Define parameters for the genetic algorithm
pop_size = 50
num_generations = 100
num_top_strategies = 10
mutation_rate = 0.1
short_window_range = (5, 50)
long_window_range = (50, 200)

# Initialize and evolve the population
population = initialize_population(pop_size, short_window_range, long_window_range)
evolved_population = evolve_population(data, population, num_generations, num_top_strategies, mutation_rate, short_window_range, long_window_range)

# Evaluate the final population to find the best strategy
final_fitness_scores = evaluate_population(data, evolved_population)
best_strategy = max(final_fitness_scores, key=lambda x: x[0])
print(f'Best strategy: Short Window = {best_strategy[1][0]}, Long Window = {best_strategy[1][1]}')z