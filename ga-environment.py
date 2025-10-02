from functions.numpy.ma import *
from functions.log import *
from functions.memory_retention import *
from functions.custom_indicators import *
from functions.indicators import *
from functions.graphing.performance_graph import *
from functions.graphing.main_graph import *
from functions.graphing.graph_logger import *

import pandas as pd
import traceback
import numpy as np
from datetime import datetime, timezone

import random
from deap import base, creator, tools
from scipy.fft import fft
from scipy.signal import hilbert

import multiprocessing
import traceback


import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict




def load_data_numpy(file_path):
    df = pd.read_csv(file_path)
    
    df.columns = df.columns.str.strip().str.lower()

    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    if df['timestamp'].isnull().any():
        raise ValueError("Some timestamps could not be converted to numeric values.")
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def split_data_custom(data, num_folds=1):
    return 0

################################ DATA MANIPULATION #####################################
def get_rolling_windows(df, current_index, min_index=0):

    relevant_data = df.iloc[min_index:current_index + 1]

    standard_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    relevant_data_np = relevant_data[standard_cols].to_numpy()

    def extract_np(data, suffix):
        cols = [col for col in data.columns if col.endswith(suffix)]
        if not cols:
            return np.empty((0, 6))
        temp = data.dropna(subset=cols)
        if temp.empty:
            return np.empty((0, 6))
        keep = ['timestamp'] + cols
        temp = temp[keep].copy()
        temp.columns = ['timestamp'] + [c.replace(suffix, '') for c in cols]
        try:
            return temp[standard_cols].to_numpy()
        except KeyError:
            return np.empty((0, 6))

    # Extract all 4 higher timeframes using consistent suffixes
    relevant_data_first_higher_np         = extract_np(relevant_data, '_h1')  
    relevant_data_second_higher_np  = extract_np(relevant_data, '_h2')  
    relevant_data_third_higher_np   = extract_np(relevant_data, '_h3')
    relevant_data_fourth_higher_np  = extract_np(relevant_data, '_h4')

    return (
        relevant_data_np,
        relevant_data_first_higher_np,
        relevant_data_second_higher_np,
        relevant_data_third_higher_np,
        relevant_data_fourth_higher_np
    )

def calculate_candles(candles: np.ndarray, price_type: str) -> np.ndarray:
    COL = {
        'open': 1,
        'high': 2,
        'low': 3,
        'close': 4,
        'volume': 5 
    }

    if price_type in COL:
        return candles[:, COL[price_type]]
    elif price_type == 'ohlc4':
        return (candles[:, 1] + candles[:, 2] + candles[:, 3] + candles[:, 4]) / 4
    elif price_type == 'hlc3':
        return (candles[:, 2] + candles[:, 3] + candles[:, 4]) / 3
    elif price_type == 'hl2':
        return (candles[:, 2] + candles[:, 3]) / 2
    elif price_type == 'hlcc4':
        return (candles[:, 2] + candles[:, 3] + 2 * candles[:, 4]) / 4
    else:
        raise ValueError(f"Invalid price_type: {price_type}")



ENV_DESC = """"""

# GA PARAMETERS
pop_size = 40 
gen_size = 7   
cross_prob = 0.9  
mut_prob = 0.6
eli_size = 24 
tournament_size = 3

BLEND_ALPHA = 0.6 
LOWER_TIMEFRAME_MIN = 30 

# Versioning
ENV_TYPE = "QA"
ENV_TOKEN = "SOL"
ENV_VERSION = "v1.0.0"

# Configuration
num_processes = min(10, multiprocessing.cpu_count())
DATA_FILE = ENV_TOKEN + 'USDT_30min_1h_2h_4h_6h_182d.csv'
STATIC_PARAM_FILE = 'top_episodes-'+ENV_TOKEN+'-v'+ENV_VERSION+'-'+ENV_TYPE+'.jsonl'
TOP_EPISODES_LOAD_COUNT = 50

OPTIMIZATION = False
RERUN_TOP_PARAMS = False

SINGLE_VALIDATION = False
POST_VALIDATION = False

RUN_TOP_EPISODE = True

PLOT_TOP_EPISODE = True

VALIDATION_DATA_FILE = ENV_TOKEN + 'USDT_30min_1h_2h_4h_6h_92d.csv'

PARTIAL_GENOME_FILE = 'partial_genomes-'+ENV_TOKEN+'-v'+ENV_VERSION+'-'+ENV_TYPE+'.jsonl'
PARTIAL_GENOME = False  
PARTIAL_FILL_MODE = "fill"  

LOG_EPISODE_PREFIX = ENV_TOKEN+'-v'+ENV_VERSION+'-'+ENV_TYPE

MA_TYPES_FULL = ["EMA", "RMA", "SMA", "WMA", "VWMA", "DEMA", "TEMA", "SMMA", "ZLEMA", "HMA", "TMA", "LSMA", "KAMA"] 
MA_TYPES = ["EMA", "RMA", "WMA", "VWMA", "DEMA", "TEMA", "ZLEMA", "HMA", "TMA", "LSMA", "KAMA"] #
RSI_TYPES = ["EMA", "RMA", "SMA", "WMA", "DEMA", "TEMA", "SMMA", "ZLEMA", "HMA", "TMA", "LSMA", "KAMA"] 



############################# Parameter bounds ##################################
PARAM_BOUNDS = {

    "ma_long_higher_period": (20, 45),
    "ma_long_higher_type": (0, len(MA_TYPES_FULL) - 1), 
    "ma_long_higher_lookback": (3, 10),
 
    "srsi_period_lower": (6, 30),
    "stoch_period_lower": (6, 30),
    "smooth_k_lower": (2, 15),
    "smooth_d_lower": (2, 15),

    "rsi_period": (6, 30),
    "rsi_smoothing_period": (3, 8),
    "rsi_smoothing_type": (0, len(RSI_TYPES) - 1), 
    #...
}

############################# Indicator classes ##################################

class Bollinger:
    def __init__(self):
        self.upper_band = None
        self.lower_band = None
        self.middle_band = None

############################### Trade logging ####################################

def log_trade(
    entry_timestamp,
    exit_timestamp,
    entry_price,
    exit_price,
    entry_index,
    exit_index,
    position_type,
    trade_highs,
    trade_lows,
    order_size,
    order_fee,
    trade_type, 
    trailing_type
) -> dict:

    # Calculate gross profit
    if position_type == "long":
        gross_profit = (exit_price - entry_price) * (order_size / entry_price)
        drawdown = (entry_price - min(trade_lows)) / entry_price
    elif position_type == "short":
        gross_profit = (entry_price - exit_price) * (order_size / entry_price)
        drawdown = (max(trade_highs) - entry_price) / entry_price
    else:
        raise ValueError("position_type must be 'long' or 'short'")

    # Account for entry and exit fees
    total_fees = 2 * order_fee * order_size
    net_profit = gross_profit - total_fees
    profit_pct = (net_profit / order_size) * 100

    was_win = net_profit > 0
    duration_time = (exit_timestamp - entry_timestamp) / (1000 * 60)
    duration_bars = exit_index - entry_index

    return {
        "entry_time": entry_timestamp,
        "exit_time": exit_timestamp,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "position_type": position_type,
        "profit_abs": net_profit,
        "profit_pct": profit_pct,
        "duration_time": duration_time,
        "duration_bars": duration_bars,
        "was_win": was_win,
        "drawdown": drawdown,
        "trade_type": trade_type, 
        "trailing_type": trailing_type, 
    }

############################# Trading environment ##################################

class TradingEnvironment:
    def __init__(self, parameters, data):
        # Data setup
        self.data =  data 
        self.parameters = parameters
        
        self.current_phase = "buy"
        self.order_size = 1000
        self.leverage = 10
        self.order_fee = 0.00055 

        # Step indexing
        self.min_index = 0 
        self.current_index = self.min_index + 180

        # Trade logging
        self.trade_log = []
        self.graph_log = []

        # Trade parameters
        self.entry_timestamp = None 
        self.exit_timestamp = None 
        self.entry_price = None 
        self.exit_price = None 
        self.entry_index = None 
        self.exit_index = None      
        self.position_type = None 
        self.profit_abs = None 
        self.profit_pct = None 
        self.trade_highs = []
        self.trade_lows = []

    def reset(self):
        self.current_phase = "buy"

        # Step indexing
        self.min_index = 0 
        self.current_index = self.min_index + 180

        # Trade logging
        self.trade_log = []
        self.graph_log = []

        self.entry_timestamp = None 
        self.exit_timestamp = None 
        self.entry_price = None
        self.exit_price = None 
        self.entry_index = None 
        self.exit_index = None  
        self.position_type = None
        self.profit_abs = None 
        self.profit_pct = None 
        self.trade_highs = []
        self.trade_lows = []
  
    def step(self):
        """Run one step of the trading simulation."""
        if self.current_index >= len(self.data) - 1: 
            return False  # End of dataset
        
        ###############################
        ### Insert Trading Strategy ###
        ###############################

        self.current_index += 1
        self.min_index += 1
        return True
    


    def run_episode(self):
        """Run a full trading episode with a maximum step limit."""
        self.reset()
        steps = 0
        max_steps = len(self.data) + 5000
    
        while self.step():
            steps += 1

            if steps > max_steps:
                print(f"Max step limit of {max_steps} reached. Invalidating episode.")
                self.trade_log = []  # Clear any trades to prevent false fitness
                break

        return


# -------------------------------------------------------------------------------#
# -----------------------GA Setup and Evaluation Functions-----------------------#
# -------------------------------------------------------------------------------#

# Fitness and individual structures
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))  # Profit, Sharpe, -Drawdown
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Toolbox for the GA
toolbox = base.Toolbox()
fitness_cache = {}

# Attribute generator for each parameter
for key, bounds in PARAM_BOUNDS.items():
    if isinstance(bounds[0], int):  # Integer bounds
        toolbox.register(f"attr_{key}", random.randint, bounds[0], bounds[1])
    else:  # Float bounds
        toolbox.register(f"attr_{key}", random.uniform, bounds[0], bounds[1])

# Register individual creation
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    [toolbox.__getattribute__(f"attr_{key}") for key in PARAM_BOUNDS.keys()],
    n=1
)

# Create the population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_population(population, data, pool, timeout=500):
    unevaluated = [ind for ind in population if not ind.fitness.valid]
    if not unevaluated:
        return []   

    try:
        result = pool.starmap_async(
            evaluate_individual_cv, [(ind, data) for ind in unevaluated])
        fitnesses = result.get(timeout=timeout)

    except multiprocessing.TimeoutError:
        return None                   

    for ind, fit in zip(unevaluated, fitnesses):
        if fit in {None, (-9999, -9999, 9999)}:
            ind[:] = toolbox.individual()[:]
            if hasattr(ind.fitness, "values"):
                del ind.fitness.values
        else:
            ind.fitness.values = fit

    return fitnesses

def parameters_to_individual(entry):
    params = entry["parameters"]
    fitness = entry.get("fitness")

    individual = creator.Individual()
    for key, bounds in PARAM_BOUNDS.items():
        if key in params:
            val = params[key]
        else:
            if isinstance(bounds[0], int):
                val = random.randint(bounds[0], bounds[1])
            else:
                val = random.uniform(bounds[0], bounds[1])

        if isinstance(bounds[0], int):
            val = max(bounds[0], min(bounds[1], int(round(val))))
        else:
            val = max(bounds[0], min(bounds[1], round(val, 4)))

        individual.append(val)

    if fitness:
        individual.fitness.values = (
            fitness["profit"],
            fitness["sharpe"],
            -fitness["drawdown"],  
        )

    return individual

def initialize_population_from_partial_optimize(partial_entries, pop_size):
    locked_indexes = []  
    population = []

    for entry in partial_entries:
        individual = creator.Individual()
        locked = []
        for idx, (key, bounds) in enumerate(PARAM_BOUNDS.items()):
            if key in entry["parameters"]:
                val = entry["parameters"][key]
                locked.append(idx)
            else:
                if isinstance(bounds[0], int):
                    val = random.randint(bounds[0], bounds[1])
                else:
                    val = random.uniform(bounds[0], bounds[1])
            individual.append(val)
        population.append(individual)
        locked_indexes.append(set(locked))

    while len(population) < pop_size:
        idx = random.randint(0, len(population) - 1)
        individual = creator.Individual(population[idx])
        population.append(individual)
        locked_indexes.append(set(locked_indexes[idx]))

    return population, locked_indexes

def initialize_population_from_partial_fill(partial_entries, pop_size):
    population = []
    for entry in partial_entries:
        individual = creator.Individual()
        for key, bounds in PARAM_BOUNDS.items():
            if key in entry["parameters"]:
                val = entry["parameters"][key]
            else:
                if isinstance(bounds[0], int):
                    val = random.randint(bounds[0], bounds[1])
                else:
                    val = random.uniform(bounds[0], bounds[1])
            individual.append(val)
        population.append(individual)

    while len(population) < pop_size:
        population.append(toolbox.individual())

    return population

def initialize_population_from_pool(pool_entries, pop_size):
    population = []
    for entry in pool_entries:
        population.append(parameters_to_individual(entry))

    while len(population) < pop_size:
        population.append(toolbox.individual())
    
    return population

# ------------------------------------------------------------------------------ #
# ----------------------------- Evaluation Preview ----------------------------- #
# ------------------------------------------------------------------------------ #

def evaluate_individual_cv(individual, data):
    key = tuple(individual)  
    

    if key in fitness_cache:
        return fitness_cache[key]
    
    if hasattr(individual.fitness, "values") and individual.fitness.valid:
        print(f"Re-evaluating individual with known fitness: {individual.fitness.values}")
    
    try:
        parameters = {
            key: max(bounds[0], min(bounds[1], int(round(value)) if isinstance(bounds[0], int) else round(value, 4)))
            for (key, bounds), value in zip(PARAM_BOUNDS.items(), individual)
        }
        env = TradingEnvironment(parameters, data)
        env.run_episode()

        BARS_PER_DAY = int(24 * 60 / LOWER_TIMEFRAME_MIN)

        trade_log = env.trade_log
        total_trades = len(trade_log)
        min_required_trades = 50

        if total_trades < min_required_trades:
            return -9999, -9999, 9999  # Invalidate low trade strategies

        # Total profit (absolute)
        total_profit = sum(t['profit_abs'] for t in trade_log)

        # Return series: percentage returns per trade
        all_returns = np.array([t['profit_pct'] / 100 for t in trade_log])

        if len(all_returns) == 0:
            return -9999, -9999, 9999 

        # Calculate Winrate (for stats only)
        positive_trades = np.sum(all_returns > 0)
        winrate = (positive_trades / total_trades) * 100 if total_trades > 0 else 0

        mean_return = np.mean(all_returns)
        std_return = np.std(all_returns)
        downside_returns = all_returns[all_returns < 0]
        std_downside = np.std(downside_returns) if len(downside_returns) > 0 else 0

        if np.std(all_returns) < 0.0005:
            print("Penalizing weak individual (very low volatility).")
            return -9999, -9999, 9999

        # Sharpe Ratio (annualized)
        sharpe = (mean_return / std_return) * np.sqrt(BARS_PER_DAY) if std_return > 0 else 0

        # Max drawdown from cumulative equity curve
        equity_curve = np.cumsum(all_returns)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve)
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Multi-objective fitness (maximize profit, sharpe, minimize drawdown)
        fitness = (total_profit, sharpe, -max_drawdown)

        # Penalty for lower than ideal number of trades
        EXPECTED_TRADES = 70 
        trade_penalty_factor = np.round(min(1.0, total_trades / EXPECTED_TRADES), 2)

        if total_trades < EXPECTED_TRADES/3:
            return -9999, -9999, 9999

        if total_trades < EXPECTED_TRADES:
            penalty_multiplier = trade_penalty_factor  

            adjusted_profit = fitness[0]
            adjusted_sharpe = fitness[1]

            # Only penalize positive values
            if adjusted_profit > 0:
                adjusted_profit *= penalty_multiplier
            if adjusted_sharpe > 0:
                adjusted_sharpe *= penalty_multiplier

            fitness = (
                adjusted_profit,
                adjusted_sharpe,
                fitness[2]  # don't touch drawdown
            )

        sortino = (mean_return / std_downside) * np.sqrt(BARS_PER_DAY) if std_downside > 0 else 0  
        
        if total_profit > -1000:
            log_episode_to_file(parameters,
                            fitness,  
                            total_profit, sharpe, sortino, winrate,
                            EXPECTED_TRADES, total_trades, max_drawdown,
                            penalized_profit=fitness[0],
                            penalized_sharpe=fitness[1],
                            file_prefix=LOG_EPISODE_PREFIX)        

            save_top_individual_advanced(parameters, fitness, total_profit, sharpe, trade_penalty_factor, EXPECTED_TRADES, total_trades, filepath=STATIC_PARAM_FILE)

        fitness_cache[key] = fitness  
        return fitness
    
    except Exception as e:
        import traceback
        print("Exception in evaluate_individual_cv:")
        traceback.print_exc()
        return -9999, -9999, 9999  # Penalize the crash

# ------------------------------------------------------------------------------- #
# ------------------------------ Mutation Functions ----------------------------- #
# ------------------------------------------------------------------------------- #

def adjust_mutation_probability(gen, max_gen, stagnation_count=0):
    base_mutpb = 0.3
    max_mutpb = 0.8
    factor = stagnation_count / max_gen
    sigmoid = 1 / (1 + np.exp(-10 * (factor - 0.5)))  
    return base_mutpb + (max_mutpb - base_mutpb) * sigmoid

def introduce_random_benefactors(population, fraction=0.3):
    num_to_replace = int(len(population) * fraction)
    
    population.sort(key=lambda ind: ind.fitness.values, reverse=True)
    
    new_individuals = [toolbox.individual() for _ in range(num_to_replace)]
    population[-num_to_replace:] = new_individuals
    print(f"Introduced {num_to_replace} new random individuals (replaced weakest).")     

def mutate_and_clamp(individual, max_generations=gen_size, stagnation_count=0, locked_indexes=None):
    try:
        base_mutpb = 0.2
        max_mutpb = 0.8
        mutpb = base_mutpb + (stagnation_count / max_generations) * (max_mutpb - base_mutpb)
        mutpb = min(mutpb, max_mutpb)

        base_sigma = 0.1
        max_sigma = 1.0
        sigma = base_sigma + (stagnation_count / max_generations) * (max_sigma - base_sigma)
        sigma = min(sigma, max_sigma)

        rand = random.random()

        if stagnation_count > 0.6 * max_generations and random.random() < 0.1:
            print("Full random reset of an individual due to heavy stagnation.")
            new_ind = toolbox.individual()
            for i in range(len(individual)):
                if not locked_indexes or i not in locked_indexes:
                    individual[i] = new_ind[i]
            return individual

        if rand < 0.4:
            for idx in range(len(individual)):
                if not locked_indexes or idx not in locked_indexes:
                    if random.random() < mutpb:
                        individual[idx] += random.gauss(0, sigma)

        elif rand < 0.7:
            for idx, (key, bounds) in enumerate(PARAM_BOUNDS.items()):
                if not locked_indexes or idx not in locked_indexes:
                    if random.random() < mutpb:
                        individual[idx] = random.uniform(bounds[0], bounds[1])

        elif rand >= 0.7:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            if not locked_indexes or (idx1 not in locked_indexes and idx2 not in locked_indexes):
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

        # Clamp
        for idx, (key, bounds) in enumerate(PARAM_BOUNDS.items()):
            if isinstance(bounds[0], int):
                individual[idx] = max(bounds[0], min(bounds[1], int(round(individual[idx]))))
            else:
                individual[idx] = max(bounds[0], min(bounds[1], round(individual[idx], 4)))

        print(f"Mutation applied (mutpb: {mutpb:.2f}, sigma: {sigma:.2f})") 

        return individual

    except Exception as e:
        print(f"Error during mutation: {e}")



# ------------------------------------------------------------------------------- #
# ----------------------------- Crossover Functions ----------------------------- #
# ------------------------------------------------------------------------------- #

def crossover_and_clamp(ind1, ind2):

    tools.cxBlend(ind1, ind2, alpha=BLEND_ALPHA)
    def clamp_and_cast(value, bounds):
        if isinstance(bounds[0], int):
            return max(bounds[0], min(bounds[1], int(round(value))))
        else:
            return max(bounds[0], min(bounds[1], round(value, 2)))

    for idx, (key, bounds) in enumerate(PARAM_BOUNDS.items()):
        ind1[idx] = clamp_and_cast(ind1[idx], bounds)
        ind2[idx] = clamp_and_cast(ind2[idx], bounds)

    return ind1, ind2

# Register evaluation function
toolbox.register("evaluate", evaluate_individual_cv)

# Genetic operators
toolbox.register("mate", crossover_and_clamp) 
toolbox.register("mutate", mutate_and_clamp, current_generation=None, max_generations=gen_size)
toolbox.register("select", tools.selTournament, tournsize=tournament_size)


# ------------------------------------------------------------------------------- #
# -------------------------------GA Runner Function------------------------------ #
# ------------------------------------------------------------------------------- #


def run_ga(data, pool, static_pool_file):
    locked_indexes_list = None  

    if PARTIAL_GENOME:
        partial_entries = load_partial_genomes(filepath=PARTIAL_GENOME_FILE)
        if PARTIAL_FILL_MODE == "fill":
            population = initialize_population_from_partial_fill(partial_entries, pop_size)
        elif PARTIAL_FILL_MODE == "optimize":
            population, locked_indexes_list = initialize_population_from_partial_optimize(partial_entries, pop_size)
        else:
            raise ValueError("Invalid PARTIAL_FILL_MODE setting.")
        print(f"Loaded partial genomes and initialized population ({len(population)}).")
    elif static_pool_file is not None:
        top_params = load_top_n_parameters(n=TOP_EPISODES_LOAD_COUNT, filepath=static_pool_file)
        print(f"Loaded {len(top_params)} individuals from static file: {static_pool_file}")
        population = initialize_population_from_pool(top_params, pop_size)
        if RERUN_TOP_PARAMS:
            print("Rerun flag active — clearing all fitnesses.")
            for ind in population:
                if hasattr(ind.fitness, "values"):
                    del ind.fitness.values
    else:
        population = toolbox.population(n=pop_size)
        print("Initialized fresh random population.")

    ngen = gen_size
    cxpb = cross_prob
    mutpb = mut_prob
    elite_size = eli_size

    best_fitness = (-float("inf"), -float("inf"), float("inf"))
    stagnation_count = 0

    gen = 0
    while gen < ngen:

        while True:
            fitnesses = evaluate_population(population, data, pool)
            if fitnesses is not None:                 # Success!
                break

            print("Evaluation timed-out, rebooting pool (generation unchanged)")
            pool.terminate(); pool.join()
            pool = multiprocessing.Pool(num_processes)
            # Don't reset population and just reset pool if child process crashed 

        elite   = list(map(toolbox.clone, tools.selBest(population, elite_size)))
        offspring = toolbox.select(population, len(population) - elite_size)
        offspring = list(map(toolbox.clone, offspring))

        # Crossover 
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values; del c2.fitness.values

        # Mutation 
        for idx, mut in enumerate(offspring):
            if random.random() < mutpb + stagnation_count / ngen:
                if PARTIAL_GENOME and PARTIAL_FILL_MODE == "optimize":
                    locked = locked_indexes_list[idx % len(locked_indexes_list)]
                    mutate_and_clamp(mut, gen, ngen, stagnation_count,
                                    locked_indexes=locked)
                else:
                    mutate_and_clamp(mut, gen, ngen, stagnation_count)
                del mut.fitness.values

        # New population
        population[:] = elite + offspring

        # Metrics & logging 
        current_best = max(ind.fitness.values for ind in population)
        print(f"Generation {gen}: best = {tuple(round(x, 4) for x in current_best)}")

        if current_best > best_fitness:
            best_fitness = current_best
            stagnation_count = 0
            print("Positive Improvement.")
        else:
            stagnation_count += 1
            print(f"No improvement. Retry is no longer implemented.")

        gen += 1                     

    return tools.selBest(population, k=1)[0], pool

# -------------------------------------------------------------------------------#
# ---------------------------------- Validation ---------------------------------#
# -------------------------------------------------------------------------------#

def process_validation_line(args):
    line, val_data, DATA_FILE, VALIDATION_DATA_FILE, LOWER_TIMEFRAME_MIN, MIN_PROFIT_THRESHOLD = args
    try:
        param_data = json.loads(line)
        true_fitness = param_data.get("true_fitness", {})
        profit = true_fitness.get("profit", -float('inf'))

        if profit < MIN_PROFIT_THRESHOLD:
            return None  # Skip low-profit lines

        parameters = param_data.get("parameters")
        BARS_PER_DAY = int(24 * 60 / LOWER_TIMEFRAME_MIN)
        min_required_trades = 10

        val_env = TradingEnvironment(parameters, val_data)
        val_env.run_episode()

        trade_log = val_env.trade_log
        total_trades = len(trade_log)

        if total_trades < min_required_trades:
            val_profit, val_sharpe, val_drawdown = -9999, -9999, 9999
        else:
            all_returns = np.array([t['profit_pct'] / 100 for t in trade_log])
            if len(all_returns) == 0 or np.std(all_returns) < 0.0005:
                val_profit, val_sharpe, val_drawdown = -9999, -9999, 9999
            else:
                val_profit = sum(t['profit_abs'] for t in trade_log)
                mean_return = np.mean(all_returns)
                std_return = np.std(all_returns)
                sharpe = (mean_return / std_return) * np.sqrt(BARS_PER_DAY) if std_return > 0 else 0
                equity_curve = np.cumsum(all_returns)
                peak = np.maximum.accumulate(equity_curve)
                drawdown = peak - equity_curve
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

                val_sharpe = sharpe
                val_drawdown = max_drawdown

        result = {
            "data_file": DATA_FILE,
            "validation_data_file": VALIDATION_DATA_FILE,
            "true_fitness": true_fitness,
            "validation_fitness": {
                "profit": val_profit,
                "sharpe": val_sharpe,
                "drawdown": val_drawdown
            },
            "parameters": parameters
        }
        return json.dumps(result)

    except Exception as e:
        print(f"[ERROR] Exception in line:\n{line[:100]}")
        traceback.print_exc()
        return None


def validate_all_episodes(STATIC_PARAM_FILE, VALIDATION_DATA_FILE, DATA_FILE, pool):
    val_data = load_data_numpy(VALIDATION_DATA_FILE)

    output_dir = "Validation"
    os.makedirs(output_dir, exist_ok=True)

    MIN_PROFIT_THRESHOLD = 1200
    output_file_name = f'Validation_{STATIC_PARAM_FILE}'
    output_path = os.path.join(output_dir, output_file_name)

    with open(STATIC_PARAM_FILE, 'r') as file:
        lines = file.readlines()

    args_list = [
        (line, val_data, DATA_FILE, VALIDATION_DATA_FILE, LOWER_TIMEFRAME_MIN, MIN_PROFIT_THRESHOLD)
        for line in lines
    ]

    results = pool.map(process_validation_line, args_list)
    pool.close()
    pool.join()

    with open(output_path, 'w') as out_file:
        for res in results:
            if res:
                out_file.write(res + '\n')

# -------------------------------------------------------------------------------#
# ---------------------------------Main Execution--------------------------------#
# -------------------------------------------------------------------------------#

if __name__ == "__main__":
    
    if OPTIMIZATION:
        print("Standard optimization mode initiated.")
        data = load_data_numpy(DATA_FILE)

        sort_top_episodes_file(STATIC_PARAM_FILE)
        pool = multiprocessing.Pool(processes=num_processes)

        try:
            best_individual, pool = run_ga(data, pool, static_pool_file=STATIC_PARAM_FILE)
        finally:
            pool.close()     
            pool.join()    
            print("Pool closed and joined successfully.")
        
        sort_top_episodes_file(STATIC_PARAM_FILE)
        deduplicate_top_episodes(STATIC_PARAM_FILE)
        
        parameters = {
            key: max(bounds[0], min(bounds[1], round(best_individual[idx], 2) if isinstance(bounds[0], float) else int(best_individual[idx])))
            for idx, (key, bounds) in enumerate(PARAM_BOUNDS.items())
        }
    
        print("Best parameters:", parameters)
    else:
        print("Validation only mode. No optimization will be initiated.")


    # ----------------------------- #
    #     Run Best Individual       #
    # ----------------------------- #

    if SINGLE_VALIDATION:
        data = load_data_numpy(VALIDATION_DATA_FILE)
        with open(STATIC_PARAM_FILE, 'r') as file:
            first_line = file.readline()
            param_data = json.loads(first_line)

        parameters = param_data.get("parameters")


    if POST_VALIDATION:
        sort_top_episodes_file(STATIC_PARAM_FILE)
        deduplicate_top_episodes(STATIC_PARAM_FILE)
                
        pool = multiprocessing.Pool(processes=num_processes)
        
        validate_all_episodes(STATIC_PARAM_FILE, VALIDATION_DATA_FILE, DATA_FILE, pool)

    if RUN_TOP_EPISODE:
        data = load_data_numpy(DATA_FILE)
        with open(STATIC_PARAM_FILE, 'r') as file:
            first_line = file.readline()
            param_data = json.loads(first_line)

        parameters = param_data.get("parameters")
        print(f"Manual Evaluation Parameters: {parameters}")
        
    env = TradingEnvironment(parameters, data)
    env.run_episode()
    

    trade_log = env.trade_log

    # Calculate returns
    all_returns = np.array([t['profit_pct'] / 100 for t in trade_log])
    total_profit = sum(t['profit_abs'] for t in trade_log)
    equity_curve = np.cumsum(all_returns)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = peak - equity_curve
    max_drawdown = np.max(drawdown)

    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    
    BARS_PER_DAY = int(24 * 60 / LOWER_TIMEFRAME_MIN)
    sharpe = (mean_return / std_return) * np.sqrt(BARS_PER_DAY) if std_return > 0 else 0

    # ----------------------------- #
    #      Performance Plotting     #
    # ----------------------------- #

    print("Profit: " + str(total_profit))
    print("Sharpe: " + str(sharpe))
    print("Max drawdown: " + str(max_drawdown))

    # Calculate max loss per single trade for long and short positions
    max_long_loss_pct = min(t['profit_pct'] for t in trade_log if t['position_type'] == 'long')
    max_short_loss_pct = min(t['profit_pct'] for t in trade_log if t['position_type'] == 'short')
    print(f"Max long loss (%): {max_long_loss_pct:.2f}%, Max short loss (%): {max_short_loss_pct:.2f}%")

    # Separate trades by type and outcome
    long_trades = [t for t in trade_log if t['position_type'] == 'long']
    short_trades = [t for t in trade_log if t['position_type'] == 'short']

    long_profits = [t['profit_pct'] for t in long_trades if t['profit_pct'] > 0]
    long_losses = [t['profit_pct'] for t in long_trades if t['profit_pct'] < 0]

    short_profits = [t['profit_pct'] for t in short_trades if t['profit_pct'] > 0]
    short_losses = [t['profit_pct'] for t in short_trades if t['profit_pct'] < 0]

    # Compute averages with safe defaults
    avg_long_profit = np.mean(long_profits) if long_profits else 0
    avg_long_loss = np.mean(long_losses) if long_losses else 0
    avg_short_profit = np.mean(short_profits) if short_profits else 0
    avg_short_loss = np.mean(short_losses) if short_losses else 0

    # Print results
    print(f"Avg long profit: {avg_long_profit:.2f}%, Avg long loss: {avg_long_loss:.2f}%")
    print(f"Avg short profit: {avg_short_profit:.2f}%, Avg short loss: {avg_short_loss:.2f}%")

    # Count how many trades were liquidated
    liquidations = sum(1 for t in trade_log if t['profit_pct'] < -env.leverage)
    print(f"Liquidations for x{env.leverage} leverage: {liquidations }")

    # Separate wins and losses
    wins = all_returns[all_returns > 0]
    losses = all_returns[all_returns < 0]

    # Calculate P(W) and P(L)
    total_trades = len(all_returns)
    p_win = len(wins) / total_trades if total_trades > 0 else 0
    p_loss = 1 - p_win

    # Calculate average reward and risk
    avg_reward = (np.mean(wins) if len(wins) > 0 else 0) 
    avg_risk = (abs(np.mean(losses)) if len(losses) > 0 else 0) 

    # Expected Value
    ev = p_win * avg_reward - p_loss * avg_risk

    # Output result
    print(f"Win Rate (P(W)): {p_win:.2%}")
    print(f"Loss Rate (P(L)): {p_loss:.2%}")
    print(f"Avg Reward: {avg_reward:.2%}")
    print(f"Avg Risk: {avg_risk:.2%}")
    print(f"Expected Value (EV): {ev*100:.4f} ({ev * 100:.2f}%)")

    performance_data = {
        "equity_curve": equity_curve,
        "drawdown": drawdown,
        "sharpe_ratio": sharpe,
        "parameters": parameters
    }

    # Group trades
    trades_by_type = defaultdict(list)
    for t in trade_log:
        trades_by_type[t["trade_type"]].append(t)

    print("\n── Trade-type breakdown ──")
    for trade_type, trades in trades_by_type.items():
        if not trades:
            continue
        
        profits = [t["profit_pct"] for t in trades if t["profit_pct"] > 0]
        losses  = [t["profit_pct"] for t in trades if t["profit_pct"] < 0]

        avg_profit = np.mean(profits) if profits else 0
        avg_loss   = np.mean(losses) if losses else 0
        max_loss   = min(losses) if losses else 0
        win_rate   = len(profits) / len(trades) if trades else 0
        loss_rate  = 1 - win_rate

        avg_reward = np.mean(profits) if profits else 0
        avg_risk   = abs(np.mean(losses)) if losses else 0
        ev         = win_rate * avg_reward - loss_rate * avg_risk

        print(f"\nTrade Type: {trade_type}")
        print(f"  Total Trades: {len(trades)}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Avg Profit: {avg_profit:.2f}%")
        print(f"  Avg Loss: {avg_loss:.2f}%")
        print(f"  Max Loss: {max_loss:.2f}%")
        print(f"  Avg Reward: {avg_reward:.2f}%")
        print(f"  Avg Risk: {avg_risk:.2f}%")
        print(f"  Expected Value: {ev*100:.2f}%")



    # --- Plot using unpacked dictionary values
    if PLOT_TOP_EPISODE:
        plot_performance_graph(
            trade_log=trade_log,
            profits=performance_data["equity_curve"].tolist(),
            drawdowns=performance_data["drawdown"].tolist(),
            sharpes=[performance_data["sharpe_ratio"]] * len(performance_data["equity_curve"]),
            TOKEN=ENV_TOKEN
        )

        fig_main = go.Figure()
        plot_main_candlestick_with_indicators(data, env.trade_log, env.graph_log)


