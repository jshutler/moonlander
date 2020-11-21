# Name:         
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Moonlander II
# Term:         Fall 2020

import random
from typing import Callable
# from pprint import pprint


class ModuleState:  # do not modify class

    def __init__(self, fuel: int, altitude: int, velocity: float, gforce: float,
                 transition: Callable[[float, float], float]) -> None:
        self.rate_max = 4
        self.fuel: int = fuel
        self.altitude: int = altitude
        self.velocity: int = velocity
        self.use_fuel: Callable[[int], ModuleState] = \
            lambda rate: ModuleState._use_fuel(fuel, altitude, velocity, gforce,
                                               transition, self.rate_max, rate)
        self.actions: Tuple[float, ...] = tuple(range(self.rate_max + 1))

    @staticmethod
    def _use_fuel(fuel: int, altitude: int, velocity: float, gforce: float,
                  transition: Callable[[float, int], float], rate_max: int,
                  rate: int) -> "ModuleState":
        fuel = max(0, fuel - rate)
        if fuel == 0:
            rate = 0
        acceleration = transition(gforce * 9.8, rate / rate_max)
        altitude = max(0, altitude + velocity + acceleration / 2)
        velocity += acceleration
        return ModuleState(fuel, altitude, velocity, gforce, transition)




def main() -> None:
    transition = lambda g, r: g * (2 * r - 1)  # example transition function
    state = ModuleState(1000, 20, 0.0, 0.1657, transition)
    pilot(state, True)


def pilot(state: ModuleState, auto: bool = True) -> None:
    if auto:
        q = learn_q(state)
        #argmax, selcts action 
        policy = lambda s: max(state.actions, key=lambda a: q(s, a))
    while state.altitude > 0:
        if auto:
            rate = policy(state)
        else:
            while True:
                rate = int(input(f"Enter Fuel Rate [0-{state.rate_max}]: "))
                if rate in state.actions:
                    break
        state = state.use_fuel(rate)
        print(f"    Fuel: {state.fuel:4} l")
        print(f"Altitude: {state.altitude:7.2f} m")
        print(f"Velocity: {state.velocity:7.2f} m/s\n")
    print(f"Impact Velocity: {state.velocity:7.2f} m/s")

class q_table:
    def __init__(self, init_state):
        self.init_state = init_state
        self.altitude = init_state.altitude
        self.velocity = init_state.velocity
        self.max_altitutde = int(self.altitude + self.altitude * 2)


        #creating the table

        altitudes = list(range(0, self.max_altitutde + 1))
        
        #each individual value of the velocity is really a bucket
        #indicating a speed range
        #0 - faster than -5
        #1 - between -1 and -5
        #2 - between 0 and -1
        #3 - greater than 0
        velocities = list(range(0, 4))

        keys = []
        for alt in altitudes:
            for vel in velocities:
                keys.append((alt, vel))


        min_starting_value = 0
        max_starting_value = 0
        self.table = \
        {key: [random.uniform(min_starting_value, max_starting_value) \
        for _ in range(5)] for key in keys}
    

    def __call__(self, state: ModuleState, action: int, learn=False):
        #makes our function callable. will return us the value in the 
        #table
        discrete_state = self.get_discrete(state)
        random_choice = random.uniform(0,1)
        possible_actions = list(range(5))
        if discrete_state[0] >= self.max_altitutde:
            discrete_state = (self.max_altitutde, discrete_state[1])

        if random_choice <= .1 and learn:
            # print('hit')
            #removes current action
            possible_actions.remove(action)
            #chooses from random choices
            action = random.choices(possible_actions)[0]
            # print('hit')
            return self.table[discrete_state][action]

        return self.table[discrete_state][action]




    def get_discrete(self, state: ModuleState):
        altitude = int(state.altitude)

        if state.velocity <= -5:
            velocity = 0

        elif state.velocity < -1:
            velocity = 1

        elif state.velocity <= 0:
            velocity = 2

        else:
            velocity = 3

        return (altitude, velocity)
            
def bellman(learning_rate, discount_rate, current_q, reward, future_q):
    return (1 - learning_rate) * current_q + \
    learning_rate * (reward + discount_rate*future_q)

def argmax(iterable):
    return iterable.index(max(iterable))

def get_reward(state):
    #winning base case
    if state.altitude <= 0 and (state.velocity < 0 and state.velocity > -1):
        # print('hit')
        return 1000
    if state.altitude <= 0 and state.velocity < -1:
        # print('hit')
        return -10

    alt_w = -1
    vel_w = -.5
    fuel_w = 0



    if state.velocity < 0 and state.velocity > -1:
        return alt_w*state.altitude + abs(state.velocity) * 5

    return state.altitude * alt_w + \
    abs(state.velocity) * vel_w + \
    +vel_w * abs(state.velocity) ** 2 +\
    state.fuel * fuel_w

def learn_q(state: ModuleState) -> Callable[[ModuleState, int], float]:
    """
    Return a Q-function that maps a state-action pair to a utility value. This
    function must be a callable with the signature shown.
    """
    actions = [0, 1, 2, 3 , 4]
    learning_rate = .2
    discount_rate = .5

    epochs = 10000

    successes = 0
    failures = 0


    init_state = state
    
    #initialize our table
    q = q_table(state)
    
    print(q.table)
    for epoch in range(epochs):
        state = init_state
        #start of main loop
        while state.altitude > 0:
        
            discrete_state = q.get_discrete(state)

            #if our discrete state is too high, we are goign to force it down
            if discrete_state[0] >= q.max_altitutde:
                discrete_state = (q.max_altitutde, discrete_state[1])
                action = 0
            else:
               action = argmax(q.table[discrete_state])

            #getting our q from our original 
            current_q = q(state, action, learn=True)

            #getting our new state
            state = state.use_fuel(action)
            #our new discreet state
            new_discreet_state = q.get_discrete(state)
            
            #if the iteration is not done
            if state.altitude > 0:
                #get reward
                #get new max q
                #update q, using belman
               
                #this reward will be based on the new state we got from using fuel
                reward = get_reward(state)

                #making sure value is in table
                if state.altitude >= q.max_altitutde:
                    new_discreet_state = (q.max_altitutde, new_discreet_state[1])
                    new_action = 0
                else:
                    new_action = argmax(q.table[new_discreet_state])
                
                best_future_q = q(state, new_action, learn=True)
                b = \
                bellman(learning_rate, discount_rate, \
                    current_q, reward, best_future_q)
                # print(b)
                q.table[discrete_state][action] = b
                # print(reward)
            #if our game is over
            else:
                reward = get_reward(state) #should be 100 or -100
                #updating for final reward
                q.table[discrete_state][action] = reward 
                # print('hit')

        if state.velocity < 0 and state.velocity > -1:
            print(state.velocity)
            successes += 1
            print("success")
        else:
            failures += 1
            print("failure")
            print(state.velocity)

    # pprint(q.table)
    print(f'successes: {successes}')
    print(f'failures: {failures}')
    return q

if __name__ == "__main__":
    main()
    # transition = lambda g, r: g * (2 * r - 1)  # example transition function
    # state = ModuleState(1000, 20, 0.0, 0.1657, transition)
    # q = q_table(state)

    # q.table[(20, 2)][1] = 32

    # learn_q(state)

    