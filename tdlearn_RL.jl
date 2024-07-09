using POMDPs, POMDPModelTools, QuickPOMDPs

# solvers
using DiscreteValueIteration, TabularTDLearning
# Temporal Difference (TD) Learning

# policy
using POMDPPolicies

using Random


# Define State data type (1-D space)

struct State
    x::Int 
end

# Define Action data type

@enum Action LEFT RIGHT


# Define State Space

null = State(-1)

S = [[State(x) for x = 1:7]..., null]


# Define Action Space

A = [LEFT, RIGHT]


# Define Transition Function

const MOVEMENTS = Dict(
    LEFT => State(-1),
    RIGHT => State(1)
)

Base.:+(s1::State,s2::State) = State(s1.x + s2.x)

function T(s::State, a::Action)
    # Deterministic() from POMDPModelTools.jl
    if R(s) != 0
        return Deterministic(null)
    end
    
    # Initialise Variables (index 1 is current state)
    len_a = length(A)
    next_states = Vector{State}(undef, len_a + 1)
    probabilities = zeros(len_a + 1)

    # enumerate() from Julia Base.Iterators
    for (index, a_prime) ∈ enumerate(A)
        prob = (a_prime == a) ? 0.8 : 0.2 # makes correct decision 80% of the time
        dest = s + MOVEMENTS[a_prime]
        next_states[index + 1] = dest

        if 1 <= dest.x <= 7
            probabilities[index + 1] += prob
        end

    end

    # handle out-of-bounds transitions
    next_states[1] = s 
    probabilities[1] = 1 - sum(probabilities)

    # SparseCat from POMDPModelTools.jl
    return SparseCat(next_states,probabilities)

end


# Define Reward Function

function R(s, a = missing)
    if s == State(1)
        return -1
    elseif s == State(7)
        return 1
    end
    return 0
end


# Set Discount Factor

γ = 0.95


# prep mdp

termination(s::State) = s == null


abstract type GridWorld <: MDP{State,Action} end


############### Value Iteration Algorithm

# define mdp using QuickPOMDPs.jl

mdp = QuickMDP(GridWorld,
    states = S,
    actions = A,
    transition = T,
    reward = R,
    discount = γ,
    isterminal = termination

)

# select solver 

solver = ValueIterationSolver(max_iterations = 30)

# solve mdp

policy = solve(solver, mdp)

"""
ValueIterationPolicy:
State(1) -> LEFT
State(2) -> RIGHT
State(3) -> RIGHT
State(4) -> RIGHT
State(5) -> RIGHT
State(6) -> RIGHT
State(7) -> LEFT
State(-1) -> LEFT
"""

value_view = [S policy.util]

"""
8×2 Matrix{Any}:
State(1)   -1.0
State(2)    0.292242
State(3)    0.634798
State(4)    0.762302
State(5)    0.844359
State(6)    0.920428
State(7)    1.0
State(-1)   0.0
"""


#### SARSA Algorithm

# agent figures out the policy itself

# define mdp using QuickPOMDPs.jl

s_mdp = QuickMDP(GridWorld,
    states = S,
    actions = A,
    transition = T,
    reward = R,
    discount = γ,
    initialstate = S, #### NEW LINE
    isterminal = termination
)

# Temporal Difference (TD) Learning, learns by sampling the environment

# select solver from TabularTDlearning.jl
# select policy from POMDPPolicies.jl

Random.seed!(1)

s_α = 0.9

s_n_episodes = 15 # try 1,10,15

s_solver = SARSASolver(
    n_episodes = s_n_episodes,
    learning_rate = s_α,
    exploration_policy = EpsGreedyPolicy(s_mdp,0.5), # 50% follow action mapped to the state
    verbose = false

)

# SARSA = State, Action, Reward, State, Action
# On-policy learning algorithm, updating policy at every step

# Q(S,A) with Reward value R ===> Q(S',A') Q value is then re-evaluated
# Agent ignores transition function but follows Epsilon Greedy Policy
# ϵ = probability of agent going on some random direction, 
# 1-ϵ = probability of agent following direction which leads to highest rewards according to latest info


# solve mdp

s_policy = solve(s_solver, s_mdp)

"""
s_n_episodes = 1

ValuePolicy{QuickMDP{GridWorld, State, Action, @NamedTuple{stateindex::Dict{State, Int64}, isterminal::typeof(termination), actionindex::Dict{Action, Int64}, transition::typeof(T), reward::typeof(R), states::Vector{State}, actions::Vector{Action}, discount::Float64, initialstate::Vector{State}}}, Matrix{Float64}, Action}:
State(1) -> LEFT
State(2) -> LEFT
State(3) -> LEFT
State(4) -> LEFT
State(5) -> LEFT
State(6) -> LEFT
State(7) -> LEFT
State(-1) -> LEFT
"""


"""
s_n_episodes = 10

ValuePolicy{QuickMDP{GridWorld, State, Action, @NamedTuple{stateindex::Dict{State, Int64}, isterminal::typeof(termination), actionindex::Dict{Action, Int64}, transition::typeof(T), reward::typeof(R), states::Vector{State}, actions::Vector{Action}, discount::Float64, initialstate::Vector{State}}}, Matrix{Float64}, Action}:
State(1) -> LEFT
State(2) -> LEFT
State(3) -> LEFT
State(4) -> RIGHT
State(5) -> LEFT
State(6) -> LEFT
State(7) -> LEFT
State(-1) -> LEFT
"""


"""
s_n_episodes = 15

ValuePolicy{QuickMDP{GridWorld, State, Action, @NamedTuple{stateindex::Dict{State, Int64}, isterminal::typeof(termination), actionindex::Dict{Action, Int64}, transition::typeof(T), reward::typeof(R), states::Vector{State}, actions::Vector{Action}, discount::Float64, initialstate::Vector{State}}}, Matrix{Float64}, Action}:
State(1) -> LEFT
State(2) -> RIGHT
State(3) -> RIGHT
State(4) -> RIGHT
State(5) -> RIGHT
State(6) -> RIGHT
State(7) -> LEFT
State(-1) -> LEFT

"""



#### Q-Learning Algorithm


q_mdp = QuickMDP(GridWorld,
    states = S,
    actions = A,
    transition = T,
    reward = R,
    discount = γ,
    initialstate = S, #### NEW LINE
    isterminal = termination
)

# Temporal Difference (TD) Learning, learns by sampling the environment

# select solver from TabularTDlearning.jl
# select policy from POMDPPolicies.jl

Random.seed!(1)

q_α = 0.9

q_n_episodes = 20 # try 1,10,15,20

q_solver = QLearningSolver(
    n_episodes = q_n_episodes,
    learning_rate = q_α,
    exploration_policy = EpsGreedyPolicy(q_mdp,0.5), # 50% follow action mapped to the state
    verbose = false

)

# Off-Policy Learning
# Agent is following some other policy and using that knowledge to learn the optimal way to behave in its environment
# Movement based on exploration / wants to explore environment
# Updates based on exploitation / reward
# Q-learning may take longer to converge


# solve mdp

q_policy = solve(q_solver, q_mdp)

"""
q_n_episodes = 1

ValuePolicy{QuickMDP{GridWorld, State, Action, @NamedTuple{stateindex::Dict{State, Int64}, isterminal::typeof(termination), actionindex::Dict{Action, Int64}, transition::typeof(T), reward::typeof(R), states::Vector{State}, actions::Vector{Action}, discount::Float64, initialstate::Vector{State}}}, Matrix{Float64}, Action}:
State(1) -> LEFT
State(2) -> LEFT
State(3) -> LEFT
State(4) -> LEFT
State(5) -> LEFT
State(6) -> LEFT
State(7) -> LEFT
State(-1) -> LEFT
"""

"""
q_n_episodes = 10

ValuePolicy{QuickMDP{GridWorld, State, Action, @NamedTuple{stateindex::Dict{State, Int64}, isterminal::typeof(termination), actionindex::Dict{Action, Int64}, transition::typeof(T), reward::typeof(R), states::Vector{State}, actions::Vector{Action}, discount::Float64, initialstate::Vector{State}}}, Matrix{Float64}, Action}:
State(1) -> LEFT
State(2) -> RIGHT
State(3) -> LEFT
State(4) -> LEFT
State(5) -> LEFT
State(6) -> LEFT
State(7) -> LEFT
State(-1) -> LEFT
"""

"""
q_n_episodes = 15

ValuePolicy{QuickMDP{GridWorld, State, Action, @NamedTuple{stateindex::Dict{State, Int64}, isterminal::typeof(termination), actionindex::Dict{Action, Int64}, transition::typeof(T), reward::typeof(R), states::Vector{State}, actions::Vector{Action}, discount::Float64, initialstate::Vector{State}}}, Matrix{Float64}, Action}:
State(1) -> LEFT
State(2) -> RIGHT
State(3) -> RIGHT
State(4) -> RIGHT
State(5) -> LEFT
State(6) -> RIGHT
State(7) -> LEFT
State(-1) -> LEFT
"""

"""
q_n_episodes = 20

ValuePolicy{QuickMDP{GridWorld, State, Action, @NamedTuple{stateindex::Dict{State, Int64}, isterminal::typeof(termination), actionindex::Dict{Action, Int64}, transition::typeof(T), reward::typeof(R), states::Vector{State}, actions::Vector{Action}, discount::Float64, initialstate::Vector{State}}}, Matrix{Float64}, Action}:
State(1) -> RIGHT
State(2) -> RIGHT
State(3) -> RIGHT
State(4) -> RIGHT
State(5) -> RIGHT
State(6) -> RIGHT
State(7) -> LEFT
State(-1) -> LEFT
"""