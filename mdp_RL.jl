## Building an MDP Framework (S,A,T,R,γ)
# (States, Actions, Transition, Reward, Discount)
# Framework for sequential decision making under uncertainty


using POMDPs, POMDPModelTools, QuickPOMDPs

## solver in different package

using DiscreteValueIteration


#define State data type (ie. coordinates)

struct State
    x::Int
    y::Int
end

# define Action data type (@enum)

@enum Action UP DOWN LEFT RIGHT

Action


# define State Space

null = State(-1,-1)

S = [
    [State(x,y) for x = 1:4, y = 1:3]..., null
]


# define Action Space

A = [UP,DOWN,LEFT,RIGHT]


# define Transition Function

const MOVEMENTS = Dict(
    UP => State(0,1),
    DOWN => State(0,-1),
    LEFT => State(-1,0),
    RIGHT => State(1,0)
)

Base.:+(s1::State, s2::State) = State(
    s1.x + s2.x, s1.y + s2.y
)

function T(s::State, a::Action)
    # Deterministic() from POMDPModelTools.jl
    if R(s) != 0
        return Deterministic(null)
    end

    # Initialise variables (index 1 is current state)
    len_a = length(A)
    next_states = Vector{State}(undef, len_a + 1)
    probabilities = zeros(len_a + 1)

    # enumerate() from Julia Base.Iterators
    for (index, a_prime) in enumerate(A)

        prob = (a_prime == a) ? 0.7 : 0.1 # 70% chance agent will move in the "right" direction
        dest = s + MOVEMENTS[a_prime]
        next_states[index + 1] = dest

        if dest.x == 2 && dest.y == 2 # locks the agent out of State/position (2,2) (obstacle)
            probabilities[index + 1] = 0
        elseif 1 <= dest.x <= 4 && 1 <= dest.y <= 3 # assigns probabilities within the borders
            probabilities[index + 1] += prob
        end

    end

    # handle out-of-bounds transitions

    next_states[1] = s
    probabilities[1] = 1 - sum(probabilities)

    # SparseCat from POMDPModelTools.jl
    return SparseCat(next_states,probabilities)

end


# define Reward Function

function R(s, a = missing)
    if s == State(4,2)
        return -100
    elseif s == State(4,3)
        return 10
    end

    return 0
end


# set Discount Factor

gamma = 0.95 # places higher value on results soon, and lesser value on future results


# Define MDP using QuickPOMDPs.jl

termination(s::State) = s == null

abstract type GridWorld <: MDP{State,Action} end

mdp = QuickMDP(GridWorld,
    states = S,
    actions = A,
    transition = T,
    reward = R,
    discount = gamma,
    isterminal = termination
)


# select solver from DiscreteValueIteration.jl

solver = ValueIterationSolver(max_iterations = 30)

# solve mdp

policy = solve(solver,mdp)

"""
ValueIterationPolicy:
State(1, 1) -> UP
State(2, 1) -> LEFT
State(3, 1) -> LEFT
State(4, 1) -> LEFT
State(1, 2) -> UP
State(2, 2) -> UP
State(3, 2) -> UP
State(4, 2) -> UP
State(1, 3) -> RIGHT
State(2, 3) -> RIGHT
State(3, 3) -> RIGHT
State(4, 3) -> UP
State(-1, -1) -> UP
"""


# view values (utility)

value_view = [S policy.util]

# Result is a Policy that maps each State to an Action

"""
13×2 Matrix{Any}:
State(1, 1)       5.39248
State(2, 1)       4.64305
State(3, 1)       1.84261
State(4, 1)     -10.2158  # bad position to be in
State(1, 2)       5.90541
State(2, 2)       5.19417
State(3, 2)      -4.73097  # has negative value as 10% chance to fall into (4,2) which is -100
State(4, 2)    -100.0      # death
State(1, 3)       6.42285
State(2, 3)       6.97975
State(3, 3)       7.58412
State(4, 3)      10.0
State(-1, -1)     0.0
"""