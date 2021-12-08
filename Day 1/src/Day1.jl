module Day1

import Random
import Distributions
import Statistics

export world_evolution

function toroid_neighbors(
    selection::Tuple{Int, Int},
    world_size::Tuple{Int, Int}
)::NTuple{5, Tuple{Int, Int}}
    """
        toroid_neighbors(selection, world_size)
    
    Compute nearest vertical and horizontal neighbors to a
    point taking into account toroidal boundary conditions
    on the world.

    # Arguments
    - `selection`: A tuple (i, j) representing the selected location
    - `world_size`: A tuple (M, N) representing the dimensions of the
       toroidal world.
    """
    i, j = selection
    M, N = world_size
    base_selection = (
        (i - 1, j),
        (i, j - 1),
        (i, j),
        (i, j + 1),
        (i + 1, j)
    )
    
    mod_sel = map(
        x -> (x[1] % M, x[2] % N),
        base_selection
    )
    complement = map(
        x -> (
            x[1] > 0 ? x[1] : M + x[1],
            x[2] > 0 ? x[2] : N + x[2]
        ),
        mod_sel
    )
    return complement
end

function start_world(
    world_size::Tuple{Int, Int},
    prob_round::Number,
    seed::Int
)::Matrix{Bool}
    Random.seed!(seed)
    prob_matrix = prob_round * ones(world_size)
    return map(rand, map(Distributions.Bernoulli, prob_matrix))
end


function vote_results(
    prior_state::Matrix{Bool},
    citizen_coord::Tuple{Int, Int}
)
    selection_elements = toroid_neighbors(
        citizen_coord, size(prior_state)
    )
    vote_sum = sum([prior_state[i, j] for (i, j) in selection_elements])
    vote_result = (vote_sum >= 3) ? 1 : 0
    
    new_state = copy(prior_state)
    for (i, j) in selection_elements
        new_state[i, j] = vote_result
    end
    return new_state
end


function evolve_world(starting_state::Matrix{Bool}, seed::Int)::Matrix{Bool}
    Random.seed!(seed)
    new_state = starting_state
    world_shape = size(starting_state)
    citizen_pick = (
        Random.rand(1:world_shape[1]),
        Random.rand(1:world_shape[2])
    )
    new_state = vote_results(starting_state, citizen_pick)
end


function track_average_opinion(
    starting_state::Matrix{Bool},
    tfinal::Int
)::Vector{Float64}
    mean_opinions = [Statistics.mean(starting_state)]
    new_state = starting_state
    for t_seed in 1:tfinal
        new_state = evolve_world(new_state, t_seed)
        push!(mean_opinions, Statistics.mean(new_state))
    end
    return mean_opinions
end


function world_evolution(
    world_size::Tuple{Int, Int},
    t_final::Int,
    start_prob::Number,
    num_trials::Int
)::Tuple{Vector{Float64}, Vector{Float64}}
    """
        world_evolution(world_size, t_final, start_prob, num_trials)
    
    Evolve the world of Day 1

    Performs the Markovian evolution for day 1 and returns
    the mean and standard deviation over `num_trials` number
    of samples of the initial world configuration
    """
    initial_states = [
        start_world(world_size, start_prob, seed)
        for seed in 1:num_trials
    ]
    avg_opinion_traces = map(
        x -> track_average_opinion(x, t_final),
        initial_states
    )
    return (
        Statistics.mean(avg_opinion_traces),
        Statistics.std(avg_opinion_traces)
    )
end


end  # module
