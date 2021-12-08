---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.3
  kernelspec:
    display_name: Julia 1.7.0
    language: julia
    name: julia-1.7
---

# Day 1 - Solution
## Cameron King

Let's try out a brute force method in Julia

```julia
using DataFrames, Distributions, Optim, Plots, Random, Statistics;
```

```julia
using Day1;
```

```julia
# function toroid_neighbors(
#     selection::Tuple{Int, Int},
#     world_size::Tuple{Int, Int}
# )::NTuple{5, Tuple{Int, Int}}
#     i, j = selection
#     M, N = world_size
#     base_selection = (
#         (i - 1, j),
#         (i, j - 1),
#         (i, j),
#         (i, j + 1),
#         (i + 1, j)
#     )
    
#     mod_sel = map(
#         x -> (x[1] % M, x[2] % N),
#         base_selection
#     )
#     complement = map(
#         x -> (
#             x[1] > 0 ? x[1] : M + x[1],
#             x[2] > 0 ? x[2] : N + x[2]
#         ),
#         mod_sel
#     )
# end
```

Let's imagine everyone decides their opinion on whether the world is flat or round at the start of time. In this, if a person votes 1, the world is round and a zero means the world is flat. We can model each individual using a Bernoulli trial with their probability of having that opinion.

```julia
# function start_world(
#     world_size::Tuple{Int, Int},
#     prob_round::Number,
#     seed::Int
# )::Matrix{Bool}
#     Random.seed!(seed)
#     prob_matrix = prob_round * ones(world_size)
#     return map(rand, map(Bernoulli, prob_matrix))
# end
```

Now we need a model for a time step. Given a certain citizen, they look around their neighbors, check their Facebook pages, and then change their opinions so that they fit in with their local bubble.

```julia
# function vote_results(
#     prior_state::Matrix{Bool},
#     citizen_coord::Tuple{Int, Int}
# )
#     selection_elements = toroid_neighbors(
#         citizen_coord, size(prior_state)
#     )
#     vote_sum = sum([prior_state[i, j] for (i, j) in selection_elements])
#     vote_result = (vote_sum >= 3) ? 1 : 0
    
#     new_state = copy(prior_state)
#     for (i, j) in selection_elements
#         new_state[i, j] = vote_result
#     end
#     return new_state
# end
```

In the function below, we set up the mechanism of selecting a citizen and having them vote.

```julia
# function evolve_world(starting_state::Matrix{Bool}, seed::Int)::Matrix{Bool}
#     Random.seed!(seed)
#     states = [starting_state,]
#     new_state = starting_state
#     world_shape = size(starting_state)
#     citizen_pick = (
#         rand(1:world_shape[1]), rand(1:world_shape[2])
#     )
#     new_state = vote_results(new_state, citizen_pick)
# end
```

I don't actually want to track the full state of the world at every time step (especially if I want to look at full grids), so instead I'll track the progress towards consensus that the world is indeed round.

```julia
# function track_average_opinion(
#     starting_state::Matrix{Bool},
#     tfinal::Int
# )::Vector{Float64}
#     mean_opinions = [mean(starting_state)]
#     new_state = starting_state
#     for t_seed in 1:tfinal
#         new_state = evolve_world(new_state, t_seed)
#         push!(mean_opinions, mean(new_state))
#     end
#     return mean_opinions
# end
```

And now, I'll actually carry out a full set of world evolutions along with averaging over initial states. Because of how the seed was set, the same citizens will be voting over the time evolution, but this should be a reasonable sampling routine.

```julia
# function world_evolution(
#     world_size::Tuple{Int, Int},
#     t_final::Int,
#     start_prob::Number,
#     num_trials::Int
# )::Tuple{Vector{Float64}, Vector{Float64}}
#     initial_states = [
#         start_world(world_size, start_prob, seed)
#         for seed in 1:num_trials
#     ]
#     avg_opinion_traces = map(
#         x -> track_average_opinion(x, t_final),
#         initial_states
#     )
#     return (mean(avg_opinion_traces), std(avg_opinion_traces))
# end
```

```julia
t_final = 4000;
prob = 0.9;
num_shots = 200;
world_sizes = [
    (10, 10),
    (20, 20),
    (30, 30),
    (40, 40),
    (50, 50),
    (60, 60)
]

means = Array{Float64}(undef, length(world_sizes), t_final + 1)
stddevs = Array{Float64}(undef, length(world_sizes), t_final + 1)
areas = Array{Float64}(undef, length(world_sizes), 1)
for (i, shape,) in enumerate(world_sizes)
    mean_val, stddev_val = world_evolution(shape, t_final, prob, num_shots);
    area = prod(shape)
    means[i, :] = mean_val
    stddevs[i, :] = stddev_val
    areas[i] = area
end
```

```julia
t_array = 1:t_final + 1;
labels = [
    "$(s[1])x$(s[2])"
    for s in world_sizes
]
plot(t_array, transpose(means), ribbon=transpose(stddevs), label=reshape(labels, 1, length(labels)))
xlabel!("t [hours]")
ylabel!("round fraction")
```

Fantastic that looks quite like an exponential settling that scales with the area of the world. Let's fit that to an expoential model and see how that does.

```julia
function test_model(betas, t)
    return [
        1.0 - betas[1] * exp(-t_i / betas[2])
        for t_i in t
    ]
end
```

```julia
settle_times = Array{Float64, 1}(undef, length(areas))
mean_sq_errors = Array{Float64, 1}(undef, length(areas))
for i in 1:length(areas)
    mean_vals = means[i, :]
    area = areas[i]
    function sqerror(betas)
        error = 0.0
        predictions = test_model(betas, 1:size(mean_vals)[1])
        error = sum((mean_vals - predictions).^2)
    end
    
    fit_res = optimize(sqerror, [0.2, 500], LBFGS())
    settle_times[i] = Optim.minimizer(fit_res)[2]
    mean_sq_errors[i] = Optim.minimum(fit_res)
end

fit_data = DataFrame(
    "area" => reshape(areas, length(areas)),
    "Setting time" => settle_times,
    "Settling time / area" => settle_times./reshape(areas, length(areas)),
    "MSE" => mean_sq_errors
);
```

```julia
fit_data
```

Let's switch that to 0.5 probability (this is going to go great...)

```julia
t_final = 100000;
prob = 0.5;
num_shots = 200;
world_sizes = [
    (10, 10),
    (20, 20),
    (30, 30),
    (40, 40),
    (50, 50),
    (60, 60)
]

means = Array{Float64}(undef, length(world_sizes), t_final + 1)
stddevs = Array{Float64}(undef, length(world_sizes), t_final + 1)
areas = Array{Float64}(undef, length(world_sizes), 1)
for (i, shape,) in enumerate(world_sizes)
    mean_val, stddev_val = world_evolution(shape, t_final, prob, num_shots);
    area = prod(shape)
    means[i, :] = mean_val
    stddevs[i, :] = stddev_val
    areas[i] = area
end
```

```julia
t_array = 1:t_final + 1;
labels = [
    "$(s[1])x$(s[2])"
    for s in world_sizes
]
plot(
    t_array,
    transpose(means),
    ribbon=transpose(stddevs),
    label=reshape(labels, 1, length(labels)),
    xaxis=:log
)
xlabel!("t [hours]")
ylabel!("round fraction")
```

Well, looks like some of the populations settle, but it takes a while as the system size grows.

```julia

```
