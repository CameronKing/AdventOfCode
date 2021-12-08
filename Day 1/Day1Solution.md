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
using DataFrames, Optim, Plots
```

```julia
using Day1;
```

At the start of the world clock, we'll use a Bernoulli distribtion to initialize the state of the world. At this point, everyone, in isolation, will decide if the world if flat or round probabilistically (controlled by `prob` below). Then, we'll start to evolve the world. At each step in time, we'll choose a citizen and that citizen will query their nearest neighbors. Since the world is actually a torus, everyone will always have 4 neighbors. Then, once everyone has checked their Facebook wall, the group will colectively update their opinions based on the vote of the majority. Then time will step again.

I'll carry out this exercise over many different initial states of the world, controlled by the `num_shots` parameter below. I'll also run this for `t_final` number of time steps. I don't actually want to track each individual's vote, so at the end of a full time trace, I'll collapse the data down and track the fraction that agrees that the earth is round.

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
