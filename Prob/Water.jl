using Turing, StatsPlots

# n number of tosses
# p is percentage of earth covered

n = 9

p = 0.71

f(n, p) = Int(round(n * p))

w = f(n, p)

w / n


# input <- model <- output

tosses = 9

water = 6


# posterior = prior × likelihood



@model function globe_toss(tosses, water)
    # prior
    percent_water ~ Beta(1, 1) # ~ means distributed as

    # likelihood
    water ~ Binomial(tosses, percent_water)

end

# infer posterior distribution

model = globe_toss(tosses, water)

sampler = NUTS() # no u-turn sampler


samples = 1000

chain = sample(model, sampler, samples)

plot(chain)

density(chain[:percent_water],
    legend = false,
    linewidth = 2,
    fill = true,
    alpha = 0.3,
    xlims = (0, 1),
    widen = true,
    title = "Posterior Distribution (approx)",
    xlabel = "percent_water",
    ylabel = "density"
)


# uses a prior belief / probability of an event
# then uses Bayes theorem (updated data) to adjust this probability


"""

P(A | B) = posterior probability
P(B | A) = likelihood
P(A) = prior likelihood
P(B) = Marginal probability (normalises distribution)


Posterior = Prior × likelihood

P(A | B) = P(B | A) × P(A) / P(B)

Posterior = prior belief × likelihood

posterior = updated belief

"""


"""
User Provided

1. Observe event
2. Define model
3. Select Inference Algorithm

Automated

1. Build chain (approximation of posterior probability)

"""