# identify clusters in random data
# hence it can identify clusters in unlabelled data

using Clustering, Plots, Random

gr(size = (600,600))

Random.seed!(1)

f1 = rand(100)
f2 = rand(100)


p_rand = scatter(f1,f2,
    xlabel = "Feature 1",
    ylabel = "Feature 2",
    title = "Random Data",
    color = :blue,
    legend = false

)

X = [f1 f2]'

k = 10

itr = 100 # will stop if converge along a solution. However if no convergence in 100, it will stop

result = kmeans(X,k ; maxiter = itr, display = :iter)
#converged in 4 iterations

a = assignments(result)

c = counts(result)


μ = result.centers

# columns are coordinates

p_kmeans_demo = scatter(f1,f2,
    xlabel = "Feature 1",
    ylabel = "Feature 2",
    title = "k-means Clustering Demo",
    legend = false,
    group = a,  # comes from the assignments
    markersize = 10,
    alpha = 0.7
)

scatter!(μ[1,:], μ[2,:],
    color = :yellow,
    markersize = 20,
    alpha = 0.7
)

########### example

using RDatasets

cats = dataset("boot", "catsM") #unlabelled dataset

vscodedisplay(cats)

p_cats = scatter(cats.BWt, cats.HWt,
    xlabel = "Body Weight (kg)",
    ylabel = "Heart Weight (g)",
    title = "Weight Data for Domestic Cats (raw data)",
    legend = false,
)

# scale features

f1 = cats.BWt
f2 = cats.HWt

f1_min = minimum(f1)
f2_min = minimum(f2)

f1_max = maximum(f1)
f2_max = maximum(f2)

f1ₙ = (f1 .- f1_min) ./ (f1_max - f1_min)
f2ₙ = (f2 .- f2_min) ./ (f2_max - f2_min)

X = [f1ₙ f2ₙ]'

p_cats_n = scatter(f1ₙ, f2ₙ,
    xlabel = "Body Weight",
    ylabel = "Heart Weight",
    title = "Weight Data for Domestic Cats (Normalised)",
    legend = false,
)

# initialise variables

k = 3 # perhaps small, medium, large cats as labels is what the algorithm will spit output

itr = 6

Random.seed!(1)

result = kmeans(X, k; maxiter = itr, display = :iter)

a = assignments(result)

c = counts(result)

μ = result.centers

p_kmeans_cats = scatter(f1ₙ,f2ₙ,
    xlabel = "Body Weight",
    ylabel = "Heart Weight",
    title = "Weight Data for Domestic Cats (iter = $itr)",
    legend = false,
    group = a,  # comes from the assignments
    markersize = 10,
    alpha = 0.7
)

scatter!(μ[1,:], μ[2,:],
    color = :yellow,
    markersize = 20,
    alpha = 0.7
)

savefig(p_kmeans_cats, "kmeans_cats_plot.svg")