### Dimension reduction, 4 to 3 so it can be plotted and analysed

using MultivariateStats, Plots, RDatasets, Random

Random.seed!(1)

f0 = collect(0.05:0.05:5)

f1 = f0 .+ rand(100)

f2 = f0 .+ rand(100)

plotlyjs(size = (480,480))


p_random = scatter(f1, f2,
    xlabel = "Feature 1",
    ylabel = "Feature 2",
    title = "Random Data",
    legend = false
)


X = [f1 f2]'


model = fit(PCA, X; maxoutdim = 1)

X_transform = MultivariateStats.transform(model, X)


y = zeros(100)

p_transform = scatter(X_transform', y,
    xlabel = "PC1",
    title = "Random Data Transform",
    legend = false,
    color = :red,
    alpha = 0.5
)

X_reconstruct = reconstruct(model, X_transform)

scatter!(p_random,
    X_reconstruct[1,:], X_reconstruct[2,:],

    color = :red,
    alpha = 0.5
)

savefig(p_random, "pca_concept_plot.svg")

##################### example

iris = dataset("datasets", "iris")

X = Matrix(iris[:,1:4])'

y = Vector{String}(iris.Species)

species = reshape(unique(iris.Species), (1,3))


model = fit(PCA, X; maxoutdim = 3)


X_transform = MultivariateStats.transform(model,X)

PC1 = X_transform[1,:]

PC2 = X_transform[2,:]

PC3 = X_transform[3,:]



plotlyjs(size = (640,480))

p_transform = scatter(PC1,PC2,PC3,
    xlabel = "PC1", ylabel = "PC2", zlabel = "PC3",
    title = "Iris Dataset PCA Transform",
    markersize = 2,
    group = y,
    label = species
)

savefig(p_transform, "iris_PCA_plot.svg")