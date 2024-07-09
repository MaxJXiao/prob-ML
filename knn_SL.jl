using NearestNeighbors, Plots, RDatasets, StatsBase, Random

gr(size = (600,600))

Random.seed!(1)

f1_train = rand(100)

f2_train = rand(100)

p_knn = scatter(f1_train, f2_train,
    xlabel = "Feature 1",
    ylabel = "Feature 2",
    title = "k-NN & k-D Tree Demo",
    legend = false,
    color = :blue 
)


# build tree
X_train = [f1_train f2_train]

X_train_t = permutedims(X_train)

kdtree = KDTree(X_train_t)

# take dataset, map to dataset to make it efficient for analysis

k = 11 ## k nearest neighbours 11 data points that are nearest to the test data points


f1_test = rand()

f2_test = rand()

X_test = [f1_test, f2_test]

# add to scatter Plots

scatter!([f1_test], [f2_test],
    color = :red, markersize = 10


)


# find nearest neighbours using k-NN & k-d tree

index_knn, distances = knn(kdtree, X_test, k, true)

output = [index_knn distances]

vscodedisplay(output) # display 11 nearest neighbours and associated euclidean distances


# plot nearest neighbours

f1_knn = [f1_train[i] for i ∈ index_knn]

f2_knn = [f2_train[i] for i ∈ index_knn]

scatter!(f1_knn,f2_knn, 
    color = :yellow, markersize = 10, alpha = 0.5
)

# connect test point to neighbours


for i ∈ 1:k 
    plot!([f1_test,f1_knn[i]], [f2_test, f2_knn[i]], color = :green)
end

p_knn

savefig(p_knn, "knn_concept_plot.svg")


##############


using RDatasets, StatsBase

using Statistics


iris = dataset("datasets","iris")

X = Matrix(iris[:,1:4])

y = Vector{String}(iris.Species)



# define function to split data (source: Huda Nassar)

function perclass_splits(y, percent)
    uniq_class = unique(y)
    keep_index = []
    for class in uniq_class
        class_index = findall(y .== class)
        row_index = randsubseq(class_index, percent)
        push!(keep_index, row_index...)
    end
    return keep_index
end

Random.seed!(1)

index_train = perclass_splits(y,0.67)

index_test = setdiff(1:length(y), index_train)


X_train = X[index_train,:]

X_test = X[index_test,:]

y_train = y[index_train,:]

y_test = y[index_test,:]


#transpose

X_train_t = permutedims(X_train)

X_test_t = permutedims(X_test)

#build tree

kdtree = KDTree(X_train_t)


## model

k = 11 # 5

index_knn, distances = knn(kdtree,X_test_t,k,true)

output = [index_test index_knn distances]

vscodedisplay(output)

# post processing


index_knn_matrix = hcat(index_knn...)

index_knn_matrix_t = permutedims(index_knn_matrix)

knn_classes = y_train[index_knn_matrix_t]

vscodedisplay(knn_classes) ## see what class are the nearest neighbours

# maybe some nearest neighbours aren't the same as they others

# use statsbase to make predictions

y_hat = [
    argmax(countmap(knn_classes[i,:]))
    for i ∈ 1:length(y_test)

]

# demo for countmaps and argmax

demo = knn_classes[53,:]

countmap_demo = countmap(demo) # like a histogram

argmax(countmap_demo) #highest category

accuracy = mean(y_hat .== y_test)

check = [y_hat[i] == y_test[i] for i ∈ 1:length(y_hat)]

check_display = [y_hat y_test check]

vscodedisplay(check_display)

# k is odd number, avoid tie votes if 2 party system
# k > # classes
# Feature scaling is important