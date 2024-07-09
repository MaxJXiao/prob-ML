using DecisionTree
using Random, Statistics

# Decision Trees not known for accuracy, and known for overfitting
# This led to a Bias-Variance Tradeoff discussion
# Bagging: Boostrap aggregating (random forest: more decision trees)
# Boosting: weak learners come together to make a strong learner
# ie. bunch of kids in a classroom coming together
# most guessed/chosen answer is assigned as the decision
# leading to Random Forest and Adaboosting


X,y = load_data("iris") # inputs and Outputs

# assign data types to data, recommended

X = float.(X)
y = string.(y)

iris = [X y]

vscodedisplay(iris)

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

train_index = perclass_splits(y, 0.67) # only used for training model

test_index = setdiff(1:length(y), train_index) # use for test data




Xtrain = X[train_index, :] # characteristics of training data without knowing their identity

Xtest = X[test_index, :]

ytrain = y[train_index]  # identity of training data

ytest = y[test_index]


# This package does not require transposition of data

############ Decision Tree ##############


model = DecisionTreeClassifier(max_depth = 2) 

# limit depth to limit how many times the model runs through 
# the data. Meaning we can generalise results for new data
# root means first decision node



fit!(model, Xtrain,ytrain)

print_tree(model)

# view training data to see what the model is using


train = [Xtrain ytrain]

vscodedisplay(train)


train_R = train[train[:,4] .> 0.8,:]

vscodedisplay(train_R)


y_hat = predict(model,Xtest)

accuracy = mean(y_hat .== ytest)

## confusion matrix is to visualise why your model got confused

DecisionTree.confusion_matrix(ytest,y_hat)

# classes from first argument are from the rows, 
# and 2nd argument is columns

#                      |                predicted                |
#          ____________| setosa | versicolor | virginica | total |
#         | setosa     |   15   |     0      |     0     |  15   |
# labeled | versicolor |   0    |     23     |    (1)    |  24   |
#         | virginica  |   0    |    (5)     |     12    |  17   |
#         | total      |   15   |     28     |     13    |  56   |


# diagonals are correct predictions. Any other number is incorrrect
# 6 / 56 = 10.7%, ie 50/56 = 89.3% the accuracy
# kappa is better predictor of the actual prediction score
# ie. the model made a prediction correct on sheer chance
# meaning the model got lucky and it isn't actually 89% accurate
# remove random positive prediction, make more conservative estimate


check = [y_hat[i] == ytest[i] for i ∈ 1:length(y_hat)]


check_display = [y_hat ytest check]

vscodedisplay(check_display)


## SVM does not make predictions based on probability
## Decision Tree makes decision based on probability

prob = predict_proba(model,Xtest)

vscodedisplay(prob)


## Decision Tree high variance, low bias
## Tends to overfit data, as in theory it can continue to split
## data until it gets all the training data correct

## Bagging (Bootstrap, aggregating): Random Forest
# increase independence of features, and models
# lower variance, but slightly higher bias

##### Thus, use bagging to lower the variance of Decision Trees

# make multiple decision tree models, only consider a fraction
# of features at every split

model = DecisionTree.RandomForestClassifier(n_trees = 20)
# number of decision trees generated


fit!(model, Xtrain, ytrain)

y_hat = predict(model,Xtest)

#check accuracy

accuracy = mean(y_hat .== ytest)

DecisionTree.confusion_matrix(ytest,y_hat)


#                      |                predicted                |
#          ____________| setosa | versicolor | virginica | total |
#         | setosa     |   15   |     0      |     0     |  15   |
# labeled | versicolor |   0    |     23     |    (1)    |  24   |
#         | virginica  |   0    |    (3)     |     14    |  17   |
#         | total      |   15   |     26     |     15    |  56   |


check = [y_hat[i] == ytest[i] for i ∈ 1:length(y_hat)]

check_display = [y_hat ytest check]

vscodedisplay(check_display)


prob = predict_proba(model,Xtest)

vscodedisplay(prob)


## Boosting is to lower bias
# Can a set of weak learners make a strong learners
# weak learner: only slightly better than random guess

## Adaboost: adaptive Boosting
# Stump: Only one root aka only one decision
# many Stumps
# data set will figure out the harder things to classify


model = DecisionTree.AdaBoostStumpClassifier(n_iterations = 20)
# number of stumps



fit!(model, Xtrain, ytrain)

y_hat = predict(model,Xtest)

#check accuracy

accuracy = mean(y_hat .== ytest)

DecisionTree.confusion_matrix(ytest,y_hat)


#                      |                predicted                |
#          ____________| setosa | versicolor | virginica | total |
#         | setosa     |   15   |     0      |     0     |  15   |
# labeled | versicolor |   0    |     23     |    (1)    |  24   |
#         | virginica  |   0    |    (2)     |     15    |  17   |
#         | total      |   15   |     25     |     16    |  56   |


check = [y_hat[i] == ytest[i] for i ∈ 1:length(y_hat)]

check_display = [y_hat ytest check]

vscodedisplay(check_display)


prob = predict_proba(model,Xtest)

vscodedisplay(prob)

## Adaboost, not confident about choices but generally guesses correctly

