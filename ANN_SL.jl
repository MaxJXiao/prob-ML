# Artificial Neural Networks

using Flux, Images, MLDatasets, Plots

using Flux: crossentropy, onecold, onehotbatch, params, train!

using LinearAlgebra, Random, Statistics

Random.seed!(1)

# load data

X_train_raw, y_train_raw = MLDatasets.MNIST(:train)[:]
X_test_raw, y_test_raw = MLDatasets.MNIST(:test)[:]


X_train_raw

index = 1

img = X_train_raw[:,:,index]

colorview(Gray,img') # need to transpose Images

y_train_raw

y_train_raw[index]

## 60000 training, 10000 testing

X_test_raw

img = X_test_raw[:,:,index]

colorview(Gray,img')

y_test_raw[index]

# flatten input data

X_train = Flux.flatten(X_train_raw)

X_test = Flux.flatten(X_test_raw)

# one hot encode labels

y_train = onehotbatch(y_train_raw,0:9)

y_test = onehotbatch(y_test_raw,0:9)


model = Chain(
    Dense(28 * 28, 32 , relu), # 1st input layer 28*28 = 784, to 2nd 32.
    Dense(32,10), #2nd layer (32) to 3rd output layer (10) which is deciding outputs (0-9)
    softmax

)


""" 
Chain(
    Dense(784 => 32, relu), # 25_120 parameters
    Dense(32 => 10), # 330 parameters

25_088 = 784 * 32 nodes
Add 32 because each node in the next layer will contain a parameter
25_120

320 = 32 * 10 nodes
Add 10 because each node in output layer will contain a parameter
330

Total = 25_450 parameters (25_120 + 330)

Parameters along edges are called weights, θ₁ (slope), W
parameters along nodes are called bias, θ₀ (y-intercept), because
WX + b


NNlib.softmax returns values between 0 and 1, and normalises it
)
"""

"""
1st step

(X) Node ------ W ------ Node (b)
ReLU(WX + b)
X is inputs/images

W (32 x 784) * X (784 x 60000) .+ b (32 x 1) = WX .+ b (32 x 60000)

2nd step

(X) Node ------ W ------ Node(b)
Softmax(WX .+ b) outputs 0 to 1, adding to 100%
W (10 x 32) * X (32 x 60000) .+ b (10 x 1) = WX .+ b (10 x 60000)


"""


# define loss function

loss(x,y) = crossentropy(model(x),y)

# agg(-sum(y .* log.(y^ .+ ϵ) ; dims ) )
# crossentropy rewards correct predictions, punishes incorrect


# track parameters

ps = params(model)
#ps[1] (32 x 784) weights for first dense layer
#extrema(ps[1]) = (-0.08, 0.08)
#ps[2] = 0s (32 x 1) bias for nodes
#ps[3] (10 x 32) weights for second dense layer
#extrema(ps[3]) = (-0.34, 0.34)
#ps[4] = 0s (10 x 1) bias for output nodes

# selective optimiser ADAM adaptive movement estimation


learning_rate = 0.01

opt = ADAM(learning_rate) 
# stochastic gradient descent
# learning rate decays over time

## train model

loss_history = []

epochs = 500

for epoch ∈ 1:epochs
    train!(loss,ps,[(X_train,y_train)],opt) # scaler containing a tuple

    # (Matrix{Float32} (784 x 60000) , OneHotArray{Int32} (10 x 60000))
    # (input, output labels)

    #print report

    train_loss = loss(X_train,y_train)
    push!(loss_history, train_loss)

    println("Epoch = $epoch : Training Loss = $train_loss")


end

#extrema(ps[1]) = (-4.19, 4.17)
#ps[2] actual biases
#extrema(ps[2]) = (-0.46, 0.42)
#extrema(ps[3]) = (-3.33, 2.21)
#extrema(ps[4]) = (-0.33, 0.20)


"""
Might consider saving parameters to save the model
"""


## make predictions

y_hat_raw = model(X_test)

# onecold matrix into column matrix with highest probability values

y_hat = onecold(y_hat_raw) .- 1 # convert index values to labels because our outputs are 0-9 which are mapped to index 1-10

y = y_test_raw

mean(y_hat .== y)

# results

check = [y_hat[i] == y[i] for i ∈ 1:length(y)]

index = collect(1:length(y))

check_display = [index y_hat y check]

vscodedisplay(check_display)

# view misclassifications

misclass_index = 9

img = X_test_raw[:,:,misclass_index]

colorview(Gray,img')


## plot leraning curve

p_l_curve = plot(1:epochs, loss_history,
    xlabel = "Epochs",
    ylabel = "Loss",
    title = "Learning Curve",
    legend = false,
    color = :blue, 
    linewidth = 2

)

savefig(p_l_curve, "ann_learning_curve.svg")

