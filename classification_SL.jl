# logistic regression to solve classification problems 
# output not continuous (discrete)
# true/false or more outcomes than 2

# cd("/home/max/project/ML")


using Plots, CSV

gr(size = (600,600))

logistic(x) = 1/ (1 + exp(-x))

p_logistic = plot(-6:0.1:6,logistic,

    xlabel = "Inputs (x)",
    ylabel = "Outputs (y)",
    title = "Logistic (Sigmoid) Curve",
    legend = false,
    color = :blue


)

# modify logistic Curve

θ₀ = 0.0 # y-intercept
θ₁ = -0.5 # slope/gradient

# hypothesis function

z(x) = θ₀ .+ θ₁ * x

h(x) = 1 ./ (1 .+ exp.(-z(x)))

plot!(h,color = :green, linestyle = :dash)


data = CSV.File("wolfspider.csv")


X = data.feature

Y_temp = data.class # feature and class are column titles


Y = []

for i ∈ 1:length(Y_temp)
    if Y_temp[i] == "present"
        y = 1.0
    else
        y = 0
    end
    push!(Y,y)
end

p_data = scatter(X,Y,
    xlabel = "Size of Grains of Sand (mm)",
    ylabel = "Probability of Observation (Absent = 0 | Present = 1)",
    title = "Wolf Spider Presence Classifier",
    legend = false,
    color = :red,
    markersize = 5
)

# not so obvious to fit a logistic curve here

# Initialise parameters
# Hypothesis function
# Cost function
# adjust parameters
# Iterate until convergence (minimisation of cost function)


# initialise parameters

θ₀ = 0.0
θ₁ = 1.0

# track parameter history

θ₀_history = []
θ₁_history = []

push!(θ₀_history, θ₀)
push!(θ₁_history, θ₁)


# define hypothesis function

z(x) = θ₀ .+ θ₁ * x 

h(x) = 1 ./ (1 .+ exp.(-z(x)))

# plot initial hypothesis

plot!(0:0.1:1.2,h , color = :green) #data only between 0 - 1.2

y_hat = h(X) # predicted values of Y

# define our cost function (reward correct prediction and punish bad ones)

m = length(X)

function cost()
    (-1/m) * sum(
        Y .* log.(y_hat) + (1 .- Y) .* log.(1 .- y_hat)
    )
end

J = cost()

J_history = []

push!(J_history,J)

# definite batch gradient descent Algorithm

# want to find out temperature (how much parameters should change)

# use partial derivative formulae from Andrew Ng (pd = partial derivative)

# ∂(cost)/∂θᵢ, i = 1,2

function pd_θ₀()
    sum(y_hat - Y)
end

function pd_θ₁()
    sum((y_hat - Y) .* X )
end


# set learning rates

α = 0.01

# initialise epochs

epochs = 0


# begin iterations

for i ∈ 1:3000


    θ₀_temp = pd_θ₀()
    θ₁_temp = pd_θ₁()

    # adjust parameters by learning rates

    θ₀ -= α*θ₀_temp
    θ₁ -= α*θ₁_temp

    y_hat = h(X)

    J = cost()

    push!(J_history,J)

    push!(θ₀_history, θ₀)
    push!(θ₁_history, θ₁)

    epochs += 1

    plot!(0:0.1:1.2,h , color = :blue, linestyle = :dash, alpha = 0.025) #data only between 0 - 1.2


end


title!("Wolf Spider Presence Classifer (epochs = $epochs)")



p₁_curve = plot(0:epochs, J_history,
    xlabel = "Epochs",
    ylabel = "Cost",
    title = "Learning Curve",
    legend = false,
    color = :blue,
    linewidth = 2
)


p_params = scatter(θ₁_history, θ₀_history,
    xlabel = "θ₁",
    ylabel = "θ₀",
    title = "Gradient Descent Path",
    legend = false,
    color = :blue,
    alpha = 0.05
)

# make predictions

newX = [0.25, 0.5, 0.75, 1.0]

h(newX)







































