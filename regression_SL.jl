#cd("/home/max/project/ML")

using CSV, GLM, Plots, TypedTables

data = CSV.File("housingdata.csv")

X = data.size

Y = round.(Int, data.price/1000)

t = Table(X = X, Y = Y)

gr(size =(600,600))

p_scatter = scatter(X,Y,
    xlims = (0,5000),
    ylims = (0,800),
    xlabel = "Size (sqft)",
    ylabel = "Price (in thousands of dollars)",
    title = "Housing Prices in Portland"
)

ols = lm(@formula(Y~X),t) #ordinary least squares method


plot!(X,predict(ols), color = :green, linewidth = 3)


newX = Table(X = [1250])

predict(ols, newX)


########################## ML


epochs = 0


X = data.size

Y = round.(Int, data.price/1000)

t = Table(X = X, Y = Y)

gr(size =(600,600))

p_scatter = scatter(X,Y,
    xlims = (0,5000),
    ylims = (0,800),
    xlabel = "Size (sqft)",
    ylabel = "Price (in thousands of dollars)",
    title = "Housing Prices in Portland (epochs = $epochs)",
    legend = false
)

#parameters


theta_0 = 0.0 # y-intercept

theta_1 = 0.0 # slope



h(x) = theta_0 .+ theta_1 * x


plot!(X,h(X),color = :blue, linewidth = 3)

##################################################


m = length(X) # m is number of samples

y_hat = h(X) # predicted value of y

function cost(X,Y)
    (1/(2*m)) * sum((y_hat - Y).^2) # mean square sum
end

J = cost(X,Y) # need to minimise cost function my adjusting parameters
# Batch Gradient Descent Algorithm

J_history = []


push!(J_history,J)

function pd_theta_0(X,Y)
    (1/m) * sum(y_hat - Y)
end

function pd_theta_1(X,Y)
    (1/m) * sum((y_hat - Y) .* X )
end

# set learning rates (alpha) usually set to 0.01

alpha_0 = 0.09
alpha_1 = 8.0e-8


##############################################################
# begin iterations, repeat until convergence

# calculate partial derivatives

for i ∈ 1:8

    theta_0_temp = pd_theta_0(X,Y) ## how much to adjust the parameters
    theta_1_temp = pd_theta_1(X,Y)

# adjust parameters by learnin rates




    theta_0 -= alpha_0 * theta_0_temp 
    theta_1 -= alpha_1 * theta_1_temp

    y_hat = h(X)

    J = cost(X,Y)

    push!(J_history,J)

    # replot

    epochs += 1

    plot!(X,y_hat, color = :blue,alpha = 0.5
    )

end



title!("Housing Prices in Portland (epochs = $epochs)")

# add ols line 

plot!(X, predict(ols), color = :green , linewidth = 3)


gr(size = (600,600))

p_line = plot(0:epochs, J_history,
    xlabel = "Epochs",
    ylabel = "Cost",
    title = "Learning Curve",
    legend = false,
    color = :blue,
    linewidth = 2,

)

newX_ml = [1250]

h(newX_ml) # machine learning prediction

predict(ols,newX) # ols model predictions # new X = Table(X = [1250])



