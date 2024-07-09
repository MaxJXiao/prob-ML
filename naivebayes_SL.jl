#cd("/home/max/project/ML")

#alt j + alt o opens repl quickly

using DelimitedFiles, CSV

data = readdlm("tennis.csv", ','; skipstart = 1)
# outlook,temp,humidity,windy,play (columns)
# will they play tennis


x₁ = data[:,1]
x₂ = data[:,2]
x₃ = data[:,3]
x₄ = data[:,4]
y = data[:,5]

# identify unique elements

uniq_x₁ = unique(x₁)
uniq_x₂ = unique(x₂)
uniq_x₃ = unique(x₃)
uniq_x₄ = unique(x₄)

uniq_y = unique(y)


# calculate probabilities for "yes" and "no" Outputs

len_y = length(y)

len_yes = count(x -> x == "yes", y) # -> anonymous function

len_no = count(x -> x == "no",y)

#f = (x,y) -> x^2 + y^2 # f(3,4) = 25

p_yes = len_yes/len_y
p_no = len_no/len_y

# split "yes" and "no" into separate matrices

data_yes = data[data[:,5] .== "yes" , :]
data_no = data[data[:,5].== "no", :]

# count features in data_yes

len_sunny_yes = count(x -> x == uniq_x₁[1], data_yes)
len_overcast_yes = count(x -> x == uniq_x₁[2], data_yes)
len_rainy_yes = count(x -> x == uniq_x₁[3], data_yes)
v_outlook_yes = [len_sunny_yes,len_overcast_yes,len_rainy_yes]

len_hot_yes = count(x -> x == uniq_x₂[1], data_yes)
len_mild_yes = count(x -> x == uniq_x₂[2], data_yes)
len_cool_yes = count(x -> x == uniq_x₂[3], data_yes)
v_temp_yes = [len_hot_yes, len_mild_yes, len_cool_yes]

len_high_yes = count(x -> x == uniq_x₃[1], data_yes)
len_normal_yes = count(x -> x == uniq_x₃[2], data_yes)
v_humid_yes  = [len_high_yes,len_normal_yes]


len_false_yes = count(x -> x == uniq_x₄[1], data_yes)
len_true_yes = count(x -> x == uniq_x₄[2], data_yes)
v_windy_yes = [len_false_yes, len_true_yes]

# count for no

len_sunny_no = count(x -> x == uniq_x₁[1], data_no)
len_overcast_no = count(x -> x == uniq_x₁[2], data_no)
len_rainy_no = count(x -> x == uniq_x₁[3], data_no)
v_outlook_no = [len_sunny_no,len_overcast_no,len_rainy_no]

len_hot_no = count(x -> x == uniq_x₂[1], data_no)
len_mild_no = count(x -> x == uniq_x₂[2], data_no)
len_cool_no = count(x -> x == uniq_x₂[3], data_no)
v_temp_no = [len_hot_no, len_mild_no, len_cool_no]


len_high_no = count(x -> x == uniq_x₃[1], data_no)
len_normal_no = count(x -> x == uniq_x₃[2], data_no)
v_humid_no  = [len_high_no,len_normal_no]


len_false_no = count(x -> x == uniq_x₄[1], data_no)
len_true_no = count(x -> x == uniq_x₄[2], data_no)
v_windy_no = [len_false_no, len_true_no]



## Naive Bayes Classifier

# prediction 1: newX = ["sunny","hot"]

p_yes_newX = (len_sunny_yes / len_yes) *
(len_hot_yes / len_yes) *
p_yes
# P(newX | yes), probability of newX given yes is true

p_no_newX = (len_sunny_no / len_no) * 
(len_hot_no / len_no) *
p_no
# P(newX | no), probability of newX give no is true

#normalise probabilities

p_yes_newXₙ = p_yes_newX / (p_yes_newX + p_no_newX)
p_no_newXₙ = p_no_newX / (p_yes_newX + p_no_newX)

# naive bayes: P(yes | newX) = [P(newX | yes)*P(yes)] / P(newX)
# bayes theorem assumes events are independent, 
# naive bayes just assumes they are and uses bayes anyways
# despite events being connected by someway (ie hot -> high temp)


# prediction 2: newX = ["sunny", "cool", "high", "true"]

p_yes_newX = 
    (len_sunny_yes / len_yes)*
    (len_cool_yes / len_yes)*
    (len_high_yes / len_yes)*
    (len_true_yes / len_yes)*
    p_yes

p_no_newX = 
    (len_sunny_no / len_no)*
    (len_cool_no / len_no)*
    (len_high_no / len_no)*
    (len_true_no / len_no)*
    p_no

# normalize probabilities


p_yes_newXₙ = p_yes_newX / (p_yes_newX + p_no_newX)
p_no_newXₙ = p_no_newX / (p_yes_newX + p_no_newX)


#outlook,temp,humidity,windy

#count(x -> x == uniq_x₁[1], data_yes)

function prob_yes(v::Vector{String})

    l = length(v) # get number of conditions
    c₁ = 0
    c₂ = 0
    c₃ = 0
    c₄ = 0
    m₁ = 1
    m₂ = 1
    m₃ = 1
    m₄ = 1
    n₁ = 1
    n₂ = 1
    n₃ = 1
    n₄ = 1
    for j ∈ 1:l
        for i ∈ 1:length(uniq_x₁)
            if uniq_x₁[i] == v[j]
                c₁ += 1
                m₁ *= v_outlook_yes[i]
                n₁ *= v_outlook_no[i]
            end
        end
        for i ∈ 1:length(uniq_x₂)
            if uniq_x₂[i] == v[j]
                c₂ += 1
                m₂ *= v_temp_yes[i]
                n₂ *= v_temp_no[i]
            end
        end
        for i ∈ 1:length(uniq_x₃)
            if uniq_x₃[i] == v[j]
                c₃ += 1
                m₃ *= v_humid_yes[i]
                n₃ *= v_humid_no[i]
            end
        end
        for i ∈ 1:length(uniq_x₄)
            if uniq_x₄[i] == conv_bool(v[j])
                c₄ += 1
                m₄ *= v_windy_yes[i]
                n₄ *= v_windy_no[i]
            end
        end
    end
    
    
    if (c₁ > 1 || c₂ > 1 || c₃ > 1 || c₄ > 1) == true
        return "Invalid Inputs"
    else
        p_y_cond = p_yes * m₁ * m₂ * m₃ * m₄ / len_yes^(c₁ + c₂ + c₃ + c₄) 
        p_n_cond = p_no * n₁ * n₂ * n₃ * n₄ / len_no^(c₁ + c₂ + c₃ + c₄) 
    
        return round(p_y_cond / (p_y_cond + p_n_cond), digits = 3 ),[c₁, c₂, c₃, c₄]
    end
    


end


function conv_bool(s::String)
    if s == "true"
        return true
    elseif s == "false"
        return false
    else
        return s
    end
end