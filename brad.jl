using CSV
using DataFrames
using Statistics
using LinearAlgebra


df = CSV.read("Bradley_Terry_model/Stats.csv", DataFrame)
df_predict = CSV.read("Bradley_Terry_model/predict.csv", DataFrame)


matrix_values, df_parameters = reformat(df)
df_logit_out, df_prob_out, epoch = bradley_terry(matrix_values, df_parameters)
df_predict = predict(df_prob_out, df_predict)

#df_result = CSV.read("Bradley_Terry_model/Stats_result.csv", DataFrame)


function reformat(df)

    new_colnames = ["row_name", "col_name", "freq"]
    rename!(df, Symbol.(new_colnames))

    sort!(df, [:col_name])
    #to bring it to the form of square matrix
    df_pivot = unstack(df, :row_name, :col_name, :freq)
    sort!(df_pivot, [:row_name])

    df_parameters = select(df_pivot, :row_name)
    #selects the other column names
    select!(df_pivot, Not(:row_name))
    for col in names(df_pivot)
        df_pivot[!, col] = Missings.coalesce.(df_pivot[!, col], 0)
    end
    matrix_values = Array(df_pivot)
    convert(Matrix{Float64}, matrix_values)
    matrix_values[diagind(matrix_values)] .= 0.0

    return matrix_values, df_parameters
end


function normalize_values(a)
    b = sum(a)
    y = Matrix{Float64}
    y = a ./ b
    return y
end

function bradley_terry(W, df_parameters)
    epoch = 0
    Parameters = ones(1, size(W, 1))
    length_para = length(Parameters)
    Previous = Parameters
    Para = zeros(1, size(W, 1))
    rounded_value = Matrix{Float64}
    while (true)
        for i in 1:length_para
            sum_W = 0.0
            div_sum = 0.0
            for j in 1:length_para
                if i != j
                    sum_W = sum_W + W[i, j]
                    #println("sum_W %f", sum_W)
                    w_sum = W[i, j] + W[j, i]
                    #println("w_sum %f", w_sum)
                    p_sum = Parameters[i] + Parameters[j]
                    #println("p_sum %f", p_sum)
                    div = w_sum / p_sum
                    #println("div %f", div)
                    div_sum = div_sum + div
                    #println("div_sum %f", div_sum)
                end
            end
            para_new = sum_W / div_sum
            #println("para_new %f", para_new)
            Para[i] = para_new
        end
        Parameters = normalize_values(Para)
        rounded_value = round.(Parameters, digits=12)
        if isequal(Previous, rounded_value)
            break
        else
            Previous = rounded_value
        end
        epoch += 1
    end
    log_val = log.(rounded_value)
    mean_val = mean(log_val)
    logit_likelihood = log_val .- mean_val
    logit_likelihood = round.(logit_likelihood, digits=9)
    convert(Matrix{Float64}, logit_likelihood)
    df_likelihood = DataFrame(transpose(logit_likelihood), :auto)
    df_out = hcat(df_parameters, df_likelihood)
    rename!(df_out, :row_name => :Parameters, :x1 => :likelihood)


    convert(Matrix{Float64}, rounded_value)
    df_probability = DataFrame(transpose(rounded_value), :auto)
    df_prob_out = hcat(df_parameters, df_probability)
    rename!(df_prob_out, :row_name => :Parameters, :x1 => :likelihood)

    return df_out, df_prob_out, epoch
end

function get_winning_probability(parameter_i, parameter_j, dict_prob_out)
    if parameter_i in keys(dict_prob_out)
        probability_i = dict_prob_out[parameter_i]
    end
    if parameter_j in keys(dict_prob_out)
        probability_j = dict_prob_out[parameter_j]
    end
    winning_prob_i = probability_i / sum(probability_i + probability_j)
    return winning_prob_i
end

function predict(df_prob_out, df_predict)
    new_col_names = ["parameter1", "parameter2", "parameter1_winning_probability"]
    rename!(df_predict, Symbol.(new_col_names))
    dict_prob_out = Dict(Pair.(df_prob_out.Parameters, df_prob_out.likelihood))
    for col in names(df_predict)
        df_predict[!, col] = Missings.coalesce.(df_predict[!, col], 0.0)
    end
    transform!(df_predict, [:parameter1, :parameter2, :parameter1_winning_probability] => ByRow((a, b, c) -> round.(get_winning_probability(a, b, dict_prob_out), digits=9)) => :parameter1_winning_probability)
    return df_predict
end











