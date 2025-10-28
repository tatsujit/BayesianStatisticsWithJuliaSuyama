using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p137-bernoulli-ancestral-sampling"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf
using IceCream
using ColorSchemes
using Colors
using Random
using LinearAlgebra
"""
これを使うとエラーを抑制できる？
"""
function Base.joinpath(s::String, n::Nothing)
    joinpath(s)
end

include(srcdir("utility_functions.jl"))

rseed = 123456789
Random.seed!(rseed)

function generate(N)
    μ = rand(Uniform(0, 1))
    X = rand(Bernoulli(μ), N)
    μ, X
end
# generate(5)

function posterior(observation, max_iter; generate=generate)
    # μ_posteriori = Vector{Float64}(undef, max_iter)
    μ_posteriori = []
    # j = 1
    for i in 1:max_iter
        μ, X = generate(length(observation))
        if sum(X) == sum(observation)
            push!(μ_posteriori, μ)
            # μ_posteriori[j] = μ
            # j += 1
        end
    end
    return μ_posteriori
end
X_obs1 = [0, 0, 0, 1, 1]
max_i = 1_000_000
# max_i = 1_000_000

post1 = posterior(X_obs1, max_i)
acceptance_rate1 = length(post) / max_i
# println("acceptance rate = $(acceptance_rate)")

fig = Figure(size=(1200, 800))

ax1 = Axis(fig[1,1])
hist!(ax1, post1)
text!(ax1, 0.7, 0.9,  # (0.5, 0.9) = 中央上部
      text = "acceptance rate = $(acceptance_rate1)\n ($(length(post1)) / $(max_i))",
      align = (:center, :top),
      space = :relative,  # これが重要
      fontsize = 14)

X_obs2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
post2 = posterior(X_obs2, max_i)
acceptance_rate2 = length(post2) / max_i
# println("acceptance rate = $(acceptance_rate2)")

ax2 = Axis(fig[1,2])
hist!(ax2, post2)
text!(ax2, 0.7, 0.9,  # (0.5, 0.9) = 中央上部
      text = "acceptance rate = $(acceptance_rate2)\n ($(length(post2)) / $(max_i))",
      align = (:center, :top),
      space = :relative,  # これが重要
      fontsize = 14)

################################################################
# p.138 prediction
################################################################
pred1 = mean(rand.(Bernoulli.(post1)))
pred2 = mean(rand.(Bernoulli.(post2)))
pred1_mean = mean(post1)
pred2_mean = mean(post2)

vlines!(ax1, [pred1], label = "mean random post = $pred1", color = :red)
vlines!(ax2, [pred2], label = "mean random post = $pred2", color = :red)
vlines!(ax1, [pred1_mean], label = "mean post = $pred1_mean", color = :orange)
vlines!(ax2, [pred2_mean], label = "mean post = $pred2_mean", color = :orange)

axislegend(ax1)
axislegend(ax2)


function generate2(N)
    μ = rand(Uniform(0, 0.5))
    X = rand(Bernoulli(μ), N)
    μ, X
end

X_obs1 = [0, 0, 0, 1, 1]
max_i = 1_000_000
# max_i = 1_000_000

post3 = posterior(X_obs1, max_i; generate = generate2)
acceptance_rate3 = length(post3) / max_i

ax3 = Axis(fig[1,3], limits = ((0, 1), nothing))
hist!(ax3, post3)
text!(ax3, 0.7, 0.9,  # (0.5, 0.9) = 中央上部
      text = "acceptance rate = $(acceptance_rate3)\n ($(length(post3)) / $(max_i))",
      align = (:center, :top),
      space = :relative,  # これが重要
      fontsize = 14)

pred3 = mean(rand.(Bernoulli.(post3)))
pred3_mean = mean(post3)
vlines!(ax3, [pred3], label = "mean random post = $pred3", color = :red)
vlines!(ax3, [pred3_mean], label = "mean post = $pred3_mean", color = :orange)

axislegend(ax3)

fig |> display
