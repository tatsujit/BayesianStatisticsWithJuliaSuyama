using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p142-144-posteriors"
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

function p_joint(X, μ)
    likelihood = prod(pdf.(Bernoulli(μ), X))
    prior = pdf(Uniform(0, 1), μ)
    return likelihood * prior
end

function approx_integration(μ_range, p)
    Δ = μ_range[2] - μ_range[1]
    marginal = X -> sum([p(X, μ) * Δ for μ in μ_range])
    return marginal, Δ
end


X_obs1 = [0, 0, 0, 1, 1]
X_obs2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
X_obss = [X_obs1, X_obs2]

μs = range(0, 1, length=100)
################################################################
# plot
################################################################
fig = Figure(size=(1200, 400))

for (i, X_obs) in enumerate(X_obss)
    ax = Axis(fig[1, i],
              xlabel = "μ",
              ylabel = "density",
              title = "posteiors for X_obs$i = $(X_obs)",
              )
    ################################################################
    # approximate posterior
    ################################################################
    μ_range = range(0, 1, length=3) # length=100
    @ic μ_range
    p_marginal, Δ = approx_integration(μ_range, p_joint)
    posterior(μ) = p_joint(X_obs, μ) / p_marginal(X_obs)
    lines!(ax, μs, posterior, color = :blue,
           label = "approximate post. (length(μ_range)=$(length(μ_range)))"
           )
    ################################################################
    # predictive distributions
    ################################################################
    # 積分の中身の式
    posterior(μ) = p_joint(X_obs, μ) / p_marginal(X_obs)
    p_inner(X, μ) = pdf.(Bernoulli(μ), X) * posterior(μ)
    # バラメータμに関する積分
    μ_range = range(0, 1, length=100)
    pred, △ = approx_integration(μ_range, p_inner)
    scatter!(ax, [pred(1)], [0], label = "prediction of 1: $(pred(1))", color = :red)
    ax2 = Axis(fig[2, i],
               title = "prediction of $i: $(pred(1))",
               )
    hidedecorations!(ax2)  # 軸ラベル、目盛り、グリッドをすべて非表示
    hidespines!(ax2)       # 軸の枠線を非表示

    ################################################################
    # exact solutions
    ################################################################
    α = 1 + sum(X_obs)
    β = 1 + length(X_obs) - sum(X_obs)
    d = Beta(α, β)
    lines!(ax, μs, pdf.(d, μs), color = :red, linestyle = :dash,
           label = "exact solution Beta($α, $β)")
    vlines!(ax, [mean(d)], label = "mean of Beta($α, $β)", color = :blue)
    vlines!(ax, [median(d)], label = "median of Beta($α, $β)", color = :green)
    vlines!(ax, [mode(d)], label = "mode of Beta($α, $β)", color = :orange)
    axislegend(ax, position = :rt, backgroundcolor = (:white, 0.75))
end

# 列の幅を相対的に設定
rowsize!(fig.layout, 1, Relative(8/9))
rowsize!(fig.layout, 2, Relative(1/9))

fig |> display
safesave(plotsdir(program_name * ".pdf"), fig)
