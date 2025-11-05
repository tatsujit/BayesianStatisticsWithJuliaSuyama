disp, save_fig = true, true
# disp, save_fig = true, false
# disp, save_fig = false, true
# disp, save_fig = false, false
"""
これを使うとエラーを抑制できる？
"""
function Base.joinpath(s::AbstractString, n::Nothing)
    joinpath(s)
end
using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p185-logistic-regression-Laplace-approximation"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf
using IceCream
# using ColorSchemes
# using Colors
using Random

include(srcdir("utility_functions.jl"))

rseed = 1234
Random.seed!(rseed)

################################################################
# logistic regression
################################################################
function generate_logistic(X, μ1, μ2, σ1, σ2)
    w1 = rand(Normal(μ1, σ1))
    w2 = rand(Normal(μ2, σ2))
    f(x) = sig(w1 * x + w2)
    Y = rand.(Bernoulli.(f.(X)))
    return Y, f, w1, w2
end

################################################################
# plot init
################################################################
fig = Figure(
    size = (1200, 800),
    figure_padding = 30,
)

x_lower, x_upper = -2, 2
y_lower, y_upper = 0, 1

aspect = 1.2

Label(fig[-1, 1:3],
      "logistic regression", #, (μ1, μ2, σ1, σ2) =  ($μ1, $μ2, $σ1, $σ2)",
      fontsize = 16, font = :bold)

################################################################
# data
################################################################
# 入カデータセット
X_obs = [-2, 1, 2]
# 出カデータセット
Y_obs = Bool.([0, 1, 1])

################################################################
# parameters
################################################################
μ1 = 0
μ2 = 0
σ1 = 10.0
σ2 = 10.0

log_joint(w, X, Y, μ1, σ1, μ2, σ2) =
    logpdf(Normal(μ1, σ1), w[1]) +
    logpdf(Normal(μ2, σ2), w[2]) +
    sum(logpdf.(Bernoulli.(sig.(w[1] .* X_obs .+ w[2])), Y_obs))

params = (X_obs, Y_obs, μ1, σ1, μ2, σ2)

ulp(w) = log_joint(w, params...)

w_init = [0.0, 0.0]
max_iter = 2_000
η = 0.1

################################################################
# gradient method to find the maximum a posteriori (MAP)
################################################################
@ic "gradient method to find the maximum a posteriori (MAP)"
function gradient_method(f, x_init, η, max_iter)
    g(x) = ForwardDiff.gradient(f, x)
    x_seq = Array{typeof(x_init[1]), 2}(undef, length(x_init), max_iter)
    x_seq[:, 1] .= x_init
    for i in 2:max_iter
        # maximum を探すから - ではなく +
        x_seq[:, i] = x_seq[:, i-1] + η * g(x_seq[:, i-1])
    end
    return x_seq
end

function inference_wrapper_gd(log_joint, params, w_init, η, max_iter)
    ulp(w) = log_joint(w, params...)
    w_seq = gradient_method(ulp, w_init, η, max_iter)
    return w_seq
end

w_seq = inference_wrapper_gd(log_joint, params, w_init, η, max_iter)


ax21 = Axis(fig[2, 1],
            title = L"w_1\text{ sequence}",
            xlabel = "iteration", ylabel = L"w_1",
            )
lines!(ax21, 1:max_iter, w_seq[1,:])#, color = :blue)

ax22 = Axis(fig[2, 2],
            title = L"w_2\text{ sequence}",
            xlabel = "iteration", ylabel = L"w_2",
            )
lines!(ax22, 1:max_iter, w_seq[2, :])#, color = :blue)

################################################################
# approx
################################################################
μ_approx = w_seq[:, end]
hessian(w) = ForwardDiff.hessian(ulp, w)
Σ_approx = inv(-hessian(μ_approx))


w1s = range(-10, 30, length=500)
w2s = range(-20, 20, length=500)

# data plot
ax11 = Axis(fig[1, 1],
            title = "unnormalized posterior",
            xlabel = L"w_1", ylabel = L"w_2")
contour!(ax11, w1s, w2s,
         [exp(ulp([w1, w2])) + eps() for w1 in w1s, w2 in w2s],
         colormap=:jet,  # または :rainbow, :turbo
         levels = 10,
         linewidth = 1,
         labels = true,
         )
vlines!(ax11, [0.0], color = :red)

ax12 = Axis(fig[1, 2],
            title = "approximate posterior (Laplace method)",
            xlabel = L"w_1", ylabel = L"w_2")
contour!(ax12, w1s, w2s,
         [pdf(MvNormal(μ_approx, Σ_approx), [w1, w2])
          for w1 in w1s, w2 in w2s],
         colormap=:jet,  # または :rainbow, :turbo
         levels = 10,
         linewidth = 1,
         labels = true,
         )
vlines!(ax12, [0.0], color = :red)


################################################################
# sampling from the approximate dist.
################################################################
W = rand(MvNormal(μ_approx, Σ_approx), 100)

xs = range(-10, 10, length=100)

ax13 = Axis(fig[1, 3],
            title = "prediction samples from the approximate posterior",
            xlabel = L"x", ylabel = L"y")
for i in 1:size(W, 2)
    w1, w2 = W[:, i]
    f(x) = sig(w1 * x + w2)
    lines!(ax13, xs, f.(xs), color = (:green, 0.1))
end

ax03 = Axis(fig[0, 3],
            title = "prediction samples from the approximate posterior",
            xlabel = L"w_1", ylabel = L"w_2")
contour!(ax03, w1s, w2s,
         [pdf(MvNormal(μ_approx, Σ_approx), [w1, w2])
          for w1 in w1s, w2 in w2s],
         colormap=:jet,  # または :rainbow, :turbo
         levels = 10,
         linewidth = 1,
         labels = true,
         )
scatter!(ax03, W[1, :], W[2, :])

################################################################
# predictive dist. by numerical integration
################################################################

Δ1 = w1s[2] - w1s[1]
Δ2 = w2s[2] - w2s[1]
# this version too slow:
p_predictive(x, y) = sum([pdf(Bernoulli(sig(w1*x+w2)), y) *
    pdf(MvNormal(μ_approx, Σ_approx), [w1, w2]) *
    Δ1 * Δ2 for w1 in w1s, w2 in w2s])

# xs = range(-10, 10, length=100)

ax23 = Axis(fig[2, 3],
            title = "posterior prediction from the approximate posterior",
            xlabel = L"x", ylabel = L"y")
lines!(ax23, xs, p_predictive.(xs, 1), label = "probability")
scatter!(ax23, X_obs, Y_obs, label = "data")
axislegend(ax23)



################################################################
# display and save plot
################################################################

disp && fig |> display
# save_fig && safesave(plotsdir(program_name * "_colorscheme=" * cs * "_.pdf"), fig)
save_fig && safesave(plotsdir(program_name * ".pdf"), fig)
