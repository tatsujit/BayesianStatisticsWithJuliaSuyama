# disp, save_fig, benchmark_flag = true, true, true
# disp, save_fig, benchmark_flag = true, false, true
# disp, save_fig, benchmark_flag = true, false, false
disp, save_fig, benchmark_flag = true, true, false
"""
これを使うとエラーを抑制できる？
"""
function Base.joinpath(s::AbstractString, n::Nothing)
    joinpath(s)
end
using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p181--184-linear-regression-Laplace-approximation-two-params-parallel-comp"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf
using IceCream
# using ColorSchemes
# using Colors
using Random
using ForwardDiff
using LinearAlgebra
using BenchmarkTools: @btime, @benchmark
import ProgressMeter: Progress, next!, @timed, @showprogress

include(srcdir("utility_functions.jl"))

rseed = 1234
Random.seed!(rseed)

# n次元単位行列
eye(n) = Diagonal{Float64}(I, n)
# パラメータ抽出
unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))

# input data
X_obs = [-2, 1, 5]
# output data
Y_obs = [-2.2, -1.0, 1.5]

################################################################
# plot init
################################################################
fig = Figure(
    size = (800, 800),
    figure_padding = 30,
)

x_lower, x_upper = -2, 2
y_lower, y_upper = 0, 1

aspect = 1.2

Label(fig[0, 1:3],
      "logistic regression", #, (μ1, μ2, σ1, σ2) =  ($μ1, $μ2, $σ1, $σ2)",
      fontsize = 16, font = :bold)


# data plot
ax11 = Axis(fig[1, 1], title = "data (N=$(length(X_obs)))", xlabel = L"x", ylabel = L"y")
scatter!(ax11, X_obs, Y_obs, color = :green)

################################################################
# linear regression with two parameters
################################################################
@ic "linear regression with two parameters"
# noise size on y
σ = 1.0

# mean and std of prior distribution
μ1, μ2 = 0.0, 0.0
σ1, σ2 = 10.0, 10.0

log_joint(w, X, Y, σ, μ1, σ1, μ2, σ2) =
    sum(logpdf.(Normal.(w[1] * X .+ w[2], σ), Y)) +
    logpdf(Normal(μ1, σ1), w[1]) +
    logpdf(Normal(μ2, σ2), w[2])

params = (X_obs, Y_obs, σ, μ1, σ1, μ2, σ2)
# unnormalized log-posterior and posterior
ulp(w) = log_joint(w, params...)

w1s = range(-5, 5, length=500)
w2s = range(-5, 5, length=500)

# data plot
ax12 = Axis(fig[1, 2],
            title = "unnormalized log-posterior",
            xlabel = L"w_1", ylabel = "log density (unnormalized)")
contour!(ax12, w1s, w2s,
         [ulp([w1, w2]) for w1 in w1s, w2 in w2s],
         colormap=:jet,  # または :rainbow, :turbo
         levels = 10,
         linewidth = 1,
         labels = true,
         )

ax13 = Axis(fig[1, 3],
            title = "unnormalized posterior",
            xlabel = L"w_1", ylabel = "density (unnormalized)")
contour!(ax13, w1s, w2s,
         [exp(ulp([w1, w2])) for w1 in w1s, w2 in w2s],
         colormap=:jet,  # または :rainbow, :turbo
         levels = 10,
         linewidth = 1,
         labels = true,
         )

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

# optimization parameters
w_init = [0.0, 0.0]
max_iter = 1_000
η = 0.01

# w_seq = gradient_method(log_joint, w_init, η, max_iter)
w_seq = inference_wrapper_gd(log_joint, params, w_init, η, max_iter)

# ; だけだと julia-snail では出力抑制できないので、 nothing を返すことにすると良い？
((w_seq = gradient_method(ulp, w_init, η, max_iter)); nothing)
((w_seq_wr = inference_wrapper_gd(log_joint, params, w_init, η, max_iter)); nothing)


# benchmark_flag = false

if benchmark_flag
    res = @benchmark begin
        w_seq = gradient_method(ulp, w_init, η, max_iter);
        nothing
    end
    res_wr = @benchmark begin
        w_seq_wr = inference_wrapper_gd(log_joint, params, w_init, η, max_iter);
        nothing
    end
    res_str = @sprintf("%.2f", median(res).time / 1e6) # nano to second
    res_wr_str = @sprintf("%.2f", median(res_wr).time / 1e6) # nano to second
end


ax22 = Axis(fig[2, 2],
            title = L"w_1\text{ sequence}",
            xlabel = "iteration", ylabel = L"w_1",
            )
lines!(ax22, 1:max_iter, w_seq[1,:])#, color = :blue)

ax23 = Axis(fig[2, 3],
            title = L"w_2\text{ sequence}",
            xlabel = "iteration", ylabel = L"w_2",
            )
lines!(ax23, 1:max_iter, w_seq[2, :])#, color = :blue)

if benchmark_flag
    text!(ax22, max_iter/4, 0.25,
          text = "time = $res_str ms",
          align = (:left, :bottom),
          color = :black)
    text!(ax23, max_iter/4, -0.5,
          text = "time = $res_wr_str ms (wrapped)",
          align = (:left, :bottom),
          color = :black)
end

# approximate MAP
μ_approx = w_seq[:, end]

hessian(x) = ForwardDiff.hessian(ulp, x) # scalar
Σ_approx = inv(-hessian(μ_approx))


ax33 = Axis(fig[3, 3],
            title = "unnormalized posterior",
            xlabel = L"w_1", ylabel = L"w_2")
contourf!(ax33, w1s, w2s, [exp(ulp([w1, w2]))
                          for w1 in w1s, w2 in w2s],
         colormap = :jet)

ax32 = Axis(fig[3, 2],
            title = "approximate posterior (Laplace approx.)",
            xlabel = L"w_1", ylabel = L"w_2")
contourf!(ax32, w1s, w2s, [ulp([w1, w2])
                          for w1 in w1s, w2 in w2s],
          levels=20,
          colormap=:jet,  # または :rainbow, :turbo
          nan_color=:transparent,
          )


################################################################
# predictive distribution (numerical integration)
################################################################
@ic "predictive distribution (numerical integration)"

Δ1 = w1s[2] - w1s[1]
Δ2 = w2s[2] - w2s[1]


# this version too slow:
p_predictive(x, y) = sum([pdf(Normal(w1*x+w2, σ), y) *
    pdf(MvNormal(μ_approx, Σ_approx), [w1, w2]) *
    Δ1 * Δ2 for w1 in w1s, w2 in w2s])

# so the wrapper version with params
params_p_predictive = (w1s, w2s, Δ1, Δ2, μ_approx, Σ_approx, σ)

function p_predictive_wrapped(x, y, params)
    w1s, w2s, Δ1, Δ2, μ_approx, Σ_approx, σ = params
    return sum(
        [pdf(Normal(w1 * x + w2, σ), y) *
            pdf(MvNormal(μ_approx, Σ_approx), [w1, w2]) *
            Δ1 * Δ2
         for w1 in w1s, w2 in w2s]
    )
end

################################################################
# faster version of p_predictive_wrapped by Claude Sonnet 4.5 -  2025年11月5日
################################################################
using StaticArrays
using LogExpFunctions: logsumexp

"""
Distributions.jl を使わず計算したもの
"""
function p_predictive_wrapped_optimized(x, y, params)
    w1s, w2s, Δ1, Δ2, μ_approx, Σ_approx, σ = params

    # 事前分布の逆行列とログ行列式を事前計算（外で行うべき）
    Σ_inv = inv(Σ_approx)
    log_det_Σ = logdet(Σ_approx)

    # 定数部分
    log_const_prior = -0.5 * (2 * log(2π) + log_det_Σ)
    log_const_lik = -0.5 * log(2π * σ^2)
    log_Δ = log(Δ1 * Δ2)

    # Generator で計算（配列を作らない）
    result = sum(
        let
            # 尤度の計算
            μ_lik = w1 * x + w2
            log_lik = log_const_lik - 0.5 * (y - μ_lik)^2 / σ^2

            # 事前分布の計算（手動で高速化）
            w = SVector(w1, w2)
            δ = w - μ_approx
            log_prior = log_const_prior - 0.5 * dot(δ, Σ_inv * δ)

            # 対数空間で計算
            exp(log_lik + log_prior + log_Δ)
        end
        for w1 in w1s, w2 in w2s
    )

    return result
end

"""
こちらがオリジナルに近く、適切な工夫をしたバージョン
"""
function p_predictive_wrapped_logspace(x, y, params)
    w1s, w2s, Δ1, Δ2, μ_approx, Σ_approx, σ = params

    prior = MvNormal(μ_approx, Σ_approx)

    # 対数空間で計算（logsumexp）
    log_weights = [
        logpdf(Normal(w1 * x + w2, σ), y) +
        logpdf(prior, [w1, w2]) +
        log(Δ1 * Δ2)
        for w1 in w1s, w2 in w2s
    ]

    # logsumexp for numerical stability
    max_log = maximum(log_weights)
    return exp(max_log) * sum(exp.(log_weights .- max_log))
end

# plot
# xs = range(-10, 10, length=100)
# ys = range(-5, 5, length=100)
xs_length, ys_length = 100, 100
xs = range(-10, 10, length=xs_length)
ys = range(-5, 5, length=ys_length)

@ic "heavy p_predictive calculation with $(length(xs)) x $(length(ys)) loops"

################ as is in the book ################
# @time density_y = p_predictive.(xs, ys')
# xs_length, ys_length = 10, 10:
#  11.067960 seconds (250.68 M allocations: 11.407 GiB, 4.89% gc time, 1.58% compilation time)

################ wrapped version ################
# @time density_y = p_predictive_wrapped.(xs, ys', Ref(params_p_predictive))
# xs_length, ys_length = 10, 10:
#   5.694825 seconds (100.23 M allocations: 8.025 GiB, 10.13% gc time, 1.14% compilation time)

################ parallel and wrapped version ################
################ 遅くなってる ################
density_y_ = zeros(xs_length, ys_length)
update_interval = (xs_length * ys_length) ÷ 200

p = Progress(xs_length * ys_length; desc="Processing: ", dt=0.5) # スレッドセーフなプログレスバーを作成
# Threads.@threadsは、イテレータを複数のチャンクに分割するためにfirstindex()とlastindex()を必要としますが、enumerate()はこれらをサポートしていません。
# p_predictive_wrapped で xs_length = ys_length = 10 で 3分くらい
# p_predictive_wrapped で xs_length = ys_length = 100 で 24時間くらい？
# TODO p_predictive_wrapped_optimmized で xs_length = ys_length = 10 で:
# TODO p_predictive_wrapped_optimmized で xs_length = ys_length = 100 で:
@time Threads.@threads for i in eachindex(xs)
    for j in eachindex(ys)
        k = ((i-1)*ys_length + j)
        x, y = xs[i], ys[j]
        # density_y_[i, j] = p_predictive_wrapped(x, y, params_p_predictive)
        density_y_[i, j] = p_predictive_wrapped_optimized(x, y, params_p_predictive)
        # ( (i-1)*ys_length + j -1 ) % ( (xs_length*ys_length) ÷ 100 ) == 0 && @ic i j density_y_[i, j]
        # 100回に1回だけ更新
        if k % update_interval == 0
            next!(p)
        end
    end
end
# xs_length, ys_length = 10, 10:
#  17.380511 seconds (100.35 M allocations: 8.036 GiB, 3.91% gc time, 5.41% compilation time)

@ic Threads.nthreads() " threads used."
## single thread version
# for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
#     density_y_[i, j] = p_predictive_wrapped(x, y, params_p_predictive)
#     # density_y_[i, j] = p_predictive(x, y)
#     i % 10 == 0 && j % 10 == 0 && @ic @sprintf("x=%.2f, y=%.2f, p=%.5f", x, y, density_y_[i, j])
# end
# density_y = density_y_

ax21 = Axis(fig[2, 1], title = "predictive dist. (contour)",
            xlabel = L"x", ylabel = L"y")
contour!(ax21, xs, ys, density_y, colormap = :jet)
scatter!(ax21, X_obs, Y_obs, color = :black)

ax31 = Axis(fig[3, 1], title = "predictive dist. (heatmap)",
            xlabel = L"x", ylabel = L"y")
heatmap!(ax31, xs, ys, density_y, colormap = :jet)
scatter!(ax31, X_obs, Y_obs, color = :black)

################################################################
# display and save plot
################################################################
@ic "display and save plot"

disp && fig |> display
# save_fig && safesave(plotsdir(program_name * "_colorscheme=" * cs * "_.pdf"), fig)
save_fig && safesave(plotsdir(program_name * ".pdf"), fig)
