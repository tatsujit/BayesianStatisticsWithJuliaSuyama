# disp, save_fig = true, true
disp, save_fig = true, false
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
program_name = "p160--168-logistic-regression-ancestral-sampling"
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
# parameters
################################################################
μ1 = 0
μ2 = 0
σ1 = 10.0
σ2 = 10.0

# visualization range
xs = range(-2, 2, length=100)

n_simulation = 3 # 6
n_row, n_col = n2grid(n_simulation)

################################################################
# data
################################################################
# 入カデータセット
X_obs = [-2, 1, 2]
# 出カデータセット
Y_obs = Bool.([0, 1, 1])

ax11 = Axis(fig[1, 1], title = "data (N=$(length(X_obs)))", xlabel = L"x", ylabel = L"y")
scatter!(ax11, X_obs, Y_obs, color = :green)

################################################################
# sampling
################################################################
# 最大サンプリング数
max_iter = 10_000
# パラメータ保存用
param_posterior = Vector{Tuple{Float64,Float64}}()
for i in 1:max_iter
    # 関数f, 出力Yの生成
    Y, f, w1, w2 = generate_logistic(X_obs, μ1, μ2, σ1, σ2)
    # 観測データと一致していれば,そのときのパラメータwを保存
    Y == Y_obs && push!(param_posterior, (w1, w2))
end
# サンプル受容率
acceptance_rate = length(param_posterior) / max_iter
# println("acceptance rate = $(acceptance_rate)")


# パラメータ抽出用の関数
unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))
# 事前分布からのサンプル (10,000組)
param_prior = [generate_logistic(X_obs, μ1, μ2, σ1, σ2)[3:4]
               for i in 1:max_iter]
w1_prior, w2_prior = unzip(param_prior)

# 事後分布からのサンプル
w1_posterior, w2_posterior = unzip(param_posterior)


# プロット
plot_dist_limits = ((-40, 40), (-40, 40))
alpha = 0.1

Label(fig[0, 2:3],
      "acceptance rate = $(acceptance_rate)",
      # "acceptance rate = $(acceptance_rate) = $(length(param_posterior)) / $(max_iter)",
      # #, (μ1, μ2, σ1, σ2) =  ($μ1, $μ2, $σ1, $σ2)",
      fontsize = 10, font = :bold)

ax2 = Axis(fig[1, 2], title = "prior param. dist.", xlabel = L"w_1", ylabel = L"w_2", limits = plot_dist_limits)
scatter!(ax2, w1_prior, w2_prior, alpha = alpha)

ax3 = Axis(fig[1, 3], title = "post. param. dist.", xlabel = L"w_1", ylabel = L"w_2", limits = plot_dist_limits)
scatter!(ax3, w1_posterior, w2_posterior, alpha = alpha)

# 関数を可視化する範囲
xs = range(-3, 3, length=100)

################################################################
# sampling from posterior
################################################################
for i in 1:n_simulation
    # 関数を可視化するための w を 1つ適当に選択
    j = round(Int, length(param_posterior) * rand()) + 1
    w1, w2 = param_posterior[j]

    #選択されたw
    scatter!(ax3, w1, w2, color = :red)
    text!(ax3, w1, w2, text = "$i", color = :yellow, align = (:left, :bottom))

    # サンプリングした例のプロット
    row, col = n2ij(i, n_simulation)
    w1_str, w2_str = @sprintf("%.2f", w1), @sprintf("%.2f", w2)
    ax = Axis(fig[row + 1, col],
              xlabel = L"x", ylabel = L"y \text{(prob.)}",
              title = L"\text{(%$i) (w_1, w_2) = (%$w1_str, %$w2_str)}")
    # 対応する関数のプロット
    f(x) = sig(w1 * x + w2)
    lines!(ax, xs, f, color = :red)
    #観測データのプロット
    scatter!(ax, X_obs, Y_obs, color = :green)

    #axes[i].set_y■ im〈 [-0.1,1.1コ ) set_options(axes[i], ?x? ,"y(PrOb.)", "(S(1))wl=$〈 round(wl,digits=3)), W2=$(rOund〈 w2,digits=3))")
end

################################################################
# visualize functions from posterior params
################################################################
ax31 = Axis(fig[3, 1],
            xlabel = L"x", ylabel = L"y \text{(prob.)}",
            title = L"\text{function samples}")
fs = []
for (i, param) in enumerate(param_posterior)
    w1, w2 = param
    f(x) = sig(w1 * x + w2)
    # 対応する関数のプロット
    lines!(ax31, xs, f.(xs), color = (:blue, 0.01))
    push!(fs, f.(xs))
end
#観測データのプロット
scatter!(ax31, X_obs, Y_obs, color = :green)

################################################################
# mean prediction
################################################################
ax32 = Axis(fig[3, 2],
          xlabel = L"x", ylabel = L"y \text{(prob.)}",
          title = L"\text{mean prediction}")
lines!(ax32, xs, mean(fs), label = "prediction")
scatter!(ax32, X_obs, Y_obs, color = :green, label = "data")
axislegend(ax32, position = :rc, backgroundcolor = (:white, 0.3))



################################################################
# prediction for specific xs
################################################################
x_list = [-1.0, 0.0, 1.5]

for (j, x) in enumerate(x_list)
    ax_f = Axis(fig[j, 4], title = "function samples")
    for (i, param) in enumerate(param_posterior)
        w1, w2 = param
        f(x) = sig(w1 * x + w2)
        # 対応する関数のプロット
        lines!(ax_f, xs, f.(xs), color = (:blue, 0.01))
    end
    # data
    scatter!(ax_f, X_obs, Y_obs, color = :green, label = "data")
    # prediction at x_p
    vlines!(ax_f, x, label = L"x_p = %$x", color = :red, linestyle = :dash)
    axislegend(ax_f)

    # histogram of the function value (probability) at the point x_p
    probs = [sig(w1*x + w2) for (w1, w2) in param_posterior]
    ax_h = Axis(fig[j, 5], title = "probability frequency")
    hist!(ax_h, probs, direction = :x)
end

################################################################
# p. 166 numerical integration
################################################################
p_joint(X, Y, w1, w2) = prod(pdf.(Bernoulli.(sig.(w1*X .+ w2)), Y)) *
    pdf(Normal(μ1, σ1), w1) * pdf(Normal(μ2, σ2), w2)

w_range = range(-30, 30, length=100)

function approx_integration_2D(w_range, p)
    Δ = w_range[2] - w_range[1]
    (X, Y) -> sum([p(X, Y, w1, w2) * Δ^2 for w1 in w_range, w2 in w_range])
end

p_marginal = approx_integration_2D(w_range, p_joint)
p_marginal(X_obs, Y_obs) # 0.2966829565005522

w_posterior = [p_joint(X_obs, Y_obs, w1, w2)
               for w1 in w_range, w2 in w_range] ./ p_marginal(X_obs, Y_obs)

ax33 = Axis(fig[3, 3], xlabel=L"w_1", ylabel=L"w_2", title = "posterior density")
cf = contourf!(ax33, w_range, w_range, w_posterior,
               levels=20,
               colormap=:jet,  # または :rainbow, :turbo
               nan_color=:transparent,
               )


################################################################
# predictive distribution
################################################################
function approx_predictive(w_posterior, w_range, p)
    Δ = w_range[2] - w_range[1]
    return (x, y) -> sum([p(x, y, w1, w2) * w_posterior[i, j] * Δ^2
                          for (i, w1) in enumerate(w_range),
                              (j, w2) in enumerate(w_range)])
end

p_likelihood(xp, yp, w1, w2) = pdf(Bernoulli(sig(w1*xp + w2)), yp)
p_predictive = approx_predictive(w_posterior, w_range, p_likelihood)

# recycle ax11
lines!(ax11, xs, p_predictive.(xs, 1), label = "probability")
scatter!(ax11, X_obs, Y_obs, color = :green, label = "data")
ax11.xlabel, ax11.ylabel, ax11.title = L"x", L"y", "data and prediction"
axislegend(ax11)

################################################################
# display and save plot
################################################################

disp && fig |> display
# save_fig && safesave(plotsdir(program_name * "_colorscheme=" * cs * "_.pdf"), fig)
save_fig && safesave(plotsdir(program_name * ".pdf"), fig)
