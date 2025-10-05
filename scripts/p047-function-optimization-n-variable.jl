# optimization method for n variable continuous (differentiable) function
# 1. initial guess x_1, max iterations maxiter ≥ 2, step size η > 0
# 2. for 2 ≤ i ≤ maxiter:
#   x_i = x_{i-1} + η * ∇f(x_{i-1})

using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p047-function-optimization-n-variable"
using DataFrames
using CairoMakie
using LaTeXStrings
using ColorSchemes
using IceCream
using ForwardDiff
using Revise
include(srcdir("utility_functions.jl"))

################################################################"
# f(x)
################################################################"

#1変数関数の最適化
function gradient_method(f, x_init, η, max_iter)
    # 最適化過程のパラメータを格納する配列
    x_seq = Array{typeof(x_init[1]), 2}(undef, length(x_init), max_iter)
    @ic typeof(x_seq)
    #  勾配
    # ∇f(x) = ForwardDiff.derivative(f, x) # derivative is for 1-variable function
    ∇f(x) = ForwardDiff.gradient(f, x) # gradient is for n-variable function
    # 初期値
    x_seq[:, 1] .= x_init
    # メインの最適化ループ
    for i in 2:max_iter
        x_seq[:, i] = x_seq[:, i-1] + η * ∇f(x_seq[:, i-1])
    end
    x_seq
end

x_init = (-0.75, -0.75) # []だと dict_list() が展開しちゃう
max_iter = 20
# η = 0.1
ηs = [0.1]
# ηs = [0.05, 0.2, 0.4]
# ηs = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
# ηs = vcat([0.0, 0.01, 0.02, 0.05], 0.1:0.1:1.0)
# ηs = collect(0.0:0.01:1.0)
config_seed = Dict(
    :η => ηs,
    :x_init => x_init,
    :max_iter => max_iter,
)
configs = dict_list(config_seed)

x_opt = [0.50, 0.25]
y_opt = -sqrt(0.05) # probably
L = 100
xs1 = range(-1, 1, length=L)
xs2 = range(-1, 1, length=L)
f(x) = -sqrt(0.05 + (x[1] - x_opt[1])^2) - (x[2] - x_opt[2])^2
f_string = "-\\sqrt{0.05 + (x_1 - 0.5)^2} - (x_2 - 0.25)^2"
∇f(x) = ForwardDiff.gradient(f, x) # gradient is for n-variable function

function opt(config)
    @unpack η, x_init, max_iter = config
    x_seq = gradient_method(f, x_init, η, max_iter)
    f_seq = [f(x_seq[:,i]) for i in 1:max_iter]
    x_errs = x_seq .- x_opt
    f_errs = f_seq .- y_opt
    # x_sign_switch = sign_changes(x_errs)
    # f_sign_switch = sign_changes(f_errs)
    return @strdict η x_init max_iter x_seq f_seq x_errs f_errs #x_sign_switch f_sign_switch
end

Threads.@threads for config in configs
    produce_or_load(opt,
                    config,
                    datadir(program_name),
                    ;
                    # force = true,
                    )
end

data = collect_results!(
    # joinpath(scriptsdir() * program_name * ".jld2"), # where to save the collected results こっちでも良い
    # joinpath(scriptsdir(program_name * ".jld2")), # where to save the collected results これでも
    scriptsdir(program_name * ".jld2"), # where to save the collected results
    datadir(program_name); # where to collect the individual results
    update = true,
    verbose = true,
)
# @show data

f_seq = data[1,:].f_seq
x_seq = data[1,:].x_seq

################################################################
# plot
################################################################
colors = tol_vibrant = ColorSchemes.tol_vibrant    # 7色（鮮やか）
# x_limits = (nothing, (10^(-3), 10^10))
# f_limits = (nothing, (10^(-6), 10^20))
fig = Figure(size = (1000, 600))

################################################################
# entire plot label
################################################################
Label(fig[0, 1:3],
      L"\text{Maximize }f(x)=%$(f_string)\text{ with }\eta = %$(ηs[1]), \text{ }x_{\text{init}} = %$x_init",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)
################################################################
# function contour plot with optimum
################################################################
ax = Axis(fig[1, 1],
          title = L"f(x_1, x_2)", xlabel = L"x_1", ylabel = L"x_2",
          # yscale = log10,
          # limits = x_limits,
          )
contour!(ax, xs1, xs2, [f([x1, x2]) for x1 in xs1, x2 in xs2],
         levels = 0.0:-0.1:-3.0,
         )
scatter!(ax, x_opt...,
         marker = :cross,
         markersize = 14,
         label = "optimal"
         )
axislegend(ax; rt = :position)

################################################################
# iteration vs f
################################################################
ax = Axis(fig[1, 2:3],
          title = "value optimization process", xlabel = "iteration", ylabel = L"f",
          # yscale = log10,
          # limits = x_limits,
          )
lines!(ax, 1:max_iter, f_seq)
scatter!(ax, 1:max_iter, f_seq,
         marker = :cross,
         markersize = 14,
         label = "f sequence",
         )
hlines!(ax, [y_opt], color = :purple, linestyle = :dash, label = L"\text{optimal }f")
axislegend(ax; position = :rb)


################################################################
# function contour plot with optimum
################################################################
ax = Axis(fig[2, 1],
          title = L"f'(x_1, x_2)", xlabel = L"x_1", ylabel = L"x_2",
          # yscale = log10,
          # limits = x_limits,
          )
# grid points
M = div(L, 8)
xs1s = range(-1, 1, length=M)
xs2s = range(-1, 1, length=M)
x1 = repeat(xs1s, 1, M)
x2 = repeat(xs2s', M, 1)
# gradient field
u = [∇f([x, y])[1] for x in xs1s, y in xs2s]
v = [∇f([x, y])[2] for x in xs1s, y in xs2s]
quiver!(ax,
        vec(x1), vec(x2),
        vec(u), vec(v),
        lengthscale = 0.1,
        arrowsize = 10,
        arrowcolor = :blue,
        linecolor = :black
        )
# arrows!(ax,
#         vec(x1), vec(x2),      # 始点座標
#         vec(u), vec(v),  # 正規化したベクトル成分
#         lengthscale = 0.3,     # 矢印の長さスケール
#         arrowsize = 15,        # 矢印サイズ（v0.23以降でも使用可能）
#         arrowcolor = :blue,    # 矢印全体の色
#         linewidth = 1.5        # 軸の太さ
# )
# arrows2d!(ax,
#         vec(x1), vec(x2),  # 始点座標
#         vec(u), vec(v),     # ベクトル成分
#         lengthscale = 0.1,
#         tipwidth = 10,      # arrowsize → tipwidth & tiplength
#         tiplength = 10,
#         tipcolor = :blue,   # arrowcolor → tipcolor
#         shaftcolor = :black # linecolor → shaftcolor
# )


################################################################
# f-contour with x sequence
################################################################
ax = Axis(fig[2, 2],
          title = L"f(x_1, x_2)", xlabel = L"x_1", ylabel = L"x_2",
          # yscale = log10,
          # limits = x_limits,
          )
contour!(ax, xs1, xs2, [f([x1, x2]) for x1 in xs1, x2 in xs2],
         levels = 0.0:-0.1:-3.0,
         )
scatter!(ax, x_seq[1,:], x_seq[2,:], marker=:cross, label="x sequence",
         )
axislegend(ax; position = :rt)

################################################################
# optimization process
################################################################
ax = Axis(fig[2, 3],
          title = "input optimization process", xlabel = "iteration", ylabel = L"x_1, x_2",
          # yscale = log10,
          # limits = x_limits,
          )
lines!(ax, 1:max_iter, x_seq[1,:], label=L"x_1", color = :blue)
lines!(ax, 1:max_iter, x_seq[2,:], label=L"x_2", color = :red)
scatter!(ax, 1:max_iter, x_seq[1,:], color = :blue, marker = :circle)
scatter!(ax, 1:max_iter, x_seq[2,:], color = :red, marker = :cross)
hlines!(ax, [x_opt[1]], color = :blue, linestyle = :dash, label = L"\text{optimal }x_1")
hlines!(ax, [x_opt[2]], color = :red, linestyle = :dash, label = L"\text{optimal }x_2")
axislegend(ax; position = :rb)


fig |> display
# safesave(plotsdir(program_name * "_η=$(η)_" * ".pdf"), fig)
safesave(plotsdir(program_name * ".pdf"), fig)


