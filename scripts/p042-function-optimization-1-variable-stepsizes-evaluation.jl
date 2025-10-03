################################################################
# stepsize と最適化の結果の関係を網羅的に把握したい
################################################################
using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
using DataFrames
using CairoMakie
using LaTeXStrings
using ColorSchemes

program_name = "p042-function-optimization-1-variable-stepsizes-evaluation"

using ForwardDiff

# optimization method for 1 variable continuous (differentiable) function
# 1. initial guess x_1, max iterations maxiter ≥ 2, step size η > 0
# 2. for 2 ≤ i ≤ maxiter:
#   x_i = x_{i-1} + η * f'(x_{i-1})


################################################################"
# f(x) = -2(x-x_opt)^2
################################################################"

#1変数関数の最適化
function gradient_method_1dim(f, x_init, η, max_iter)
    # 最適化過程のパラメータを格納する配列
    x_seq = Array{typeof(x_init), 1}(undef, max_iter)
    #  勾配
    df(x) = ForwardDiff.derivative(f, x)
    # 初期値
    x_seq[1] = x_init
    # メインの最適化ループ
    for i in 2:max_iter
        x_seq[i] = x_seq[i-1] + η * df(x_seq[i-1])
    end
    x_seq
end

"""
    sign_changes(v::AbstractVector{<:Real}) -> Int

実数ベクトルの符号切り替わり回数を返す。
ゼロは無視される。
"""
function sign_changes(v::AbstractVector{<:Real})
    # ゼロでない要素の符号のみを抽出
    signs = sign.(filter(!iszero, v))
    # 符号が変わった回数をカウント
    return count(i -> signs[i] != signs[i+1], 1:length(signs)-1)
end

x_init = -2.5
max_iter = 200
# η = 0.2
# ηs = [0.2]
# ηs = [0.05, 0.2, 0.4]
# ηs = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
# ηs = vcat([0.0, 0.01, 0.02, 0.05], 0.1:0.1:1.0)
ηs = collect(0.0:0.01:1.0)
config_seed = Dict(
    :η => ηs,
    :x_init => x_init,
    :max_iter => max_iter,
)
configs = dict_list(config_seed)

x_opt = 0.50
y_opt = 0.0
f(x) = -2 * (x - x_opt)^2
df(x) = ForwardDiff.derivative(f, x)

function opt(config)
    @unpack η, x_init, max_iter = config
    x_seq = gradient_method_1dim(f, x_init, η, max_iter)
    f_seq = f.(x_seq)
    x_errs = x_seq .- x_opt
    f_errs = f_seq .- y_opt
    x_sign_switch = sign_changes(x_errs)
    f_sign_switch = sign_changes(f_errs)
    return @strdict η x_init max_iter x_seq f_seq x_errs f_errs x_sign_switch f_sign_switch
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
@show data

# check_iters = [5, 10, 20, 50]
check_iters = [5, 10, 20, 50, 200]

# data = transform(data,
#                  [:x_seq => ByRow(xs -> xs[iter]) => Symbol("x_seq_$(lpad(iter, 2, '0'))")
#                   for iter in check_iters]...,
#                  [:f_seq => ByRow(fs -> fs[iter]) => Symbol("f_seq_$(lpad(iter, 2, '0'))")
#                   for iter in check_iters]...
#                       )

data = filter(row->row.max_iter==200, data)
data = sort(data, :η)

################################################################
# plot
################################################################
colors = tol_vibrant = ColorSchemes.tol_vibrant    # 7色（鮮やか）
x_limits = (nothing, (10^(-17), 10^5))
f_limits = (nothing, (10^(-34), 10^10))
fig = Figure(size = (800, 600))
Label(fig[0, 1:2],
      L"\text{Maximize }f(x)=-x^2+1\text{ with various }ηs, \text{ }x_{\text{init}} = %$x_init",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)
ax = Axis(fig[1, 1],
          title = "x errors (points and lines disappear when 0)", xlabel = L"\eta", ylabel = L"|x\text{ error}|",
          yscale = log10,
          limits = x_limits,
          )
for (i, iter) in enumerate(check_iters)
    xs = [abs(data.x_errs[j][iter]) for j in 1:nrow(data)]
    lines!(ax, ηs, xs, color = colors[i])
    scatter!(ax, ηs, xs, color = colors[i], label = "x error at $iter", marker = :diamond)
end
axislegend(ax; position = :rb)

ax = Axis(fig[1, 2],
          title = L"f(x)\text{ errors (points and lines disappear when 0)}", xlabel = L"\eta", ylabel = L"|f(x)\text{ error}|",
          yscale = log10,
          limits = f_limits,
          )
for (i, iter) in enumerate(check_iters)
    xs = [abs(data.f_errs[j][iter]) for j in 1:nrow(data)]
    lines!(ax, ηs, xs, color = colors[i])
    scatter!(ax, ηs, xs, color = colors[i], label = "f error at $iter", marker = :diamond)
end
axislegend(ax; position = :rb)

fig |> display
# safesave(plotsdir(program_name * "_η=$(η)_" * ".pdf"), fig)
safesave(plotsdir(program_name * ".pdf"), fig)



# for η in ηs
#     fig = Figure(resolution = (800, 600))

#     Label(fig[0, 1:3], L"x_{\text{init}}=$(x_init), η = $η",
#           fontsize = 16, font = :bold)

#     xs = range(-3, 3, length=100)
#     ax = Axis(fig[1, 1],
#               title = L"-x^2+1", xlabel = L"x", ylabel = L"y")
#     lines!(ax, xs, f.(xs), color = :blue, label = "function")
#     scatter!(ax, [x_opt], [f(x_opt)], color = :red, label = "optimal")
#     axislegend(ax; position = :rb)



#     ax2 = Axis(fig[1, 2:3],
#                title = L"\max\text{ }f", xlabel = "iteration", ylabel = L"f")
#     lines!(ax2, 1:maxiter, f_seq, color = :green,
#            # label = "optimization path",
#            )
#     scatter!(ax2, 1:maxiter, f_seq, color = :green, marker = :diamond, label = L"f(x)\text{ sequence}")
#     hlines!(ax2, [y_opt], color = :red, label = L"\max\text{ }f")
#     axislegend(ax2; position = :rb)

#     ax3 = Axis(fig[2, 1],
#                title = L"\max f", xlabel = L"x", ylabel = L"f")
#     lines!(ax3, xs, f.(xs), color = :blue, label = "function")
#     scatter!(ax3, x_seq, f_seq, color = :green, marker = :diamond, label = "optimization path")
#     scatter!(ax3, [x_opt], [f(x_opt)], color = :red, label = "optimal")
#     axislegend(ax3; position = :rb)

#     ax4 = Axis(fig[2, 2:3],
#                title = L"x", xlabel = "iteration", ylabel = L"x")
#     lines!(ax4, 1:maxiter, x_seq, color = :green, label = L"x\text{ sequence}")
#     scatter!(ax4, 1:maxiter, x_seq, marker = :diamond, color = :green,
#              #label = "function"
#              )
#     hlines!(ax4, [x_opt], color = :red, linestyle = :dash, label = "optimal x")
#     axislegend(ax4; position = :rb)

#     fig |> display
#     # safesave(plotsdir(program_name * "_η=$(η)_" * ".pdf"), fig)
# end
