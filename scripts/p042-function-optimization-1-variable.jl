using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
using CairoMakie
using LaTeXStrings

program_name = "p042-function-optimization-1-variable"

using ForwardDiff

# optimization method for 1 variable continuous (differentiable) function
# 1. initial guess x_1, max iterations maxiter ≥ 2, step size η > 0
# 2. for 2 ≤ i ≤ maxiter:
#   x_i = x_{i-1} + η * f'(x_{i-1})

fig = Figure(resolution = (800, 600))

################################################################"
# f(x) = -2(x-x_opt)^2
################################################################"
x_init = -2.5
maxiter = 20
# η = 0.25
# η = 0.2
η = 0.1


x_opt = 0.50
y_opt = 0.0
f(x) = -2 * (x - x_opt)^2
f_string = "-2(x-0.5)^2"
df(x) = ForwardDiff.derivative(f, x)
xs = range(-3, 3, length=100)
ax = Axis(fig[1, 1],
          title = L"y = -x^2+1", xlabel = L"x", ylabel = L"y")
lines!(ax, xs, f.(xs), color = :blue, label = "function")
scatter!(ax, [x_opt], [f(x_opt)], color = :red, label = "optimal")
scatter!(ax, [x_init], [f(x_init)], color = :purple, markersize = 10, label = "initial")
axislegend(ax; position = :rb)

Label(fig[0, 1:3],
      L"\text{Maximize }f(x)=%$(f_string)\text{ with }\eta = %$η, \text{ }x_{\text{init}} = %$x_init",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)


#1変数関数の最適化
function gradient_method_1dim(f, x_init, η, maxiter)
    # 最適化過程のパラメータを格納する配列
    x_seq = Array{typeof(x_init), 1}(undef, maxiter)
    #  勾配
    df(x) = ForwardDiff.derivative(f, x)
    # 初期値
    x_seq[1] = x_init
    # メインの最適化ループ
    for i in 2:maxiter
        x_seq[i] = x_seq[i-1] + η * df(x_seq[i-1])
    end
    x_seq
end


x_seq = gradient_method_1dim(f, x_init, η, maxiter)
f_seq = f.(x_seq)

ax2 = Axis(fig[1, 2:3],
           title = L"\max\text{ }f", xlabel = "iteration", ylabel = L"f")
lines!(ax2, 1:maxiter, f_seq, color = :green,
       # label = "optimization path",
       )
scatter!(ax2, 1:maxiter, f_seq, color = :green, marker = :diamond, label = L"f(x)\text{ sequence}")
hlines!(ax2, [y_opt], color = :red, label = L"\max\text{ }f")
axislegend(ax2; position = :rb)

ax3 = Axis(fig[2, 1],
           title = L"\max f", xlabel = L"x", ylabel = L"f(x)")
lines!(ax3, xs, f.(xs), color = :blue, label = "function")
scatter!(ax3, x_seq, f_seq, color = :green, marker = :diamond, label = "optimization path")
scatter!(ax3, [x_opt], [f(x_opt)], color = :red, label = "optimal")
axislegend(ax3; position = :rb)

ax4 = Axis(fig[2, 2:3],
           title = L"x", xlabel = "iteration", ylabel = L"x")
lines!(ax4, 1:maxiter, x_seq, color = :green, label = L"x\text{ sequence}")
scatter!(ax4, 1:maxiter, x_seq, marker = :diamond, color = :green,
         #label = "function"
         )
hlines!(ax4, [x_opt], color = :red, linestyle = :dash, label = "optimal x")
axislegend(ax4; position = :rb)

fig |> display
safesave(plotsdir(program_name * "_η=$(η)" * ".pdf"), fig)
