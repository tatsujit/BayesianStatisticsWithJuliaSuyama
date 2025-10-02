using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
using CairoMakie
using ForwardDiff

program_name = "p039-automatic-differentiation"

fig = Figure(resolution = (900, 300))

################################################################"
# f(x) = -x^2 + 1
################################################################"
f(x) = -(x + 1) * (x - 1)
df(x) = ForwardDiff.derivative(f, x)
xs1 = range(-1, 1, length=100)
ys1 = f.(xs1)
dys1 = df.(xs1)

ax1 = Axis(fig[1, 1],
          title = "-x^2+1", xlabel = "x", ylabel = "y")
lines!(ax1, xs1, ys1, color = :blue, label = "f(x)")
lines!(ax1, xs1, dys1, color = :red, label = "df(x)")


################################################################"
# sin and sin' = cos
################################################################"
dsin(x) = ForwardDiff.derivative(sin, x)
xs2 = range(0, 6π, length=100)
ys2 = sin.(xs2)
dys2 = dsin.(xs2)

ax2 = Axis(fig[1, 2],
          title = "sin(x)", xlabel = "x", ylabel = "y")
lines!(ax2, xs2, ys2, color = :blue, label = "f(x)")
lines!(ax2, xs2, dys2, color = :red, label = "df(x)")


################################################################"
# sigmoid and sigmoid'
################################################################"
sigmoid(x) = 1 / (1 + exp(-x))
dsigmoid(x) = ForwardDiff.derivative(sigmoid, x)
xs3 = range(-5, 5, length=100)
ys3 = sigmoid.(xs3)
dys3 = dsigmoid.(xs3)

ax3 = Axis(fig[1, 3],
          title = "sigmoid(x)", xlabel = "x", ylabel = "y")
lines!(ax3, xs3, ys3, color = :blue, label = "f(x)")
lines!(ax3, xs3, dys3, color = :red, label = "df(x)")



fig |> display
safesave(plotsdir(program_name * ".pdf"), fig)
