using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p109-MvNormal-marginals"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf
using IceCream

include(srcdir("utility_functions.jl"))

################################################################
# Multivariate Normal
################################################################
μs = [[0.0, 0.0],
      [0.0, 0.0],
      [0.0, 0.0],
      [0.0, 0.0],
      ]
Σs = [
    [1.0 0.0;
     0.0 1.0],
    [1.0 0.5;
     0.5 1.0],
    [1.0 -0.5;
     -0.5 1.0],
    [1.0 0.9;
     0.9 1.0],

    # LoadError: PosDefException: matrix is not positive definite; Factorization failed.
    # [1.0 1.0;
    #  1.0 1.0],
    # LoadError: PosDefException: matrix is not Hermitian; Factorization failed.
    # [1.0 0.5;
    #  -0.5 1.0],
]
@assert length(μs) == length(Σs)
n_row, n_col = n2grid(length(μs))

################################################################
# plot init
################################################################
fig = Figure(
    size = (800, 800),
    figure_padding = 30
)
Label(fig[0, 1:n_col],
      "Multivariate Normal distributions", #"\mu=%$μ\text{, }\sigma=%$σ",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)

x_lower, x_upper = -5, 5
y_lower, y_upper = x_lower, x_upper

xs = range(x_lower, x_upper, length=100)
ys = range(y_lower, y_upper, length=100)
limits = ((x_lower, x_upper), (y_lower, y_upper))

################################################################
# MvNormal and the marginals
################################################################

i = 2
d = MvNormal(μs[i], Σs[i])

row, col = 1, 1
ax = Axis(fig[row, col],
          xlabel = L"x",
          ylabel = L"y",
          title = "prob. density",
          limits = limits,
          )

contour!(ax, xs, ys, (x, y) -> pdf(d, [x, y]),
         levels = 5,
         linewidth = 1,
         labels = true,
         )
text!(ax, 0.5, 0.9,  # (0.5, 0.9) = 中央上部
      text = "μ = $(μs[i]), Σ = $(Σs[i])",
      align = (:center, :top),
      space = :relative,  # これが重要
      fontsize = 14)

row, col = 1, 2
ax = Axis(fig[row, col],
          xlabel = L"\text{density}",
          ylabel = L"y",
          title = "prob. density",
          limits = ((0, 0.5), (y_lower, y_upper)),
          )
y_range = range(y_lower, y_upper, length=100)
px_marginal(x) = approx_integration(y_range, y -> pdf(d, [x, y]))[1]
densities = px_marginal.(y_range)
lines!(ax, densities, y_range)

row, col = 2, 1
ax = Axis(fig[row, col],
          xlabel = L"x",
          ylabel = L"\text{density}",
          title = "prob. density",
          limits = ((x_lower, x_upper), (0, 0.5)),
          )
x_range = range(x_lower, x_upper, length=100)
py_marginal(y) = approx_integration(x_range, x -> pdf(d, [x, y]))[1]
densities = py_marginal.(x_range)
lines!(ax, x_range, densities)


fig |> display
safesave(plotsdir(program_name * ".pdf"), fig)
