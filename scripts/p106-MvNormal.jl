using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p106-MvNormal"
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
    size = (1200, 800),
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
# multiple MvNormals
################################################################

for i in 1:length(μs)
    d = MvNormal(μs[i], Σs[i])
    row, col = n2ij(i, length(μs))

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

    N = 1000
    X = rand(d, N)

    ax = Axis(fig[row + n2grid(length(μs))[1], col],
              xlabel = L"x",
              ylabel = L"y",
              title = "sample (N=$N)",
              limits = limits,
              )

    scatter!(ax, X,
             )

end


fig |> display
safesave(plotsdir(program_name * ".pdf"), fig)
