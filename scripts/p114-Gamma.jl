using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p114-Gamma"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf

################################################################
# Gamma
################################################################
xs = range(0, 150, length=1000)
αs = [0.5, 1.0, 2.0, 5.0]
θs = [0.5, 1.0, 1.5, 10.0]
params = vec(collect(Iterators.product(αs, θs)))

μs = [α * θ for (α, θ) in params]
σ²s = [α * θ^2 for (α, θ) in params]

n_params = length(params)
n_row, n_col = n2grid(n_params)

################################################################
# plot init
################################################################
fig = Figure(
    size = (1200, 800),
    figure_padding = 50
)
str1 = "\\text{Gamma}(\\alpha, \\theta)\\text{ distributions; }"
str2 = "\\alpha, \\theta \\text{ are shape and scale params, }"
str3 = "\\text{(or rate with }\\lambda = \\frac{1}{\\theta}\\text{), }"
str4 = "\\mu = \\alpha\\theta\\text{, }\\sigma^2 = \\alpha\\theta^2,"
Label(fig[0, 1:n_col],
      # L"\text{Gamma}(\alpha, \theta)\text{ distributions; }\mu = \alpha\theta\text{, }\sigma^2 = \alpha\theta^2",
      LaTeXString(str1 * str2 * str3 * str4),
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)
################################################################
# multiple Poissons
################################################################
for (i, (α, θ)) in enumerate(params)
    row, col = n2ij(i, n_params)
    d = Gamma(α, θ)
    μ = mean(d)
    σ² = var(d)
    σ = sqrt(σ²)
    μ_str = @sprintf("%.2f", μ)
    σ²_str = @sprintf("%.2f", σ²)
    σ_str = @sprintf("%.2f", σ)
    # [μ-σ, μ+σ] の面積
    sm, sp = cdf(d, [μ-σ, μ+σ])
    oneσp = sp-sm
    oneσpf = @sprintf("%.2f", oneσp)

    ax = Axis(fig[row, col],
              title = "Gamma(α=$α, θ=$θ), P(μ±σ)=$(oneσpf)",
              xlabel = L"x",
              ylabel = "prob. density",
              # aspect = DataAspect(),
              aspect = 2,
              limits = ((0, μ+3σ), nothing)
              )
    lw = 2
    vlines!(ax, [μ], label="mean", color = :red, linewidth = lw)
    vlines!(ax, [μ-σ, μ+σ], label="mean±σ", color = :orange, linestyle = :dash, linewidth = lw)
    lines!(ax, xs, pdf.(d, xs))
    text!(ax, 0.5, 0.9,  # (0.5, 0.9) = 中央上部
          text = "μ = $(μ_str), σ² = $(σ²_str)\n σ = $(σ_str)",
          align = (:center, :top),
          space = :relative,  # これが重要
          fontsize = 14)
end

fig |> display
# safesave(plotsdir(program_name * ".pdf"), fig)
