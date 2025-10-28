using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p123-lognormal-multiple"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf

include(srcdir("utility_functions.jl"))

################################################################
# NegativeBinomial
################################################################
μs = [-1.0, 0.0, 1.0]
σs = [0.2, 1.0, 1.5]
# ds = Normal.(μs, σs)
n_μs = length(μs)
n_σs = length(σs)
n_params = n_σs * n_μs
n_row, n_col = n2grid(n_params)

################################################################
# plot init
################################################################
fig = Figure(
    size = (1200, 800),
    figure_padding = 30
)
str1 = "\\text{LogNormal}(\\mu, \\sigma)\\text{ distributions. The red vline is the mean.}"
str2 = "\\text{mean, median, mode, and var are, respectively, }"
str3 = "e^{\\mu+\\sigma^2/2}, e^\\mu, e^{\\mu-\\sigma^2}, \\text{ and, } e^{2^\\mu+\\sigma^2}(e^{\\sigma^2}-1)."
Label(fig[0, 1:n_col],
      # L"\text{Gamma}(\alpha, \theta)\text{ distributions; }\mu = \alpha\theta\text{, }\sigma^2 = \alpha\theta^2",
      LaTeXString(str1 * str2 * str3),
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      # "aaa",
      fontsize = 16, font = :bold)

################################################################
# multiple Normals
################################################################
for (i, (μ, σ)) in enumerate(Iterators.product(μs, σs))
    row, col = n2ij(i, n_params)
    d = LogNormal(μ, σ)
    m = mean(d)
    v = var(d)
    σ = sqrt(v)
    m_str = @sprintf("%.2f", m)
    v_str = @sprintf("%.2f", v)
    σ_str = @sprintf("%.2f", σ)
    # [μ-σ, μ+σ] の面積
    sm, sp = cdf(d, [m-σ, m+σ])
    oneσp = sp-sm
    oneσpf = @sprintf("%.2f", oneσp)

    ax = Axis(fig[row, col],
              title = "Normal(μ=$μ, σ=$(round(σ, digits=2)))",
              xlabel = L"x",
              ylabel = "probability",
              # aspect = DataAspect(),
              aspect = 2,
              )
    # barplot!(ax, xs, pdf.(d, xs)) # xs もないし

    xs = range(0, 5m, length=100)
    # xs = -4:0.01:4
    lines!(ax, xs, pdf.(d, xs))
    text!(ax, 0.7, 0.9,  # (0.5, 0.9) = 中央上部
          text = "mean = $(m_str), var = $(v_str)\n std = $(σ_str), P(μ±σ) = $oneσpf",
          align = (:center, :top),
          space = :relative,  # これが重要
          fontsize = 14)
    lw = 2
    vlines!(ax, [m], label="mean", color = :red, linewidth = lw)
    # vlines!(ax, [m-σ, m+σ], label="mean±σ", color = :orange, linestyle = :dash, linewidth = lw)
end

fig |> display
safesave(plotsdir(program_name * ".pdf"), fig)
