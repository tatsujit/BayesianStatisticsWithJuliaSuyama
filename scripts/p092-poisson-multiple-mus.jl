using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p092-poisson-multiple-mus"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf

include(srcdir("utility_functions.jl"))
################################################################
# poisson
################################################################
μs = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0]#, 32.0, 64.0, 128.0]
n_μs = length(μs)
n_row, n_col = n2grid(n_μs)
################################################################
# plot init
################################################################
fig = Figure(
    size = (800, 600),
    figure_padding = 30
)
Label(fig[0, 1:n_col],
      "Poisson distributions",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)
################################################################
# multile Poissons
################################################################
xs = 0:25 # plot range
for (i, μ) in enumerate(μs)
    row, col = n2ij(i, n_μs)
    d = Poisson(μ)
    σ = sqrt(μ)

    # [μ-σ, μ+σ] の面積
    sm, sp = cdf(d, [μ-σ, μ+σ])
    oneσp = sp-sm
    oneσpf = @sprintf("%.2f", oneσp)

    ax = Axis(fig[row, col],
              title = "Poisson($μ), P(μ-σ)-P(μ+σ)=$(oneσpf)",
              xlabel = L"x",
              ylabel = "probability",
              # aspect = DataAspect(),
              aspect = 2,
              )
    barplot!(ax, xs, pdf.(d, xs))
end

################################################################
# multipe Poissons with P(μ-σ)-P(μ+σ), compared to that of Normal
################################################################

# oneσp = sp-sm の計算
mus = 0.0:0.1:100.0
oneσps = Float64[]
# Threads.@threads
for mu in mus
    d = Poisson(mu)
    σ = sqrt(mu)
    # [μ-σ, μ+σ] の面積
    sm, sp = cdf(d, [mu-σ, mu+σ])
    oneσp = sp-sm
    # oneσpf = @sprintf("%.2f", oneσp)
    push!(oneσps, oneσp)
end
oneσps

# for normal distribution
d = Normal(0, 1)
sm, sp = cdf(d, [-1, +1])
oneσp = sp - sm

ax2 = Axis(fig[n_row+1, 1:n_col],
           xlabel = "μ", ylabel = "prob.", title = "Poisson vs Normal, μ±σ")
lines!(ax2, mus, oneσps)
hlines!(ax2, [oneσp],
        label="Normal μ±σ",
        color = :red,
        linestyle = :dashed,
        )

fig |> display
# safesave(plotsdir(program_name * ".pdf"), fig)
