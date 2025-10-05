using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p83--discrete-distributions"
using CairoMakie
using LaTeXStrings
using Distributions

################################################################
# plot init
################################################################
fig = Figure(size = (800, 600))
Label(fig[0, 1:4],
      "Distributions",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)
################################################################
# Bernoulli
################################################################
p = 0.3
d = Bernoulli(p)
d_pdf = [pdf(d, x) for x in 0.0:0.1:1.0]

ax = Axis(fig[1,1],
          title = "Bernoulli($p)",
          xlabel = L"x",
          ylabel = "probability"
          )
barplot!(ax, [0, 1], [d_pdf[1], d_pdf[11]])

################################################################
# Binomial
################################################################
n_samples = 10^5
m = 20
p = 0.3
d = Binomial(20, 0.3)
d_pdf = pdf(d) # d_pdf[1] は「0回」の確率なので注意
xs = rand(d, n_samples)

ax = Axis(fig[1,2:4],
          title = "Binomial($m, $p), n_samples = $(n_samples), `normalization=:probability`",
          xlabel = L"x",
          ylabel = "probability"
          )
hist!(ax, xs,
      bins=50,
      # normalization=:pdf,
      normalization=:probability,
      )
lines!(ax,
       0:20, # Julia is 1-index
       d_pdf,
      color = :black,
)


fig |> display
