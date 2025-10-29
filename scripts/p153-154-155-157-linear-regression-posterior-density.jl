# disp, save_fig = true, true
disp, save_fig = true, false
# disp, save_fig = false, true
# disp, save_fig = false, false
using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p153-linear-regression-posterior-density"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf
using IceCream

include(srcdir("utility_functions.jl"))

################################################################
# data
################################################################
X_obs = [-2, 1, 5]
Y_obs = [-2.2, -1.0, 1.5]


################################################################
# plot init
################################################################
fig = Figure(
    size = (1600, 800),
    figure_padding = 30
)
Label(fig[0, 1:4],
      "linear regression", #"\mu=%$μ\text{, }\sigma=%$σ",
      # L"η = %$η,\text{ }x_{\text{init}} = %$x_init",
      fontsize = 16, font = :bold)

x_lower, x_upper = -3, 3
y_lower, y_upper = x_lower, x_upper

xs = range(x_lower, x_upper, length=100)
ys = range(y_lower, y_upper, length=100)
limits = ((x_lower, x_upper), (y_lower, y_upper))

################################################################
# plot data
################################################################
ax1 = Axis(fig[1, 1], title = "data (N=$(length(X_obs)))")
scatter!(ax1, X_obs, Y_obs)


################################################################
# calculate marginal likelihood
################################################################

# 同時分布
p_joint(X, Y, w1, w2) = prod(pdf.(Normal.(w1 .* X .+ w2, σ), Y)) *
    pdf(Normal(μ1, σ1), w1) *
    pdf(Normal(μ2, σ2), w2)

# 数値積分
function approx_integration_2D(w_range, p)
    Δ = w_range[2] - w_range[1]
    (X, Y) -> sum([p(X, Y, w1, w2) * Δ^2 for w1 in w_range, w2 in w_range])
end
# w の積分範囲
w_range = range(-3, 3, length=100)
# 数値積分の実行
p_marginal = approx_integration_2D(w_range, p_joint)
p_marginal(X_obs, Y_obs) # 6.924264340150274e-5

# 事後分布の計算
w_posterior = [p_joint(X_obs, Y_obs, w1, w2) for w1 in w_range, w2 in w_range] ./
    p_marginal(X_obs, Y_obs)

################################################################
# plot
################################################################
ax12 = Axis(fig[1, 2],
           xlabel = L"w_1",
           ylabel = L"w_2",
           title = "posterior density (contour)",
           limits = limits,
           )

contour!(ax12, w_range, w_range, w_posterior,
         levels = 10,
         linewidth = 1,
         labels = true,
         )

ax22 = Axis(fig[2, 2],
           xlabel = L"w_1",
           ylabel = L"w_2",
           title = "posterior density (contourf)",
           limits = limits,
           )

hm = contourf!(ax22, w_range, w_range, w_posterior,
               levels = 10,
               )

Colorbar(fig[2,3], hm)

################################################################
# predictive distribution
################################################################
function approx_predictive(w_posterior, w_range, p)
    Δ = w_range[2] - w_range[1]
    return (x, y) -> sum([p(x, y, w1, w2) * w_posterior[i, j] * Δ^2
                          for (i, w1) in enumerate(w_range),
                              (j, w2) in enumerate(w_range)])
end
p_likelihood(xp, yp, w1, w2) = pdf(Normal(w1*xp + w2, σ), yp)
p_predictive = approx_predictive(w_posterior, w_range, p_likelihood)

xp = 4.0

################################################################
# plot predictive distribution
################################################################
ax21 = Axis(fig[2, 1],
          xlabel = L"y_p",
          ylabel = L"\text{density}",
          title = L"\text{predictive distribution }p(y_p |x_p = 4.0, X=X_\text{obs}, Y=Y_\text{obs})")
ys = range(-5, 5, length=100)
lines!(ax21, ys, p_predictive.(xp, ys))

################################################################
# the global prediction distribution
################################################################
xs = range(-10, 10, length=100)
ys = range(-5, 5, length=100)
limits = (extrema(xs), extrema(ys))
density_y = p_predictive.(xs, ys')

################################################################
# plot
################################################################
ax14 = Axis(fig[1, 4],
           xlabel = L"x",
           ylabel = L"y",
           title = "predictive distribution (contour)",
           limits = limits,
           )

contour!(ax14,
         xs, ys, density_y,
         levels = 10,
         linewidth = 1,
         labels = true,
         )

scatter!(ax14, X_obs, Y_obs, color = :red, label = "data")
axislegend(ax3, position = :lt)

ax24 = Axis(fig[2, 4],
           xlabel = L"x",
           ylabel = L"y",
           title = "predictive distribution (contourf)",
           limits = limits,
           )

hm = contourf!(ax24,
               xs, ys, density_y,
               levels = 10,
               )

scatter!(ax24, X_obs, Y_obs, color = :red, label = "data")
axislegend(ax24, position = :lt)

Colorbar(fig[2,5], hm)

################################################################
# exact solution
################################################################

# 切片用の擬似データ
X_extended = hcat(X_obs, ones(size(X_obs)))'

# parameters for prior
Σ = [σ1^2 0;
     0 σ2^2]
μ = [μ1, μ2]

# parameters for posterior
Σ_hat = inv(σ^(-2) * X_extended * X_extended' + inv(Σ))
μ_hat = Σ_hat * (σ^(-2) .+ X_extended * Y_obs + inv(Σ) * μ)

# parameters for predictive dist.
μp(xp) = (μ_hat' * xp)[1]
σp(xp) = sqrt(σ^2 + (xp' * Σ_hat * xp)[1])

# prob density func for predictive
p_predictive_exact(xp, yp) = pdf(Normal(μp(xp), σp(xp)), yp)

# 描画範囲
xs = range(-10, 10, length=100)
ys = range(-5, 5, length=100)
density_y = [p_predictive_exact([x, 1.0], y) for x in xs, y in ys]

################################################################
# plot
################################################################
ax16 = Axis(fig[1, 6],
           xlabel = L"x",
           ylabel = L"y",
           title = "exact pred. dist. (contour)",
           limits = limits,
           )

contour!(ax16,
         xs, ys, density_y,
         levels = 10,
         linewidth = 1,
         labels = true,
         )

scatter!(ax16, X_obs, Y_obs, color = :red, label = "data")
axislegend(ax16, position = :lt)

ax26 = Axis(fig[2, 6],
           xlabel = L"x",
           ylabel = L"y",
           title = "exact pred. dist. (contourf)",
           limits = limits,
           )

hm = contourf!(ax26,
               xs, ys, density_y,
               levels = 10,
               )

scatter!(ax26, X_obs, Y_obs, color = :red, label = "data")
axislegend(ax26, position = :lt)

Colorbar(fig[2,5], hm)



disp && fig |> display
save_fig && safesave(plotsdir(program_name * ".pdf"), fig)
