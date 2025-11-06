# 2025年11月5日 Claude Opus 4.1
# バグありでこのままでは動かず
# Refactored version with metaprogramming and reduced duplication
disp, save_fig, benchmark_flag = true, true, false

function Base.joinpath(s::AbstractString, n::Nothing)
    joinpath(s)
end

using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p195--linear-regression-MCMC"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf
using IceCream
using Random
using ForwardDiff
using LinearAlgebra
using BenchmarkTools: @btime, @benchmark
import ProgressMeter: Progress, next!, @timed, @showprogress

include(srcdir("utility_functions.jl"))

Random.seed!(1234)

# ================================================================
# Data and Model Definition
# ================================================================
# Observed data
X_obs = [-2, 1, 5]
Y_obs = [-2.2, -1.0, 1.5]

# Model parameters
σ = 1.0  # noise size on y
μ1, μ2 = 0.0, 0.0  # prior means
σ1, σ2 = 10.0, 10.0  # prior stds

# Joint log-probability
log_joint(w, X, Y, σ, μ1, σ1, μ2, σ2) =
    sum(logpdf.(Normal.(w[1] * X .+ w[2], σ), Y)) +
    logpdf(Normal(μ1, σ1), w[1]) +
    logpdf(Normal(μ2, σ2), w[2])

params = (X_obs, Y_obs, σ, μ1, σ1, μ2, σ2)
ulp(w) = log_joint(w, params...)  # unnormalized log-posterior

# ================================================================
# Structured Data for MCMC Results
# ================================================================
struct MCMCResults
    name::String
    samples::Matrix{Float64}
    samples_bi::Matrix{Float64}  # burnin removed
    acceptance_rate::Float64
    stats::NamedTuple
end

function compute_stats(samples)
    return (
        means = mean(samples, dims=2),
        medians = median(samples, dims=2),
        stds = std(samples, dims=2)
    )
end

function MCMCResults(name, samples, num_accepted, max_iter, burnin)
    samples_bi = samples[:, burnin+1:end]
    return MCMCResults(
        name,
        samples,
        samples_bi,
        num_accepted / max_iter,
        (
            full = compute_stats(samples),
            burnin = compute_stats(samples_bi)
        )
    )
end

# ================================================================
# MCMC Sampling
# ================================================================
w_init = randn(2)
max_iter = 2000
burnin = 500

# Run both MCMC methods
samples_GMH, num_accepted_GMH = inference_wrapper_GMH(
    log_joint, params, w_init, max_iter=max_iter, σ=1.0)
samples_HMC, num_accepted_HMC = inference_wrapper_HMC(
    log_joint, params, w_init, max_iter=max_iter, L=10, ε=1e-1)

# Store results in structs
results = Dict(
    :GMH => MCMCResults("GMH", samples_GMH, num_accepted_GMH, max_iter, burnin),
    :HMC => MCMCResults("HMC", samples_HMC, num_accepted_HMC, max_iter, burnin)
)

# ================================================================
# Plotting Configuration
# ================================================================
const PLOT_CONFIG = (
    w1_limits = ((-0.5, 1.5), (0, 3.5)),  # (ylim, xlim) for rotated hist
    w2_limits = ((-3.0, 1.5), (0, 1.5)),
    mean_color = :red,
    median_color = :purple,
    hist_bins = 50,
    hist_xticks = 3,
    w2_yticks = 5
)

# ================================================================
# Plot Generation with Metaprogramming
# ================================================================
fig = Figure(size=(1500, 500), figure_padding=30)

Label(fig[0, 1:7],
    "logistic regression (red: mean, purple: median, red dash: mean±std)",
    fontsize=20, font=:bold)

# Data plot
ax11 = Axis(fig[1, 1], title="data (N=$(length(X_obs)))", xlabel=L"x", ylabel=L"y")
scatter!(ax11, X_obs, Y_obs, color=:green)

# Function to create trace plot
function create_trace_plot!(fig, row, col, param_idx, result::MCMCResults, limits)
    param_name = param_idx == 1 ? L"w_1" : L"w_2"
    title = if param_idx == 1
        L"%$(param_name)\text{ sequence (%$(result.name)), acceptance rate = }%$(round(result.acceptance_rate, sigdigits=2))"
    else
        L"%$(param_name)\text{ sequence (%$(result.name))}"
    end
    
    ax = Axis(fig[row, col],
        title=title,
        xlabel="iteration",
        ylabel=param_name,
        limits=(nothing, limits[1]),
        yticks=(param_idx == 2 ? LinearTicks(PLOT_CONFIG.w2_yticks) : :automatic)
    )
    lines!(ax, result.samples[param_idx, :])
    return ax
end

# Function to create histogram
function create_histogram!(fig, row, col, param_idx, samples, stats, limits, title_text)
    ax = Axis(fig[row, col],
        title=title_text,
        xlabel="prob dens",
        limits=(limits[2], limits[1]),  # swap for rotated histogram
        xticks=LinearTicks(PLOT_CONFIG.hist_xticks),
        yticks=(param_idx == 2 ? LinearTicks(PLOT_CONFIG.w2_yticks) : :automatic)
    )
    
    hist!(ax, samples[param_idx, :], direction=:x, normalization=:pdf, bins=PLOT_CONFIG.hist_bins)
    
    # Add statistical lines
    m, med, s = stats.means[param_idx], stats.medians[param_idx], stats.stds[param_idx]
    hlines!(ax, [m], color=PLOT_CONFIG.mean_color)
    hlines!(ax, [med], color=PLOT_CONFIG.median_color)
    hlines!(ax, [m + s, m - s], color=PLOT_CONFIG.mean_color, linestyle=:dash)
    
    return ax
end

# Generate plots using metaprogramming
let col_offset = 1
    for (method_key, result) in pairs(results)
        base_col = method_key == :GMH ? 2 : 5
        
        # Trace plots for w1 and w2
        for (param_idx, limits) in enumerate([PLOT_CONFIG.w1_limits, PLOT_CONFIG.w2_limits])
            create_trace_plot!(fig, param_idx, base_col, param_idx, result, limits)
        end
        
        # Histograms (full samples)
        for (param_idx, limits) in enumerate([PLOT_CONFIG.w1_limits, PLOT_CONFIG.w2_limits])
            create_histogram!(fig, param_idx, base_col + 1, param_idx, 
                result.samples, result.stats.full, limits, "hist")
        end
        
        # Histograms (burnin removed)
        for (param_idx, limits) in enumerate([PLOT_CONFIG.w1_limits, PLOT_CONFIG.w2_limits])
            title = param_idx == 1 ? "hist with burnin=$burnin removed" : "hist with burnin=$burnin removed"
            create_histogram!(fig, param_idx, base_col + 2, param_idx,
                result.samples_bi, result.stats.burnin, limits, title)
        end
    end
end

# ================================================================
# Layout Adjustment
# ================================================================
# Column widths
for (col, width) in enumerate([0.1, 0.25, 0.1, 0.1, 0.25, 0.1, 0.1])
    colsize!(fig.layout, col, Relative(width))
end

# Row heights
rowsize!(fig.layout, 1, Relative(0.5))
rowsize!(fig.layout, 2, Relative(0.5))

# ================================================================
# Display and Save
# ================================================================
disp && fig |> display
save_fig && safesave(plotsdir(program_name * ".pdf"), fig)

# ================================================================
# Optional: Easy access to results for debugging
# ================================================================
if @isdefined(benchmark_flag) && benchmark_flag
    println("\n=== MCMC Results Summary ===")
    for (method, res) in results
        println("\n$method:")
        println("  Acceptance rate: $(round(100*res.acceptance_rate, digits=1))%")
        println("  w1: mean=$(round(res.stats.burnin.means[1], digits=3)), " *
                "std=$(round(res.stats.burnin.stds[1], digits=3))")
        println("  w2: mean=$(round(res.stats.burnin.means[2], digits=3)), " *
                "std=$(round(res.stats.burnin.stds[2], digits=3))")
    end
end
