using DrWatson
@quickactivate "BayesianStatisticsWithJuliaSuyama"
program_name = "p101-normal-cdf-and-areas"
using CairoMakie
using LaTeXStrings
using Distributions
using Printf
using IceCream

include(srcdir("utility_functions.jl"))

d = Normal(0, 1)
X = rand(d, 10000)

mean(0.0 .< X .< 1.0)
