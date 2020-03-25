using FileIO
using Mmap
using StatsBase
using WebIO # for show_pdf
using WeightedOnlineStats
using MultivariateStats
using DoubleFloats
using ESDLPlots
using Base.Iterators
using DataFrames
using BenchmarkTools
using Distributions
using Distributed
using ESDL
using OnlineStats
using RCall
using Plots
gr()
using Dates
using JLD2
using LinearAlgebra
using ProgressMeter
using Statistics


# data_cube_dir = "/scratch/DataCube/v1.0.2/esdc-8d-0.25deg-1x720x1440-1.0.2_1/"
# data_cube_dir = "/scratch/DataCube/v1.0.2/esdc-8d-0.25deg-46x90x90-1.0.2_1/"

# data_cube_dir = "/scratch/DataCube/v1.0.2/esdc-8d-0.25deg-1x720x1440-1.0.2/"
# data_cube_dir = "/scratch/DataCube/v1.0.2/esdc-8d-0.25deg-46x90x90-1.0.2/"
# data_cube_dir = "/scratch/DataCube/v1.0.0/low-res/"
#cubes_base_dir =  "/Net/Groups/BGI/scratch/gkraemer/global_pca_j1"
data_cube_dir = "/scratch/DataCube/v1.0.0_fapar_tip/low-res"
cubes_base_dir =  "/scratch/gkraemer/global_pca_j1"
ESDLdir("/scratch/gkraemer/")

ispath(cubes_base_dir) || mkdir(cubes_base_dir)


vars = [
    # "aerosol_optical_thickness_1610", "aerosol_optical_thickness_550", "aerosol_optical_thickness_555","aerosol_optical_thickness_659", "aerosol_optical_thickness_865","air_temperature_2m",
    #"bare_soil_evaporation",
    "black_sky_albedo",
    #"burnt_area","c_emissions",    # many zeroes
    # "country_mask",
    "evaporation","evaporative_stress",
    "fapar_tip",
    # "fractional_snow_cover",
    "gross_primary_productivity",
    #"interception_loss",           # many zeroes
    # "land_surface_temperature",
    "latent_energy",
    # "leaf_area_index",
    "net_ecosystem_exchange",
    #"open_water_evaporation",      # many zeroes
    #"ozone", "potential_evaporation", "precipitation",
    "root_moisture","sensible_heat",
    #"snow_sublimation",            # many zeroes
    #"snow_water_equivalent", "soil_moisture", "srex_mask",
    "surface_moisture", "terrestrial_ecosystem_respiration",
    #"transpiration",
    # "water_mask", "water_vapour",
    "white_sky_albedo"
];
length(vars)

fig_path = joinpath(cubes_base_dir, "fig")
if fig_path |> !ispath
    mkpath(fig_path)
end


R"""
library(grid)
library(ggplot2)
library(ggthemes)
library(raster)
library(animation)
library(gridExtra)
library(RColorBrewer)
library(maptools)
library(maps)
library(strucchange)
library(energy)
library(ncdf4)
library(tidyverse)
"""



ex_axis = CategoricalAxis(:Extrema, ["min", "max"])

missing_to_nan!(x::Array{Union{Missing, T}}) where T = map!(y -> y === missing ? T(NaN) : y, x, x)
missing_to_nan(x::Array{Union{Missing, T}}) where T = map(y -> y === missing ? T(NaN) : y, x)
missing_to_nan(x::Array{T}) where T = x
missing_to_nan!(x::Array{T}) where T = x
missing_to_zero!(x::Array{Union{Missing, T}}) where T = map!(y -> y === missing ? T(0) : y, x, x)
missing_to_zero(x::Array{Union{Missing, T}}) where T = map(y -> y === missing ? T(0) : y, x)
Base.filter(f::Function, x::NTuple{T}) where T = x[findall(f.(x))]

function drop_size_one_dims(x)
    dims2drop = (findall(y -> y == 1, size(x))...,)
    dropdims(x, dims = dims2drop)
end

function rm_tails(x, q = [0.1, 0.9])
    @assert length(q) == 2
    @assert q[1] < q[2]
    @assert q[1] >= 0
    @assert q[2] <= 1
    qq = quantile(vec(x)[.!isnan.(vec(x))], q)
    xx = copy(x)
    xx[xx .< qq[1]] = qq[1]
    xx[xx .> qq[2]] = qq[2]
    xx
end

function join_last_dims(x, n)
    reshape(x, size(x)[1:end - n - 1]..., prod(size(x)[end - 1:end]))
end
function keep_first_dims(x, n)
    reshape(x, size(x)[1:n]..., prod(size(x)[n + 1:end]))
end

#TODO: remove this cell after updating ESDL!
macro loadOrGenerate2(x...)
  code = x[end]
  x = x[1:end-1]
  x2 = map(x) do i
        if isa(i,Symbol)
            (i, string(i))
        elseif i.head==:call && i.args[1]==:(=>)
            path = eval(i.args[3])::String
            (i.args[2], path)
        else
            error("Wrong Argument type")
        end
  end
  xnames = map(i->i[2],x2)

  loadEx = map(x2) do i
    :($(i[1]) = loadCube($(i[2])))
  end
  loadEx = Expr(:block,loadEx...)

  saveEx = map(x2) do i
    :(saveCube($(i[1]),$(i[2])))
  end
  saveEx=Expr(:block,saveEx...)

  rmEx=map(x2) do i
    :(ispath($(i[2])) && rm($(i[2])))
  end
  rmEx=Expr(:block,rmEx...)
  esc(quote
    if !ESDL.recalculate() && all(i->isdir(joinpath(ESDLdir(),i)),$xnames)
      $loadEx
    else
      $rmEx
      $code
      $saveEx
    end
  end)
end

"""
make the correlation pc1 ~ gpp > 0 and pc2 ~ evaporative_stress < 0
"""
function fix_pca_directions(proj, varnames)
    proj = copy(proj)
    gpp_idx = findfirst(==("gross_primary_productivity"), varnames)
    evs_idx = findfirst(==("evaporative_stress"), varnames)
    alb_idx = findfirst(==("black_sky_albedo"), varnames)
    if proj[gpp_idx, 1] < 0
        proj[:, 1] .= proj[:, 1] .* -1
    end
    if proj[evs_idx, 2] > 0
        proj[:, 2] .= proj[:, 2] .* -1
    end
    if proj[alb_idx, 3] < 0
        proj[:, 3] .= proj[:, 3] .* -1
    end
    proj
end

function Base.convert(::Type{WeightedCovMatrix{T}}, o::WeightedCovMatrix) where T
    WeightedCovMatrix{T}(
        convert(Matrix{T}, o.C),
        convert(Matrix{T}, o.A),
        convert(Vector{T}, o.b),
        convert(T,         o.W),
        convert(T,         o.W2),
        o.n
    )
end

StatsBase.transform(t::StatsBase.AbstractDataTransform, x) = StatsBase.transform!(x, t, x)
StatsBase.transform(t::StatsBase.AbstractDataTransform, x) = StatsBase.transform!(similar(x), t, x)
function StatsBase.transform!(y::AbstractVecOrMat, t::ZScoreTransform, x::AbstractVecOrMat)
    d = t.dim
    size(x,1) == size(y,1) == d || throw(DimensionMismatch("Inconsistent dimensions."))
    n = size(y,2)
    size(x,2) == n || throw(DimensionMismatch("Inconsistent dimensions."))

    m = t.mean
    s = t.scale

    if isempty(m)
        if isempty(s)
            if x !== y
                copyto!(y, x)
            end
        else
            broadcast!(/, y, x, s)
        end
    else
        if isempty(s)
            broadcast!(-, y, x, m)
        else
            broadcast!((x, m, s) -> (x - m) / s, y, x, m, s)
        end
    end
    return y
end

MultivariateStats.transform(M::PCA, x::AbstractVecOrMat) =
    mul!(similar(x), transpose(M.proj), centralize(x, M.mean))

function show_pdf(url::String; width::Integer = 500, height::Integer = 500)
    node(
        :object, "pdf not found",
        attributes = Dict(
            :type => "application/pdf",
            :width => "$width",
            :height => "$height",
            :data => "$url"
        )
    )
end

places = [
    [250, 750, 1200, 450, 1050, 900],
    [500, 570,  650, 350,  450, 570],
    ["Mexico", "Germany", "Siberia", "Amazon", "India", "Moscow"]
]
function colorbar_2d_teuling2011(x, y, xlim, ylim, nonsignificant_color = "#FFFFFF")

    col = [
        "#F58439" "#FDB739" "#BED85A" "#76C04B" "#00AC4E";
        "#F47C57" "#FAB380" "#D5E6A2" "#81C99B" "#00B189";
        "#F1687A" "#F6A3AE" "#F0F0F0" "#77CDCE" "#00B4B7";
        "#EE4498" "#C880B6" "#A8A4D1" "#44B5E8" "#00B3E5";
        "#A53F97" "#7E54A2" "#596AB2" "#1977BD" "#007EC4"
    ] |>
        x -> x[:, end:-1:1] |>
        #permutedims |>
        collect

    scale_1_5(xx, xxlim) =
        ((xx - xxlim[1]) / (xxlim[2] - xxlim[1]) * 5) |>
            xxx -> ceil(Int, xxx) |>
            xxx -> clamp(xxx, 1, 5)

    map(x, y) do xx, yy
        if ismissing(xx) || ismissing(yy)
            missing
        elseif xx == 0 && yy == 0
            # the actual zeros are the nonsignificant values
            nonsignificant_color
        else
            i = scale_1_5(xx, xlim)
            j = scale_1_5(yy, ylim)
            col[i, j]
        end
    end |> permutedims |> collect
end

get_q(xx, q) = xx |>
    collect |>
    missing_to_nan |>
    vec |>
    xxx -> filter(!isnan, xxx) |>
    xxx -> quantile(xxx, [q, 1 - q]) |>
    xxx -> abs.(xxx) |>
    xxx -> max(xxx...) |>
    xxx -> (-xxx, xxx)

function colorbar_2d_teuling2011_quant(x, y, xquant, yquant, nonsignificant_color = "#FFFFFF")
    #xquant = xquant < 0.5 ? xquant : (1 - xquant)
    #yquant = yquant < 0.5 ? yquant : (1 - yquant)

    #get_q(xx, q) = xx |>
    #    collect |>
    #    missing_to_nan |>
    #    vec |>
    #    xxx -> filter(!isnan, xxx) |>
    #    xxx -> quantile(xxx, [q, 1 - q]) |>
    #    xxx -> abs.(xxx) |>
    #    xxx -> max(xxx...) |>
    #    xxx -> (-xxx, xxx)

    #xq = get_q(x, xquant)
    #yq = get_q(y, yquant)

    xq = xquant
    yq = yquant

    get_ticks(lim) = [
        lim[1] + (lim[2] - lim[1]) / 5 * i
        for i in 0:5
    ]

    # WARNING: we permute dims (and x and y) here, because Julia matrices have column major layout:
    Dict(
        :col_map => permutedims(colorbar_2d_teuling2011(x, y, xq, yq, nonsignificant_color)),
        :x_ticks => get_ticks(xq),
        :y_ticks => get_ticks(yq)
    )
end

function sen_naive_0(y::AbstractVector{Union{Missing, T}}) where T
    n = length(y)

    ss = fill(T(NaN), n, n)
    @inbounds for i in 1:n
        for j in 1:n
            tmp = (y[i] - y[j]) / (i - j)
            ss[j, i] = !ismissing(tmp) ? tmp : NaN
        end
    end
    return median(filter(!isnan, ss))
end

function sen_naive_1(y::AbstractVector{Union{Missing, T}}) where T
    n = length(y)
    ss = (y .- y') ./ ((1:n) .- (1:n)')
    median(filter(x -> !ismissing(x) && !isnan(x), ss))
end

function sen_naive_5(y::AbstractVector{Union{Missing, T}}) where T
    n = length(y)
    ns = ((n ^ 2) - n) ÷ 2

    ss = Array{T}(undef, ns)
    k = 1
    @inbounds for i in 1:(n - 1)
        for j in (i + 1):n
            tmp = (y[i] - y[j]) / (i - j)
            ss[k] = !ismissing(tmp) ? tmp : T(NaN)
            k += 1
        end
    end
    median(filter!(!isnan, ss))
end

function sen_naive_3(y::AbstractVector{Union{Missing, T}}) where T
    n = length(y)
    ns = ((n ^ 2) - n) ÷ 2

    ss = Array{T}(undef, ns)
    k = 1
    @inbounds for i in 1:(n - 1)
        for j in (i + 1):n
            tmp = (y[i] - y[j]) / (i - j)
            if !ismissing(tmp)
                ss[k] = tmp
                k += 1
            end
        end
    end

    if k - 1 != ns
        resize!(ss, k - 1)
    end

    ns2 = length(ss)

    sort!(ss)

    if isodd(ns2)
        ss[(ns2 ÷ 2) + 1]
    else
        (ss[ns2 ÷ 2] + ss[(ns2 ÷ 2) + 1]) / 2
    end
end

function sen_bc(x::AbstractArray, y::AbstractArray)

end

pca_axis = CategoricalAxis(:PcaAxis, ["PCA_$i" for i in 1:length(vars)])
