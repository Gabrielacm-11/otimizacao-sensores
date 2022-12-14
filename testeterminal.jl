using HiGHS
using JuMP, BenchmarkTools
using PlotlyJS, LinearAlgebra,Distributions,Random
include("funcoesLP.jl")
	Random.seed!(15)
	println("valores de X, digitar [[12.,12],[12.,12]]")
	X =  readline()
	println("valor de Xₖ, digitar [12.,12]")
	xₖ=  readline()
	v = 2
	σ = [1,2,3]
	ω = Float64[]
	for σi in σ
		push!(ω,rand(Normal(0,σi/v)))
	end
	n = length(X)
	@assert xₖ ∉ X "xₖ belongs to X"
	v = 0.001
	σ = fill(4.,n)
	R = calcula_rᵢ(n,σ,xₖ,X,v)
	Aᵢ(xᵢ) = ((xₖ - X[end])/norm(xₖ - X[end]))  - ((xₖ - xᵢ)/norm(xₖ - xᵢ))
	Cᵢ(xᵢ) = norm(xₖ - X[end]) -  norm(xₖ - xᵢ) + dot(Aᵢ(xᵢ),-xₖ)
	C = R +  Cᵢ.(X[1:end-1])
	A = Aᵢ.(X[1:end-1])
	A = permutedims(hcat(A...))
	solucao_LP(A,C,n)
	


v = 2
	σ = [1,2,3]
	ω = Float64[]
	for σi in σ
		push!(ω,rand(Normal(0,σi/v)))
	end
	n = length(X)
	@assert xₖ ∉ X "xₖ belongs to X"
	v = 0.001
	σ = fill(4.,n)
	R = calcula_rᵢ(n,σ,xₖ,X,v)
	Aᵢ(xᵢ) = ((xₖ - X[end])/norm(xₖ - X[end]))  - ((xₖ - xᵢ)/norm(xₖ - xᵢ))
	Cᵢ(xᵢ) = norm(xₖ - X[end]) -  norm(xₖ - xᵢ) + dot(Aᵢ(xᵢ),-xₖ)
	C = R +  Cᵢ.(X[1:end-1])
	A = Aᵢ.(X[1:end-1])
	A = permutedims(hcat(A...))
	solucao_LP(A,C,n)
