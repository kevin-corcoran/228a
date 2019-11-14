using LinearAlgebra
using SparseArrays

# Laplacian
function Laplacian(S, N)
    L = spzeros(S, N)
    L[diagind(L,0)]  .= 2 # put 2 along diagonal
    L[diagind(L,1)] .= 1 # put 1 along superdiagonal
    L[diagind(L,-1)] .= 1 # put 1 along subdiagonal
    # deal with boundary conditions
    L[1,end] = 1
    L[end,1] = 1
    # L
    return L
end

function RkAdapt(N,S,T,u0)
    u = spzeros(S*3, N)
    # add two more copies to each element below each element. For use in Runge-Kutta
    u0 = [vcat(u0[i],u0[i],u0[i]) for i = 1:S]
    u0 = vcat(u0...)
    u[:, 1] = u0
    h = T/N

    b1 = [1/4, 3/4]
    b2 = [1/4, 3/8, 3/8]
    A = [0 0 0; 2/3 0 0; 0 2/3 0]
    L = Laplacian(S,N)
    # add two more copies of each row just below each row. For use in Runge-Kutta
    L = [vcat(L[i,:]',L[i,:]',L[i,:]') for i = 1:S]
#     L = permutedims(reshape(hcat(L...), length(L[1]), length(L)))'
    L = vcat(L...)

    e = [1,1,1]
    for i = 1 : N-1
        for j = 1:S
            # reshape Laplacian
            J = findnz(L[j:j+2,:])[2] # column indices of nonzero values
            T = reshape(L[3*j,J],(3,3))
            k = (I-h*T*A)^(-1)*T*u[j:j+2,i]
            u[j:j+2, i+1] = u[j:j+2, i] + (h*b2'*k)*e
        end
    end
    # remove duplicated elements
    u = u[1:3:end] # start:step:stop
    return u
end

N = 100 # number of steps
S = 10 # number of spaces in grid
T = 1
a = 0
b = 2*pi
f(x) = exp(-5*(x-pi)^2)
xs = range(a,stop=b,length=S)
u₀ = [f(i) for i in xs] # discritized initial condition
A = RkAdapt(N,S,T,u₀)
figure(1,figsize=(5,5))
plot(xs[1:99], A[1:99, 1000],"r-",label=L"ans")

# function RkAdapt(N,S,T,u0)
#     u = zeros(S*3, N)
#     # add two more copies to each element below each element. For use in Runge-Kutta
#     u0 = [vcat(u0[i],u0[i],u0[i]) for i = 1:S]
#     u0 = vcat(u0...)
#     u[:, 1] = u0
#     h = T/N

#     b1 = [1/4, 3/4, 0]
#     b2 = [1/4, 3/8, 3/8]
#     A = [0 0 0; 2/3 0 0; 0 2/3 0]
#     L = Laplacian(S,N)
#     # add two more copies of each row just below each row. For use in Runge-Kutta
#     L = [vcat(L[i,:]',L[i,:]',L[i,:]') for i = 1:S]
# #     L = permutedims(reshape(hcat(L...), length(L[1]), length(L)))'
#     L = vcat(L...)

#     Δx = 2*pi/100
#     e = [1,1,1]
#     for i = 1 : N-1
#         for j = 1:S
#             # reshape Laplacian
#             J = findnz(L[j:j+2,:])[2] # column indices of nonzero values
#             T = reshape(L[3*j,J],(3,3))
#             k = (I-h*T*A/Δx^2)^(-1)*(T*u[j:j+2,i]/Δx^2)
#             u[j:j+2, i+1] = u[j:j+2, i] + (h*b2'*k)*e
#         end
#     end
#     # remove duplicated elements
#     u = u[1:3:end,:] # start:step:stop
#     return u
# end
