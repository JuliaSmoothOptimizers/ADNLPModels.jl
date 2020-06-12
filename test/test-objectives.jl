# define a few objective functions

function arglina(x)
  n = length(x)
  m = 2 * n
  return sum((x[i] - 2/m * sum(x[j] for j = 1:n) - 1)^2 for i = 1:n) + sum((-2/m * sum(x[j] for j = 1:n) - 1)^2 for i = n+1:m)
end

function arwhead(x)
  n = length(x)
  return sum((x[i]^2 + x[n]^2)^2 - 4 * x[i] + 3 for i=1:n-1)
end

function chainwoo(x)
  n = length(x)
  n % 4 == 0 || error("number of variables must be a multiple of 4")
  return 1.0 + sum(100 * (x[2*i]   - x[2*i-1]^2)^2 + (1 - x[2*i-1])^2 +
              90 * (x[2*i+2] - x[2*i+1]^2)^2 + (1 - x[2*i+1])^2 +
              10 * (x[2*i] + x[2*i+2] - 2)^2 + 0.1 * (x[2*i] - x[2*i+2])^2 for i=1:div(n,2)-1)
end

test_objectives = [arglina, arwhead, chainwoo]
