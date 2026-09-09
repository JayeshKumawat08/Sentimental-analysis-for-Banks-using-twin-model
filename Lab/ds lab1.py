import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)


#Markov inequality
X = df['total_bill'].values
N = len(X)
a = 10
mean_X=np.mean(X)

empirical_markov = np.mean(X >= a)
markov_bound = np.mean(X) / a


print("\t\t\t Markov Inequality")
print(f"Total Records (N):{N}")
print(f"Average Bill (E[X] : ${mean_X:.2f})")
print(f"P(X >= {a}) Empirical : {empirical_markov:.2f}")
print(f"Markov Bound E[X]/{a} : {markov_bound:.2f}")
print()

plt.figure(figsize=(10, 6))
plt.hist(X, bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='Restaurant Bills')

plt.axvline(mean_X, color='orange', linestyle='dashed', linewidth=2, label=f'Mean (E[X]) = ${mean_X:.2f}')
plt.axvline(a, color='red', linestyle='solid', linewidth=2, label=f'Threshold (a) = ${a:.2f}')


plt.title('Distribution of Restaurant Bills with Markov Bounds')
plt.xlabel('Total Bill Amount ($)')
plt.ylabel('Number of Customers (Frequency)')
plt.legend()

plt.show()

print("-"*50)

X = df['total_bill'].values
mu = np.mean(X)
sigma = np.std(X)
k = 2

lower_bound = mu - (k*sigma)
upper_bound = mu - (k*sigma)

empirical_chebyshev = np.mean(np.abs(X - mu) >= k * sigma)
chebyshev_bound = 1 / (k ** 2)

print("\t\t\tChebyshev Inequality ")
print(f"Total Records(N) : {N}")
print(f"Average Bill(μ) : ${mu:.2f}")
print(f"Standard Deviation(σ) :${sigma:.2f}")
print(f"k Multiplier : {k}")
print("-"*50)
print(f"P(|X-μ| >= {k}σ) Empirical : {empirical_chebyshev:.4f}")
print(f"Chebyshev Bound : {chebyshev_bound:.4f}")
print()

plt.figure(figsize=(10, 6))
plt.hist(X, bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='Restaurant Bills')

#mean
plt.axvline(mu, color='orange', linestyle='dashed', linewidth=2, label=f'Mean (μ) = ${mu:.2f}')

#Chebyshev Boundaries
plt.axvline(lower_bound, color='red', linestyle='solid', linewidth=2, label=f'Lower Bound (μ - {k}σ)')
plt.axvline(upper_bound, color='red', linestyle='solid', linewidth=2, label=f'Upper Bound (μ + {k}σ)')


plt.axvspan(xmin=min(X)-5, xmax=lower_bound, color='red', alpha=0.1)
plt.axvspan(xmin=upper_bound, xmax=max(X)+5, color='red', alpha=0.1, label='Chebyshev Region')

plt.title(f'Chebyshev Bounds (k={k}) on Restaurant Bills')
plt.xlabel('Total Bill Amount ($)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

print("-"*50)


#Coin flips example for Chernoff
N = 100000
n = 50
p = 0.5

samples = np.random.binomial(1,p, size =(N,n))
S = np.sum(samples, axis=1)

mu = n * p
delta = 0.4

target = (1 + delta) * mu
empirical_chernoff = np.mean(S >= target)
chernoff_bound = np.exp(-(mu * delta**2) / (2 + delta))

print("\t\t\tChernoff Bound ")
print(f"Experimental run(N) :{N}")
print(f"Expected Heads(μ) :{mu}")
print(f"Threshold Target : {target} heads")
print('-'*50)
print(f"Empirical Probability : {empirical_chernoff:.6f}")
print(f"Chernoff Bound : {chernoff_bound:.6f}")


plt.figure(figsize=(10, 6))

#distribution of heads
plt.hist(S, bins=25, density=True, color='purple', edgecolor='black', alpha=0.6, label='Sum of Coin Flips')

#Mean
plt.axvline(mu, color='gold', linestyle='dashed', linewidth=2, label=f'Mean (μ) = {mu}')

#Target Threshold
plt.axvline(target, color='red', linestyle='solid', linewidth=2, label=f'Threshold = {target}')

#The Danger Zone
plt.axvspan(xmin=target, xmax=max(S)+2, color='red', alpha=0.2, label='Chernoff Tail (Extreme Rare Event)')

plt.title("Chernoff Bound: Probability of Extreme Deviations in Coin Flips")
plt.xlabel("Total Sum (Number of Heads)")
plt.ylabel("Density / Frequency")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

print("-"*50)
