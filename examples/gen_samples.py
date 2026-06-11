"""Generate 4 sample CSV files for PCA testing"""
import numpy as np
np.random.seed(42)

x = np.arange(10, 70, 0.5)

# Sample A: strong peak at 26.6, medium at 20.8
y_a = 12 + 90*np.exp(-(x-26.6)**2/0.4) + 35*np.exp(-(x-20.8)**2/0.7) + np.random.randn(len(x))*1.5
with open('examples/sample_a.csv', 'w') as f:
    f.write('# title: Sample A XRD\n')
    for i in range(len(x)):
        f.write(f'{x[i]:.1f},{y_a[i]:.2f}\n')

# Sample B: peaks at different positions (30.5, 45.2)
y_b = 10 + 70*np.exp(-(x-30.5)**2/0.5) + 50*np.exp(-(x-45.2)**2/0.6) + np.random.randn(len(x))*1.5
with open('examples/sample_b.csv', 'w') as f:
    f.write('# title: Sample B XRD\n')
    for i in range(len(x)):
        f.write(f'{x[i]:.1f},{y_b[i]:.2f}\n')

# Sample C: similar to A (26.6, 20.8) but weaker
y_c = 8 + 55*np.exp(-(x-26.6)**2/0.45) + 25*np.exp(-(x-20.8)**2/0.65) + np.random.randn(len(x))*1.5
with open('examples/sample_c.csv', 'w') as f:
    f.write('# title: Sample C XRD\n')
    for i in range(len(x)):
        f.write(f'{x[i]:.1f},{y_c[i]:.2f}\n')

# Sample D: similar to B (30.5, 45.2) but stronger
y_d = 14 + 95*np.exp(-(x-30.5)**2/0.55) + 65*np.exp(-(x-45.2)**2/0.65) + np.random.randn(len(x))*1.5
with open('examples/sample_d.csv', 'w') as f:
    f.write('# title: Sample D XRD\n')
    for i in range(len(x)):
        f.write(f'{x[i]:.1f},{y_d[i]:.2f}\n')

print("Generated 4 sample files")
