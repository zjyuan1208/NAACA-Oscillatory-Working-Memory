import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Parameters from paper
G = 64  # Grid size
C = 527  # Number of PANN dimensions
dt = 0.01  # Time step
dx = 1  # Spatial step
kp = kv = 10.0  # Damping coefficients
f_min, f_max = 50, 1200  # Target frequency range in Hz

# Compute target frequencies for each PANN dimension (linear mapping)
target_frequencies = np.linspace(f_min, f_max, C)

# Compute wave speed for each PANN dimension using Theorem 2.1
# c_i = tan(π f_i Δt) / (sqrt((1+Δt k_p)(1+Δt k_v)) * Δt * sqrt(2))
def compute_wave_speed(f_i):
    numerator = np.tan(np.pi * f_i * dt)
    denominator = np.sqrt((1 + dt * kp) * (1 + dt * kv)) * dt * np.sqrt(2)
    c_i = numerator / denominator
    # Clamp to [0.1, 70] for numerical stability
    c_i = np.clip(c_i, 0.1, 70.0)
    return c_i

wave_speeds = np.array([compute_wave_speed(f) for f in target_frequencies])

print(f"Wave speed range: [{wave_speeds.min():.2f}, {wave_speeds.max():.2f}]")
print(f"Number of unique speeds: {len(np.unique(wave_speeds))}")

# Map PANN dimensions to spatial parcels (row-major order)
# Each parcel gets approximately floor(G^2 / C) ≈ 7-8 grid points
c_field = np.zeros((G, G))

n_per_parcel = G * G // C  # ≈ 7-8 points per dimension
print(f"Points per parcel: {n_per_parcel}")

# Assign wave speeds to parcels in row-major order
for i in range(C):
    start_idx = i * n_per_parcel
    end_idx = min((i + 1) * n_per_parcel, G * G)
    
    # Convert linear indices to 2D coordinates (row-major)
    for linear_idx in range(start_idx, end_idx):
        y = linear_idx // G  # Row index
        x = linear_idx % G   # Column index
        c_field[y, x] = wave_speeds[i]

# Handle remaining points (assign to last PANN dimension)
remaining_start = C * n_per_parcel
for linear_idx in range(remaining_start, G * G):
    y = linear_idx // G
    x = linear_idx % G
    c_field[y, x] = wave_speeds[-1]

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Custom colormap: blue (slow) to yellow (fast)
colors = ['#2E4057', '#048BA8', '#16DB93', '#EFEA5A', '#F29E4C']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('wave_speed', colors, N=n_bins)

# Panel 1: Full c(x,y) field
im1 = axes[0].imshow(c_field, cmap=cmap, aspect='equal', origin='lower')
axes[0].set_xlabel('Grid X', fontsize=12)
axes[0].set_ylabel('Grid Y', fontsize=12)
axes[0].set_title('(a) OWM Wave Speed Field $c(x,y)$', fontsize=13, fontweight='bold')
cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
cbar1.set_label('Wave Speed $c$', fontsize=11)
axes[0].grid(False)

# Panel 2: Y-direction profile (showing striped structure)
# Average along x-direction to show the y-dependent structure
c_profile_y = np.mean(c_field, axis=1)
axes[1].plot(c_profile_y, np.arange(G), linewidth=2, color='#048BA8')
axes[1].fill_betweenx(np.arange(G), 0, c_profile_y, alpha=0.3, color='#048BA8')
axes[1].set_xlabel('Average Wave Speed $\\langle c \\rangle_x$', fontsize=12)
axes[1].set_ylabel('Grid Y', fontsize=12)
axes[1].set_title('(b) Y-direction Profile (Striped Structure)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].set_xlim([0, 75])

# Panel 3: Histogram of wave speeds
axes[2].hist(c_field.flatten(), bins=50, color='#048BA8', alpha=0.7, edgecolor='black')
axes[2].set_xlabel('Wave Speed $c$', fontsize=12)
axes[2].set_ylabel('Frequency (# of grid points)', fontsize=12)
axes[2].set_title('(c) Distribution of Wave Speeds', fontsize=13, fontweight='bold')
axes[2].axvline(np.median(c_field), color='red', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(c_field):.1f}')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
plt.savefig('owm_wave_speed_field.png', dpi=300, bbox_inches='tight')
plt.savefig('owm_wave_speed_field.pdf', bbox_inches='tight')
print("Figure saved as 'owm_wave_speed_field.png' and 'owm_wave_speed_field.pdf'")
plt.show()

# Print statistics
print("\n=== Wave Speed Field Statistics ===")
print(f"Min speed: {c_field.min():.2f}")
print(f"Max speed: {c_field.max():.2f}")
print(f"Mean speed: {c_field.mean():.2f}")
print(f"Median speed: {np.median(c_field):.2f}")
print(f"Std deviation: {c_field.std():.2f}")
print(f"Binary contrast (max/min): {c_field.max()/c_field.min():.1f}")

# Verify striped structure (should be constant along x for each y)
x_variance = np.var(c_field, axis=1)
print(f"\nVariance along x-direction (should be ~0 for perfect stripes):")
print(f"Max variance: {x_variance.max():.2e}")
print(f"Mean variance: {x_variance.mean():.2e}")

# Additional visualization: Show a few example parcels
fig2, ax = plt.subplots(1, 1, figsize=(10, 10))
im = ax.imshow(c_field, cmap=cmap, aspect='equal', origin='lower')

# Highlight a few example parcels
example_parcels = [0, 100, 250, 400, 526]  # First, and a few others
for i in example_parcels:
    start_idx = i * n_per_parcel
    end_idx = min((i + 1) * n_per_parcel, G * G)
    
    # Get bounding box of this parcel
    y_coords = [idx // G for idx in range(start_idx, end_idx)]
    x_coords = [idx % G for idx in range(start_idx, end_idx)]
    
    if len(y_coords) > 0:
        y_min, y_max = min(y_coords), max(y_coords)
        x_min, x_max = min(x_coords), max(x_coords)
        
        # Draw rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle((x_min-0.5, y_min-0.5), x_max-x_min+1, y_max-y_min+1,
                         linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        ax.text(x_min, y_max+1, f'P{i}\n{target_frequencies[i]:.0f}Hz\nc={wave_speeds[i]:.1f}',
                fontsize=8, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('Grid X', fontsize=12)
ax.set_ylabel('Grid Y', fontsize=12)
ax.set_title('OWM Wave Speed Field with Example Parcels', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Wave Speed $c$', fontsize=11)
ax.grid(False)

plt.tight_layout()
plt.savefig('owm_wave_speed_field_parcels.png', dpi=300, bbox_inches='tight')
print("Detailed figure saved as 'owm_wave_speed_field_parcels.png'")
plt.show()