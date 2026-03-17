#!/usr/bin/env python3
"""
8-Qubit MMA-UPTE with 3 Correlated Noise Models
Depolarizing, Amplitude Damping, Phase Damping
MMA optimization, virtual distillation
"""
import numpy as np
import scipy.linalg as la
from qiskit.quantum_info import SparsePauliOp
import warnings
import time

warnings.filterwarnings('ignore')

N_QUBITS = 8
DIM = 2**N_QUBITS
MAX_ITERATIONS = 40

DEPOL_BASE = 0.005
DEPOL_CORR_LENGTH = 2.0
DEPOL_CORR_STRENGTH = 0.3
AMP_DAMP_BASE = 0.008
AMP_DAMP_CORR_LENGTH = 2.0
AMP_DAMP_CORR_STRENGTH = 0.25
PHASE_DAMP_BASE = 0.012
PHASE_DAMP_CORR_LENGTH = 2.0
PHASE_DAMP_CORR_STRENGTH = 0.3
VD_COPIES = 2

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI_TERMS = [f"{p1}{p2}" for p1 in ['X', 'Y', 'Z'] for p2 in ['X', 'Y', 'Z']]
N_TERMS = len(PAULI_TERMS)
N_PARAMS = 2 * N_TERMS


def build_full_operator_from_list(op_list):
    if not op_list:
        return np.eye(1, dtype=complex)
    full_op = op_list[0]
    for q_idx in range(1, len(op_list)):
        full_op = np.kron(full_op, op_list[q_idx])
    return full_op


def apply_correlated_depolarizing_noise(rho_ideal, n_qubits, base_strength,
                                        correlation_length, correlation_strength):
    rho_current = rho_ideal.copy()
    for qubit in range(n_qubits):
        p_single = base_strength
        K_ops = [
            np.sqrt(1 - 3*p_single/4) * I,
            np.sqrt(p_single/4) * X,
            np.sqrt(p_single/4) * Y,
            np.sqrt(p_single/4) * Z
        ]
        rho_qubit_noisy = np.zeros_like(rho_current)
        for K_1q in K_ops:
            op_list = [I] * n_qubits
            op_list[qubit] = K_1q
            K_full = build_full_operator_from_list(op_list)
            rho_qubit_noisy += K_full @ rho_current @ K_full.conj().T
        rho_current = rho_qubit_noisy
    for qubit1 in range(n_qubits-1):
        qubit2 = qubit1 + 1
        distance = 1.0
        p_corr = correlation_strength * base_strength * np.exp(-distance / correlation_length)
        if p_corr > 1e-12:
            rho_2q_noisy = (1.0 - p_corr) * rho_current
            for pauli_pair in ['XX', 'YY', 'ZZ']:
                P1 = globals()[pauli_pair[0]]
                P2 = globals()[pauli_pair[1]]
                op_list = [I] * n_qubits
                op_list[qubit1] = P1
                op_list[qubit2] = P2
                P_full = build_full_operator_from_list(op_list)
                rho_2q_noisy += (p_corr / 3.0) * P_full @ rho_current @ P_full.conj().T
            rho_current = rho_2q_noisy
    trace = np.trace(rho_current)
    if abs(trace) > 1e-12:
        rho_current = rho_current / trace
    return rho_current


def apply_correlated_amplitude_damping(rho_ideal, n_qubits, gamma_base,
                                       correlation_length, correlation_strength):
    rho_current = rho_ideal.copy()
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma_base)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma_base)], [0, 0]], dtype=complex)
    K_ops = [K0, K1]
    for qubit in range(n_qubits):
        rho_single_noisy = np.zeros_like(rho_current)
        for K in K_ops:
            op_list = [I] * n_qubits
            op_list[qubit] = K
            K_full = build_full_operator_from_list(op_list)
            rho_single_noisy += K_full @ rho_current @ K_full.conj().T
        rho_current = rho_single_noisy
    for qubit1 in range(n_qubits - 1):
        qubit2 = qubit1 + 1
        distance = 1.0
        p_corr = correlation_strength * gamma_base * np.exp(-distance / correlation_length)
        if p_corr > 1e-12:
            rho_2q_noisy = (1.0 - p_corr) * rho_current
            for K1a, K1b in [(K0, K0), (K0, K1), (K1, K0), (K1, K1)]:
                op_list = [I] * n_qubits
                op_list[qubit1] = K1a
                op_list[qubit2] = K1b
                K_full = build_full_operator_from_list(op_list)
                rho_2q_noisy += (p_corr / 4.0) * K_full @ rho_current @ K_full.conj().T
            rho_current = rho_2q_noisy
    trace = np.trace(rho_current)
    if abs(trace) > 1e-12:
        rho_current = rho_current / trace
    return rho_current


def apply_correlated_phase_damping(rho_ideal, n_qubits, lambda_base,
                                   correlation_length, correlation_strength):
    rho_current = rho_ideal.copy()
    K0 = np.array([[1, 0], [0, np.sqrt(1 - lambda_base)]], dtype=complex)
    K1 = np.array([[0, 0], [0, np.sqrt(lambda_base)]], dtype=complex)
    K_ops = [K0, K1]
    for qubit in range(n_qubits):
        rho_single_noisy = np.zeros_like(rho_current)
        for K in K_ops:
            op_list = [I] * n_qubits
            op_list[qubit] = K
            K_full = build_full_operator_from_list(op_list)
            rho_single_noisy += K_full @ rho_current @ K_full.conj().T
        rho_current = rho_single_noisy
    for qubit1 in range(n_qubits - 1):
        qubit2 = qubit1 + 1
        distance = 1.0
        p_corr = correlation_strength * lambda_base * np.exp(-distance / correlation_length)
        if p_corr > 1e-12:
            rho_2q_noisy = (1.0 - p_corr) * rho_current
            for K1a, K1b in [(K0, K0), (K0, K1), (K1, K0), (K1, K1)]:
                op_list = [I] * n_qubits
                op_list[qubit1] = K1a
                op_list[qubit2] = K1b
                K_full = build_full_operator_from_list(op_list)
                rho_2q_noisy += (p_corr / 4.0) * K_full @ rho_current @ K_full.conj().T
            rho_current = rho_2q_noisy
    trace = np.trace(rho_current)
    if abs(trace) > 1e-12:
        rho_current = rho_current / trace
    return rho_current


def apply_all_correlated_noises(rho_ideal, n_qubits,
                                depol_base, depol_corr_len, depol_corr_str,
                                amp_base, amp_corr_len, amp_corr_str,
                                phase_base, phase_corr_len, phase_corr_str):
    rho = rho_ideal.copy()
    rho = apply_correlated_depolarizing_noise(
        rho, n_qubits, depol_base, depol_corr_len, depol_corr_str)
    rho = apply_correlated_amplitude_damping(
        rho, n_qubits, amp_base, amp_corr_len, amp_corr_str)
    rho = apply_correlated_phase_damping(
        rho, n_qubits, phase_base, phase_corr_len, phase_corr_str)
    return rho


def virtual_distillation_enhancement(rho_noisy, psi_target, n_copies=2):
    rho_M = rho_noisy
    for _ in range(n_copies - 1):
        rho_M = rho_M @ rho_noisy
    trace_rho_M = np.trace(rho_M)
    if abs(trace_rho_M) < 1e-12:
        return 0.0
    rho_distilled = rho_M / trace_rho_M
    fidelity = np.real(psi_target.conj().T @ rho_distilled @ psi_target)
    return np.clip(fidelity, 0.0, 1.0)


class TopologyAwareUPTE:
    def __init__(self, n_qubits, pauli_terms):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.pauli_terms = pauli_terms
        self.n_terms = len(pauli_terms)
        self.H_terms = self._get_pauli_operators()

    def _get_pauli_operators(self):
        H_terms = []
        for pauli_str in self.pauli_terms:
            op = SparsePauliOp(pauli_str, coeffs=[1.0])
            H_matrix = op.to_matrix(sparse=False)
            H_terms.append(H_matrix)
        return H_terms

    def construct_upte_operator(self, params):
        rho = params[:self.n_terms]
        tau = params[self.n_terms:]
        rho_positive = np.maximum(rho, 1e-6)
        rho_normalized = rho_positive / (np.sum(rho_positive) + 1e-12)
        delta = 1.0
        B = np.zeros((self.dim, self.dim), dtype=complex)
        for j in range(self.n_terms):
            H_j = self.H_terms[j]
            U_j = la.expm(-1j * H_j * delta * tau[j])
            B += rho_normalized[j] * U_j
        try:
            W, s, Vh = la.svd(B)
            U_safe = W @ Vh
        except la.LinAlgError:
            U_safe = np.eye(self.dim, dtype=complex)
        return U_safe


def noise_aware_objective_function(params, topology_optimizer, U_target):
    U_circuit = topology_optimizer.construct_upte_operator(params)
    psi_0 = np.zeros(topology_optimizer.dim, dtype=complex)
    psi_0[0] = 1.0
    psi_ideal = U_circuit @ psi_0
    rho_ideal = np.outer(psi_ideal, np.conj(psi_ideal))
    rho_noisy = apply_all_correlated_noises(
        rho_ideal,
        topology_optimizer.n_qubits,
        DEPOL_BASE, DEPOL_CORR_LENGTH, DEPOL_CORR_STRENGTH,
        AMP_DAMP_BASE, AMP_DAMP_CORR_LENGTH, AMP_DAMP_CORR_STRENGTH,
        PHASE_DAMP_BASE, PHASE_DAMP_CORR_LENGTH, PHASE_DAMP_CORR_STRENGTH
    )
    psi_target = U_target @ psi_0
    ideal_fidelity = np.abs(np.vdot(psi_target, psi_ideal))**2
    noisy_fidelity = np.real(psi_target.conj().T @ rho_noisy @ psi_target)
    noisy_fidelity = np.clip(noisy_fidelity, 0.0, 1.0)
    return 1.0 - noisy_fidelity, noisy_fidelity, ideal_fidelity, rho_noisy


def reliable_hybrid_gradient(params, topology_optimizer, U_target):
    n_terms = topology_optimizer.n_terms
    n_samples = 2
    grad_samples = []
    alpha_momentum = 0.4
    for _ in range(n_samples):
        grad_sample = np.zeros_like(params)
        eps_rho = 1e-4
        for i in range(n_terms):
            params_plus = np.clip(params.copy(), 1e-6, None)
            params_plus[i] += eps_rho
            loss_plus, _, _, _ = noise_aware_objective_function(
                params_plus, topology_optimizer, U_target)
            params_minus = np.clip(params.copy(), 1e-6, None)
            params_minus[i] -= eps_rho
            loss_minus, _, _, _ = noise_aware_objective_function(
                params_minus, topology_optimizer, U_target)
            grad_sample[i] = (loss_plus - loss_minus) / (2 * eps_rho)
        eps_tau = np.pi / 4.0
        for i in range(n_terms, N_PARAMS):
            params_plus = params.copy()
            params_plus[i] += eps_tau
            loss_plus, _, _, _ = noise_aware_objective_function(
                params_plus, topology_optimizer, U_target)
            params_minus = params.copy()
            params_minus[i] -= eps_tau
            loss_minus, _, _, _ = noise_aware_objective_function(
                params_minus, topology_optimizer, U_target)
            grad_sample[i] = (loss_plus - loss_minus) / (2 * eps_tau)
        grad_samples.append(grad_sample)
    grad_avg = np.mean(grad_samples, axis=0)
    if not hasattr(reliable_hybrid_gradient, 'previous_grad'):
        reliable_hybrid_gradient.previous_grad = grad_avg
        return grad_avg
    grad_smoothed = (alpha_momentum * grad_avg +
                     (1 - alpha_momentum) * reliable_hybrid_gradient.previous_grad)
    reliable_hybrid_gradient.previous_grad = grad_smoothed
    return grad_smoothed


class MMAOptimizer:
    def __init__(self, n_params):
        self.n_params = n_params
        self.move_limit = 0.4
        self.gamma = 0.5
        self.L = None
        self.U = None
        self.previous_params = None
        self.previous_loss = None

    def initialize_bounds(self, initial_params):
        self.L = initial_params - 0.6
        self.U = initial_params + 0.6
        self.previous_params = initial_params.copy()

    def solve_convex_subproblem(self, current_params, grad):
        x_new = np.zeros_like(current_params)
        for i in range(self.n_params):
            g_i = grad[i]
            x_i = current_params[i]
            L_i = self.L[i]
            U_i = self.U[i]
            if g_i < 0:
                p_i = abs(g_i) * (U_i - x_i)**2
                q_i = 0.0
            else:
                p_i = 0.0
                q_i = abs(g_i) * (x_i - L_i)**2
            denominator = (p_i / (U_i - x_i + 1e-12)**2 +
                           q_i / (x_i - L_i + 1e-12)**2)
            if denominator > 1e-12:
                numerator = (p_i / (U_i - x_i + 1e-12) -
                             q_i / (x_i - L_i + 1e-12))
                x_new[i] = x_i + numerator / denominator
            else:
                x_new[i] = x_i
            x_new[i] = max(x_i - self.move_limit,
                           min(x_i + self.move_limit, x_new[i]))
            x_new[i] = np.clip(x_new[i], L_i + 1e-6, U_i - 1e-6)
        return x_new

    def update_asymptotes(self, current_params, current_loss):
        progress_good = True
        if self.previous_loss is not None:
            progress_good = (current_loss < self.previous_loss - 1e-8)
        for i in range(self.n_params):
            if progress_good:
                self.L[i] = (current_params[i] -
                             (1.2 / self.gamma) * (current_params[i] - self.L[i]))
                self.U[i] = (current_params[i] +
                             (1.2 / self.gamma) * (self.U[i] - current_params[i]))
            else:
                self.L[i] = (current_params[i] -
                             self.gamma * (current_params[i] - self.L[i]))
                self.U[i] = (current_params[i] +
                             self.gamma * (self.U[i] - current_params[i]))
        self.L = np.minimum(self.L, self.U - 1e-4)
        self.U = np.maximum(self.U, self.L + 1e-4)
        self.previous_params = current_params.copy()
        self.previous_loss = current_loss


def run_mma_optimization(initial_params, topology_optimizer, U_target,
                         max_iterations=MAX_ITERATIONS):
    mma = MMAOptimizer(len(initial_params))
    mma.initialize_bounds(initial_params)
    current_params = initial_params.copy()
    current_loss, current_fidelity, current_ideal_fid, _ = \
        noise_aware_objective_function(current_params, topology_optimizer, U_target)

    print(f"Initial: Noisy Fidelity={current_fidelity:.6f}, "
          f"Ideal Fidelity={current_ideal_fid:.6f}")
    best_fidelity = current_fidelity
    best_params = current_params.copy()
    stagnation_count = 0

    if hasattr(reliable_hybrid_gradient, 'previous_grad'):
        del reliable_hybrid_gradient.previous_grad

    for iteration in range(max_iterations):
        gradients = reliable_hybrid_gradient(
            current_params, topology_optimizer, U_target)
        new_params = mma.solve_convex_subproblem(current_params, gradients)
        new_loss, new_fidelity, new_ideal_fid, _ = \
            noise_aware_objective_function(new_params, topology_optimizer, U_target)
        delta_fidelity = new_fidelity - current_fidelity

        accept = new_loss < current_loss - 1e-6 or delta_fidelity > -1e-5

        if accept:
            current_params = new_params
            current_loss = new_loss
            current_fidelity = new_fidelity
            current_ideal_fid = new_ideal_fid
            if current_fidelity > best_fidelity:
                best_fidelity = current_fidelity
                best_params = current_params.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            if delta_fidelity > 0.01:
                mma.move_limit = min(0.6, mma.move_limit * 1.4)
            elif delta_fidelity > 0.001:
                mma.move_limit = min(0.5, mma.move_limit * 1.2)
            else:
                mma.move_limit = max(0.15, mma.move_limit * 0.9)
        else:
            mma.move_limit = max(0.1, mma.move_limit * 0.8)
            stagnation_count += 1

        mma.update_asymptotes(current_params, current_loss)

        if iteration % 5 == 0 or iteration == max_iterations - 1:
            print(f"Iter {iteration+1:2d}: NoisyFid={current_fidelity:.6f} "
                  f"(Δ={delta_fidelity:+.4f})")

        if stagnation_count > 15:
            print(f"Stopping: Stagnated at iteration {iteration+1}.")
            break

    print(f"Optimization Complete. Best Noisy Fidelity: {best_fidelity:.6f}")
    return best_params, best_fidelity


def generate_nearest_neighbor_pauli_terms(n_qubits):
    target_paulis = []
    for i in range(n_qubits - 1):
        for pair in ['XX', 'YY', 'ZZ']:
            p_str = ['I'] * n_qubits
            p_str[i] = pair[0]
            p_str[i+1] = pair[1]
            target_paulis.append("".join(p_str))
    return target_paulis


if __name__ == "__main__":
    print("=" * 60)
    print("8-Qubit MMA-UPTE w/ Correlated Noises")
    print("=" * 60)
    start_time = time.time()

    UPTE_PAULI_TERMS = [f"{p1}{p2}" for p1 in ['X', 'Y', 'Z']
                        for p2 in ['X', 'Y', 'Z']]
    UPTE_PAULI_TERMS = ["I" * (N_QUBITS - 2) + p for p in UPTE_PAULI_TERMS]

    topology_optimizer = TopologyAwareUPTE(N_QUBITS, UPTE_PAULI_TERMS)

    rng_target = np.random.default_rng()
    target_paulis = generate_nearest_neighbor_pauli_terms(N_QUBITS)
    coeffs = rng_target.uniform(low=-0.5, high=0.5, size=len(target_paulis))
    H_target = SparsePauliOp(target_paulis, coeffs=coeffs).to_matrix(sparse=False)
    t = 1.0
    U_target = la.expm(-1j * t * H_target)

    np.random.seed(42)
    initial_params_rho = np.random.uniform(low=0.1, high=0.4, size=N_TERMS)
    initial_params_tau = np.random.uniform(low=0.1, high=np.pi, size=N_TERMS)
    initial_params = np.concatenate([initial_params_rho, initial_params_tau])

    _, initial_noisy_fid, initial_ideal_fid, initial_rho = \
        noise_aware_objective_function(initial_params, topology_optimizer, U_target)
    psi_target = U_target @ np.eye(topology_optimizer.dim, dtype=complex)[:, 0]
    initial_vd_fid = virtual_distillation_enhancement(
        initial_rho, psi_target, VD_COPIES)

    print(f"Initial ideal fidelity: {initial_ideal_fid:.6f}")
    print(f"Initial noisy fidelity: {initial_noisy_fid:.6f}")
    print(f"Initial VD fidelity: {initial_vd_fid:.6f}\n")

    final_params, mma_fid = run_mma_optimization(
        initial_params, topology_optimizer, U_target, MAX_ITERATIONS)
    _, final_noisy_fid, final_ideal_fid, final_rho = \
        noise_aware_objective_function(final_params, topology_optimizer, U_target)
    final_vd_fid = virtual_distillation_enhancement(
        final_rho, psi_target, VD_COPIES)

    end_time = time.time()

    print(f"\nFinal VD Enhanced Fidelity (x{VD_COPIES} copies): {final_vd_fid:.6f}")
    print("=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)
    print(f"Target Problem Ideal Fidelity: {final_ideal_fid:.6f}")
    print(f"Initial Noisy Fidelity: {initial_noisy_fid:.6f}")
    print(f"Final Optimized Noisy Fidelity: {final_noisy_fid:.6f}")
    print(f"Final VD Enhanced Fidelity: {final_vd_fid:.6f}")
    print("-" * 60)
    print(f"Raw Fidelity Improvement: {final_noisy_fid - initial_noisy_fid:+.6f}")
    print(f"VD Enhancement: {final_vd_fid - final_noisy_fid:+.6f}")
    print(f"Total Improvement (VD): {final_vd_fid - initial_vd_fid:+.6f}")
    print(f"Total Runtime: {end_time - start_time:.2f} seconds")