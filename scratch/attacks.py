# attacks.py
import numpy as np
from qutip import Qobj, basis

# helpers for reconstructing Qobj from state vector
def state_from_array(arr):
    return Qobj(arr.reshape((-1,1)))

# Honest measurement: measure first qubit in {|+_delta>, |-_delta>} basis
def honest_measurement(state_array, delta):
    """
    state_array: numpy vector for n-qubit state (column or flat)
    delta: measurement angle sent by Alice (radians)
    returns: (outcome, post_state_array)
    """
    from qutip import basis, tensor
    state = state_from_array(state_array)
    # single-qubit basis vectors
    plus = (basis(2,0) + np.exp(1j*delta) * basis(2,1)).unit()
    minus = (basis(2,0) - np.exp(1j*delta) * basis(2,1)).unit()
    # projectors on first qubit
    dim = int(np.log2(state.shape[0]))
    if dim == 1:
        P_plus = plus * plus.dag()
        P_minus = minus * minus.dag()
    else:
        from qutip import tensor, qeye
        P_plus = tensor(plus*plus.dag(), qeye(2**(dim-1)))
        P_minus = tensor(minus*minus.dag(), qeye(2**(dim-1)))
    p_plus = (state.dag() * P_plus * state).tr().real
    # numerical safety
    p_plus = float(np.clip(p_plus, 0.0, 1.0))
    outcome = np.random.choice([0,1], p=[p_plus, 1-p_plus])
    meas_proj = P_plus if outcome == 0 else P_minus
    post = (meas_proj * state).unit()
    return int(outcome), np.array(post.full()).reshape(-1)

# Tomography attack: estimate the Bloch phase (theta) of single-qubit |+_theta>
# Assumes Alice sends multiple identical single-qubit copies (copies >= 3 recommended).
def tomography_attack(state_array_list):
    """
    state_array_list: list of numpy arrays, each a single-qubit state vector
    returns: (estimated_theta, fake_outcome, post_state_array)
    Note: consumes the copies (simulated).
    """
    from qutip import Qobj, sigmax, sigmay, sigmaz
    # Build density matrix estimate from measurement statistics on X, Y, Z
    # For simplicity: for each axis, measure all copies allocated
    copies = len(state_array_list)
    if copies < 3:
        # fallback: try to estimate from the one copy directly (non-physical in general)
        vec = state_array_list[0]
        psi = Qobj(vec.reshape((-1,1)))
        # compute Bloch expectation
        sx = float((psi.dag() * sigmax() * psi).tr().real)
        sy = float((psi.dag() * sigmay() * psi).tr().real)
        # estimate theta from X and Y on |+_theta>: ⟨X⟩ = cos(theta), ⟨Y⟩ = sin(theta)
        est_theta = np.arctan2(sy, sx) % (2*np.pi)
        # we must still return a post_state; we'll return the original collapsed state as-is
        return est_theta, None, vec
    # Split copies into 3 groups for X, Y, Z
    nx = copies // 3
    ny = copies // 3
    nz = copies - nx - ny
    def measure_in_basis(vecs, axis):
        # perform projective measurements on the provided vectors and return expectation
        from qutip import sigmax, sigmay, sigmaz
        obs = {'X': sigmax(), 'Y': sigmay(), 'Z': sigmaz()}[axis]
        vals = []
        for v in vecs:
            psi = Qobj(v.reshape((-1,1)))
            vals.append(float((psi.dag() * obs * psi).tr().real))
        return np.mean(vals)
    est_x = measure_in_basis(state_array_list[:nx], 'X') if nx>0 else 0.0
    est_y = measure_in_basis(state_array_list[nx:nx+ny], 'Y') if ny>0 else 0.0
    # For |+_theta> we expect <Z> = 0, so ignore z
    est_theta = np.arctan2(est_y, est_x) % (2*np.pi)
    # The attacker may choose an outcome consistent with their estimate (e.g., simulate measuring in delta basis)
    fake_outcome = None
    # No well-defined post-state if he used many copies; return zeros to indicate "consumed".
    post_state = np.zeros_like(state_array_list[0])
    return est_theta, fake_outcome, post_state

# Deviation attack: Bob measures in a different basis (e.g., adds bias) and optionally lies
def deviation_attack(state_array, delta, bias=0.2, lie_probability=0.3):
    """
    Bob measures in basis rotated by 'bias' relative to delta.
    He then with probability lie_probability flips the outcome before reporting.
    """
    biased_delta = delta + bias
    outcome, post = honest_measurement(state_array, biased_delta)
    # Possibly flip outcome to simulate lying
    if np.random.rand() < lie_probability:
        outcome = 1 - outcome
    return int(outcome), post
