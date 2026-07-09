// SPDX-License-Identifier: MIT
// PLDA transforms (``vbx_setup``) + ``VBx`` / ``cluster_vbx`` (``vbx.py``).

#ifndef PLDA_VBX_H_
#define PLDA_VBX_H_

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace cppannote::plda_vbx {

using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct PldaModel {
  int lda_dimension = 128;
  Eigen::VectorXd mean1;
  Eigen::VectorXd mean2;
  RowMatrixXd lda;  // NumPy row-major ``(embedding_dim, lda_out)`` from
                    // ``xvec_transform.npz``.
  Eigen::VectorXd plda_mu;
  Eigen::MatrixXd plda_tr;
  /// Between-class diagonal in PLDA latent space (``PLDA.phi`` / ``_plda_psi``
  /// head).
  Eigen::VectorXd phi_between;

  // Note: the upstream file-based ``load(xvec_transform_npz, plda_npz)`` has
  // been removed in this vendored copy to avoid a cnpy/zlib dependency; only
  // the compiled-in ``load_from_arrays`` path is supported.

  /// Load from raw NumPy-export tensors (same layout as HF
  /// ``xvec_transform.npz`` / ``plda.npz``).
  void load_from_arrays(const double* mean1, int n_mean1, const float* mean2,
                        int n_mean2, const float* lda, int lda_rows,
                        int lda_cols, const double* mu, int n_mu,
                        const double* tr, int tr_side, const double* psi,
                        int n_psi, int lda_dim);
  Eigen::MatrixXd xvec_tf(const Eigen::MatrixXd& embeddings) const;
  Eigen::MatrixXd plda_tf(const Eigen::MatrixXd& x0, int lda_dim) const;
  Eigen::MatrixXd operator()(const Eigen::MatrixXd& embeddings) const;
};

void softmax_rows(const Eigen::MatrixXd& logits, Eigen::MatrixXd& out);

/// ``cluster_vbx`` from ``vbx.py`` (``return_model=True`` path).
/// If ``elbo_epsilon < 0``, ELBO early stopping is disabled (useful for
/// deterministic parity tests). When non-null, ``trace_gamma`` / ``trace_pi``
/// receive a snapshot after each iteration (post gamma/pi update).
/// ``out_vbx_iters`` / ``out_last_elbo_delta`` (optional) support
/// ``PYANNOTE_CPP_PARITY`` light logging.
void cluster_vbx(const std::vector<int>& ahc_init, const Eigen::MatrixXd& fea,
                 const Eigen::VectorXd& Phi, double Fa, double Fb,
                 int max_iters, double init_smoothing, Eigen::MatrixXd& gamma,
                 Eigen::VectorXd& pi, double elbo_epsilon = 1e-4,
                 std::vector<Eigen::MatrixXd>* trace_gamma = nullptr,
                 std::vector<Eigen::VectorXd>* trace_pi = nullptr,
                 int* out_vbx_iters = nullptr,
                 double* out_last_elbo_delta = nullptr);

}  // namespace cppannote::plda_vbx

#endif  // PLDA_VBX_H_
