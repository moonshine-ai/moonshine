// SPDX-License-Identifier: MIT

#include "plda_vbx.h"

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace cppannote::plda_vbx {

using RowMatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Hugging Face ``cppannote/speaker-diarization-community-1`` ``plda.npz``
// (128×128 ``tr``): Eigen's generalized eigenvectors can differ from SciPy
// ``eigh`` by **per-column** signs. Fingerprint ``tr``, then align each
// ``evecs`` column using SciPy's ``wccn[0, j]`` (ascending eigenvalues).
constexpr double kCommunity1PldaTrSum = -6.923248683599031;
constexpr double kCommunity1PldaTrFro = 19.227821386182885;
constexpr std::array<double, 128> kCommunity1SciPyWccnRow0 = {
    -0.0033954634318117635,  -0.0029756524923024875,  0.00098840931467854124,
    -0.0079384162014615074,  0.00098356796202681875,  -0.0036290794434025663,
    -0.00030141208364042979, -0.0024467474429338753,  0.0040381169727581575,
    -0.0007865637323122306,  0.0022480342802435604,   0.0050555737961515933,
    -0.00091617892245531973, -0.010753637989592977,   -0.0020271768736715582,
    -0.0022924798704292864,  0.0015846338377371611,   0.0045445845924315742,
    0.0014891678977383166,   0.0077624438657284247,   -0.0043484726407827081,
    -0.0021720767084415522,  0.0028298433428699901,   0.0092858167803416131,
    -0.0041847610563746657,  -0.0058234110736610709,  -0.0054050177852315454,
    -0.0025555332871907586,  -0.0010368980332107504,  0.0036063242244259476,
    0.0029195456623689877,   0.0047566779880388061,   -0.00043818358767745913,
    0.0026005891580769412,   -0.0042518910681148426,  0.0037089785529460565,
    0.00018765061877369006,  -0.0040714427251499763,  -0.00042881453412262413,
    -0.0032374883072830416,  0.0028887184890476576,   0.0084087210301520901,
    -0.0056032811276475105,  0.0005954664065670977,   -0.0014218646978600203,
    -0.0021297232470902389,  0.0013622634949925724,   -0.0037050859144387564,
    0.00022119907463957742,  4.8816914357830134e-05,  -0.0026697170868004621,
    -0.0024804763174611898,  0.0067079078051495008,   0.0049029522334883948,
    -0.0055603663812243478,  -0.0092410465880061656,  0.0075024496328911601,
    -0.0064333638913648484,  -0.00030253541583077481, 0.0061705966440731407,
    0.011429133105439431,    0.0035637688986348368,   -0.00083770017882024716,
    0.0042841660147541968,   -0.0025900114669715878,  0.019676716662849932,
    -0.005505467818164491,   0.0078566027592071389,   -0.0015964212711396292,
    -0.0022094533862385469,  0.00081050006761742026,  0.011967530773278447,
    -0.0091485812610701822,  -0.013545565283392224,   0.0059962793380347566,
    -0.00050616265505980837, -0.0014287369735356528,  -0.0018383943632644816,
    -0.024706679649297306,   -0.0093397779047840589,  -0.0073755119909759432,
    0.024668093027618642,    -0.021818724576921564,   -0.0026632287796624012,
    -0.0031996114803744878,  -0.0080986059325932856,  0.0031259338722462362,
    -0.0081059322731651031,  -0.011831042554358974,   -0.0056124230197808576,
    -0.0025691177759062079,  0.004099303667885355,    0.0071985949458587906,
    -0.0019588953922858691,  0.010048532805130536,    0.015314766927670406,
    -0.014542371019619816,   -0.019422140000592756,   -0.022701757964085537,
    -0.013854821508680655,   -0.0016036791316736477,  0.000738557765462525,
    0.016314389406842447,    0.013734955653246256,    -0.017294124309366356,
    -0.014823992515892856,   -0.012784458698388151,   0.00073472354302963467,
    -0.016338025368738858,   0.0088320056336558712,   0.032839420138977532,
    0.0068695867299960137,   -0.0089083845103694482,  0.034321260499731755,
    -0.010265220332353527,   0.039049705571901176,    0.02971560609946081,
    0.055727067641499144,    0.022429930999621053,    -0.14573989196451809,
    0.074633530551557428,    0.077786122261532684,    0.0043146580431851853,
    -0.13560074327292224,    -0.03428022485556604,    -0.083532431339810448,
    -0.019530693733352034,   -1.6368925072961353,
};

void align_eigen_evecs_to_scipy_wccn_row0_for_community1_plda(
    const RowMatrixXd& tr_file, Eigen::MatrixXd& evecs) {
  const int d = static_cast<int>(evecs.rows());
  if (d != 128 || evecs.cols() != 128) {
    return;
  }
  if (std::abs(tr_file.sum() - kCommunity1PldaTrSum) > 1.0e-6 ||
      std::abs(tr_file.norm() - kCommunity1PldaTrFro) > 1.0e-6) {
    return;
  }
  for (int j = 0; j < d; ++j) {
    const double a = evecs(0, j);
    const double r = kCommunity1SciPyWccnRow0[static_cast<std::size_t>(j)];
    if (a * r < 0.0) {
      evecs.col(j) *= -1.0;
    }
  }
}

namespace {

void row_l2_normalize(Eigen::MatrixXd& M) {
  for (int i = 0; i < M.rows(); ++i) {
    double n = M.row(i).norm();
    if (n > 1e-12) {
      M.row(i) /= n;
    }
  }
}

void logsumexp_rowwise(const Eigen::MatrixXd& M, Eigen::VectorXd& lse,
                       Eigen::MatrixXd& centered) {
  const int r = static_cast<int>(M.rows());
  const int c = static_cast<int>(M.cols());
  lse.resize(r);
  centered.resize(r, c);
  for (int i = 0; i < r; ++i) {
    const double m = M.row(i).maxCoeff();
    double s = 0.0;
    for (int j = 0; j < c; ++j) {
      s += std::exp(M(i, j) - m);
    }
    lse(i) = m + std::log(std::max(s, 1e-300));
    for (int j = 0; j < c; ++j) {
      centered(i, j) = M(i, j) - lse(i);
    }
  }
}

}  // namespace

void PldaModel::load_from_arrays(const double* mean1_p, int n_mean1,
                                 const float* mean2_p, int n_mean2,
                                 const float* lda_p, int lda_rows, int lda_cols,
                                 const double* mu_p, int n_mu,
                                 const double* tr_p, int tr_side,
                                 const double* psi_p, int n_psi, int lda_dim) {
  lda_dimension = lda_dim;
  mean1 = Eigen::Map<const Eigen::VectorXd>(mean1_p, n_mean1);
  mean2 = Eigen::Map<const Eigen::VectorXf>(mean2_p, n_mean2).cast<double>();
  lda = Eigen::Map<const RowMatrixXf>(lda_p, lda_rows, lda_cols).cast<double>();
  plda_mu = Eigen::Map<const Eigen::VectorXd>(mu_p, n_mu);
  Eigen::MatrixXd tr_file =
      Eigen::Map<const RowMatrixXd>(tr_p, tr_side, tr_side);
  Eigen::VectorXd psi_file = Eigen::Map<const Eigen::VectorXd>(psi_p, n_psi);
  const Eigen::MatrixXd Wmat = (tr_file.transpose() * tr_file).inverse();
  Eigen::MatrixXd tr_t = tr_file.transpose();
  for (int j = 0; j < tr_t.cols(); ++j) {
    tr_t.col(j) /= std::max(psi_file(j), 1e-12);
  }
  const Eigen::MatrixXd Bmat = (tr_t * tr_file).inverse();
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(Bmat, Wmat);
  if (ges.info() != Eigen::Success) {
    throw std::runtime_error(
        "generalized eigen decomposition failed in vbx_setup");
  }
  Eigen::VectorXd evals = ges.eigenvalues();
  Eigen::MatrixXd evecs = ges.eigenvectors();
  align_eigen_evecs_to_scipy_wccn_row0_for_community1_plda(tr_file, evecs);
  const int d = static_cast<int>(evals.size());
  phi_between.resize(std::min(d, lda_dimension));
  for (int i = 0; i < phi_between.size(); ++i) {
    phi_between(i) = evals(d - 1 - i);
  }
  const Eigen::MatrixXd Et = evecs.transpose();
  plda_tr.resize(d, d);
  for (int i = 0; i < d; ++i) {
    plda_tr.row(i) = Et.row(d - 1 - i);
  }
}

// The upstream file-based PldaModel::load() (cnpy NPZ loading) is removed in
// this vendored copy; see plda_vbx.h.
#if 0
void PldaModel::load(const std::string& xvec_transform_npz,
                     const std::string& plda_npz, int lda_dim) {
  lda_dimension = lda_dim;
  {
    cnpy::npz_t x = cnpy::npz_load(xvec_transform_npz);
    if (!x.count("mean1") || !x.count("mean2") || !x.count("lda")) {
      throw std::runtime_error("xvec_transform.npz missing mean1/mean2/lda");
    }
    const cnpy::NpyArray& m1 = x["mean1"];
    const cnpy::NpyArray& m2 = x["mean2"];
    const cnpy::NpyArray& ld = x["lda"];
    if (m1.word_size == sizeof(double)) {
      mean1 = Eigen::Map<const Eigen::VectorXd>(
          m1.data<double>(), static_cast<Eigen::Index>(m1.shape[0]));
    } else if (m1.word_size == sizeof(float)) {
      mean1 = Eigen::Map<const Eigen::VectorXf>(
                  m1.data<float>(), static_cast<Eigen::Index>(m1.shape[0]))
                  .cast<double>();
    } else {
      throw std::runtime_error(
          "mean1 must be float32 or float64 in xvec_transform.npz");
    }
    if (m2.word_size == sizeof(double)) {
      mean2 = Eigen::Map<const Eigen::VectorXd>(
          m2.data<double>(), static_cast<Eigen::Index>(m2.shape[0]));
    } else if (m2.word_size == sizeof(float)) {
      mean2 = Eigen::Map<const Eigen::VectorXf>(
                  m2.data<float>(), static_cast<Eigen::Index>(m2.shape[0]))
                  .cast<double>();
    } else {
      throw std::runtime_error(
          "mean2 must be float32 or float64 in xvec_transform.npz");
    }
    if (ld.shape.size() != 2) {
      throw std::runtime_error("lda must be 2-D in xvec_transform.npz");
    }
    const int r = static_cast<int>(ld.shape[0]);
    const int c = static_cast<int>(ld.shape[1]);
    if (ld.word_size == sizeof(double)) {
      lda = Eigen::Map<const RowMatrixXd>(ld.data<double>(), r, c);
    } else if (ld.word_size == sizeof(float)) {
      lda =
          Eigen::Map<const RowMatrixXf>(ld.data<float>(), r, c).cast<double>();
    } else {
      throw std::runtime_error(
          "lda must be float32 or float64 in xvec_transform.npz");
    }
  }
  Eigen::MatrixXd tr_file;
  Eigen::VectorXd psi_file;
  {
    cnpy::npz_t p = cnpy::npz_load(plda_npz);
    if (!p.count("mu") || !p.count("tr") || !p.count("psi")) {
      throw std::runtime_error("plda.npz missing mu/tr/psi");
    }
    const cnpy::NpyArray& mu = p["mu"];
    const cnpy::NpyArray& tr = p["tr"];
    const cnpy::NpyArray& ps = p["psi"];
    plda_mu = Eigen::Map<const Eigen::VectorXd>(
        mu.data<double>(), static_cast<Eigen::Index>(mu.shape[0]));
    if (tr.shape.size() != 2 || tr.shape[0] != tr.shape[1]) {
      throw std::runtime_error("plda tr must be square 2-D");
    }
    const int d = static_cast<int>(tr.shape[0]);
    tr_file = Eigen::Map<const RowMatrixXd>(tr.data<double>(), d, d);
    psi_file = Eigen::Map<const Eigen::VectorXd>(
        ps.data<double>(), static_cast<Eigen::Index>(ps.shape[0]));
  }

  const Eigen::MatrixXd Wmat = (tr_file.transpose() * tr_file).inverse();
  Eigen::MatrixXd tr_t = tr_file.transpose();
  for (int j = 0; j < tr_t.cols(); ++j) {
    tr_t.col(j) /= std::max(psi_file(j), 1e-12);
  }
  const Eigen::MatrixXd Bmat = (tr_t * tr_file).inverse();
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(Bmat, Wmat);
  if (ges.info() != Eigen::Success) {
    throw std::runtime_error(
        "generalized eigen decomposition failed in vbx_setup");
  }
  Eigen::VectorXd evals = ges.eigenvalues();
  Eigen::MatrixXd evecs = ges.eigenvectors();
  align_eigen_evecs_to_scipy_wccn_row0_for_community1_plda(tr_file, evecs);
  const int d = static_cast<int>(evals.size());
  phi_between.resize(std::min(d, lda_dimension));
  for (int i = 0; i < phi_between.size(); ++i) {
    phi_between(i) = evals(d - 1 - i);
  }
  const Eigen::MatrixXd Et = evecs.transpose();
  plda_tr.resize(d, d);
  for (int i = 0; i < d; ++i) {
    plda_tr.row(i) = Et.row(d - 1 - i);
  }
}
#endif  // 0 (file-based load removed)

Eigen::MatrixXd PldaModel::xvec_tf(const Eigen::MatrixXd& embeddings) const {
  const int emb_dim = static_cast<int>(mean1.size());
  if (embeddings.cols() != emb_dim) {
    throw std::runtime_error("embedding dimension mismatch for xvec_tf");
  }
  Eigen::MatrixXd xm = embeddings.rowwise() - mean1.transpose();
  row_l2_normalize(xm);
  xm *= std::sqrt(static_cast<double>(lda.rows()));
  Eigen::MatrixXd y = xm * lda;
  y.rowwise() -= mean2.transpose();
  row_l2_normalize(y);
  y *= std::sqrt(static_cast<double>(lda.cols()));
  return y;
}

Eigen::MatrixXd PldaModel::plda_tf(const Eigen::MatrixXd& x0,
                                   int lda_dim) const {
  Eigen::MatrixXd c = x0.rowwise() - plda_mu.transpose();
  Eigen::MatrixXd out = c * plda_tr.transpose();
  if (lda_dim < out.cols()) {
    out = out.leftCols(lda_dim);
  }
  return out;
}

Eigen::MatrixXd PldaModel::operator()(const Eigen::MatrixXd& embeddings) const {
  return plda_tf(xvec_tf(embeddings), lda_dimension);
}

void softmax_rows(const Eigen::MatrixXd& logits, Eigen::MatrixXd& out) {
  out.resize(logits.rows(), logits.cols());
  for (int i = 0; i < logits.rows(); ++i) {
    const double m = logits.row(i).maxCoeff();
    double s = 0.0;
    for (int j = 0; j < logits.cols(); ++j) {
      out(i, j) = std::exp(logits(i, j) - m);
      s += out(i, j);
    }
    out.row(i) /= std::max(s, 1e-300);
  }
}

void cluster_vbx(const std::vector<int>& ahc_init, const Eigen::MatrixXd& fea,
                 const Eigen::VectorXd& Phi, double Fa, double Fb,
                 int max_iters, double init_smoothing, Eigen::MatrixXd& gamma,
                 Eigen::VectorXd& pi, double elbo_epsilon,
                 std::vector<Eigen::MatrixXd>* trace_gamma,
                 std::vector<Eigen::VectorXd>* trace_pi, int* out_vbx_iters,
                 double* out_last_elbo_delta) {
  const int T = static_cast<int>(fea.rows());
  const int D = static_cast<int>(fea.cols());
  if (static_cast<int>(Phi.size()) != D) {
    throw std::runtime_error("Phi dimension mismatch");
  }
  if (static_cast<int>(ahc_init.size()) != T) {
    throw std::runtime_error("ahc_init length mismatch");
  }
  int k_max = 0;
  for (int v : ahc_init) {
    k_max = std::max(k_max, v);
  }
  const int S = k_max + 1;
  Eigen::MatrixXd qinit(T, S);
  qinit.setZero();
  for (int t = 0; t < T; ++t) {
    qinit(t, ahc_init[static_cast<std::size_t>(t)]) = 1.0;
  }
  if (init_smoothing >= 0.0) {
    qinit *= init_smoothing;
    softmax_rows(qinit, gamma);
  } else {
    gamma = qinit;
  }

  pi.resize(S);
  pi.setConstant(1.0 / static_cast<double>(S));

  Eigen::MatrixXd alpha;
  Eigen::MatrixXd invL;

  Eigen::VectorXd G(T);
  for (int t = 0; t < T; ++t) {
    double sq = 0.0;
    for (int j = 0; j < D; ++j) {
      sq += fea(t, j) * fea(t, j);
    }
    G(t) = -0.5 * (sq + static_cast<double>(D) *
                            std::log(2.0 * std::numbers::pi));
  }
  Eigen::VectorXd V = Phi.array().sqrt();
  Eigen::MatrixXd rho(T, D);
  for (int t = 0; t < T; ++t) {
    for (int j = 0; j < D; ++j) {
      rho(t, j) = fea(t, j) * V(j);
    }
  }

  Eigen::MatrixXd log_p_(T, S);
  Eigen::VectorXd log_p_x;
  Eigen::MatrixXd centered;
  double ELBO = 0.0;
  double ELBO_prev = 0.0;
  double last_step_delta = 0.0;
  int completed_iters = 0;

  for (int it = 0; it < max_iters; ++it) {
    if (it > 0 || alpha.size() == 0 || invL.size() == 0) {
      Eigen::VectorXd gsum = gamma.colwise().sum();
      invL.resize(S, D);
      for (int s = 0; s < S; ++s) {
        for (int j = 0; j < D; ++j) {
          invL(s, j) = 1.0 / (1.0 + Fa / Fb * gsum(s) * Phi(j));
        }
      }
      const Eigen::MatrixXd gr = gamma.transpose() * rho;
      alpha = ((Fa / Fb) * invL.array() * gr.array()).matrix();
    }

    Eigen::VectorXd term_s(S);
    for (int s = 0; s < S; ++s) {
      double acc = 0.0;
      for (int j = 0; j < D; ++j) {
        acc += (invL(s, j) + alpha(s, j) * alpha(s, j)) * Phi(j);
      }
      term_s(s) = 0.5 * acc;
    }

    for (int t = 0; t < T; ++t) {
      for (int s = 0; s < S; ++s) {
        double dot = 0.0;
        for (int j = 0; j < D; ++j) {
          dot += rho(t, j) * alpha(s, j);
        }
        log_p_(t, s) = Fa * (dot - term_s(s) + G(t));
      }
    }

    const double eps = 1e-8;
    Eigen::MatrixXd lpi(S, 1);
    for (int s = 0; s < S; ++s) {
      lpi(s) = std::log(pi(s) + eps);
    }
    Eigen::MatrixXd weighted = log_p_;
    for (int t = 0; t < T; ++t) {
      for (int s = 0; s < S; ++s) {
        weighted(t, s) += lpi(s);
      }
    }
    logsumexp_rowwise(weighted, log_p_x, centered);
    double log_pX_ = log_p_x.sum();

    gamma = centered.array().exp().matrix();
    pi = gamma.colwise().sum().transpose();
    pi /= pi.sum();

    if (trace_gamma != nullptr) {
      trace_gamma->push_back(gamma);
    }
    if (trace_pi != nullptr) {
      trace_pi->push_back(pi);
    }

    double reg = 0.0;
    for (int s = 0; s < S; ++s) {
      for (int j = 0; j < D; ++j) {
        const double il = invL(s, j);
        reg += std::log(il) - il - alpha(s, j) * alpha(s, j) + 1.0;
      }
    }
    ELBO = log_pX_ + Fb * 0.5 * reg;
    if (it > 0) {
      last_step_delta = ELBO - ELBO_prev;
    }
    if (elbo_epsilon >= 0.0 && it > 0 && ELBO - ELBO_prev < elbo_epsilon) {
      if (ELBO - ELBO_prev < 0.0) {
        // match Python warning behavior (no stderr in library)
      }
      completed_iters = it + 1;
      ELBO_prev = ELBO;
      break;
    }
    ELBO_prev = ELBO;
    completed_iters = it + 1;
  }
  if (out_vbx_iters != nullptr) {
    *out_vbx_iters = completed_iters;
  }
  if (out_last_elbo_delta != nullptr) {
    *out_last_elbo_delta = last_step_delta;
  }
}

}  // namespace cppannote::plda_vbx
