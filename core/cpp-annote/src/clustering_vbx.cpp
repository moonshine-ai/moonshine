// SPDX-License-Identifier: MIT

#include "clustering_vbx.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

#include "filter_train.h"
#include "hungarian.h"
#include "parity_log.h"
#include "scipy_linkage.h"

namespace cppannote::clustering_vbx {
namespace {

void row_normalize(Eigen::MatrixXd& M) {
  for (int i = 0; i < M.rows(); ++i) {
    double n = M.row(i).norm();
    if (n > 1e-12) {
      M.row(i) /= n;
    }
  }
}

void cdist_cosine(const Eigen::MatrixXd& X, const Eigen::MatrixXd& C,
                  Eigen::MatrixXd& out) {
  out.resize(X.rows(), C.rows());
  for (int i = 0; i < X.rows(); ++i) {
    double xn = X.row(i).norm();
    if (xn < 1e-12) {
      xn = 1e-12;
    }
    for (int k = 0; k < C.rows(); ++k) {
      double cn = C.row(k).norm();
      if (cn < 1e-12) {
        cn = 1e-12;
      }
      const double cs = X.row(i).dot(C.row(k)) / (xn * cn);
      out(i, k) = 1.0 - cs;
    }
  }
}

std::vector<int> kmeans_fit_predict(const Eigen::MatrixXd& Xnorm, int k,
                                    int n_init, std::uint32_t seed) {
  const int n = static_cast<int>(Xnorm.rows());
  const int d = static_cast<int>(Xnorm.cols());
  if (n < k) {
    std::vector<int> r(static_cast<std::size_t>(n));
    std::iota(r.begin(), r.end(), 0);
    return r;
  }
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> uni(0, n - 1);
  double best_inertia = std::numeric_limits<double>::infinity();
  std::vector<int> best_labs(static_cast<std::size_t>(n), 0);
  for (int trial = 0; trial < n_init; ++trial) {
    Eigen::MatrixXd cent(k, d);
    for (int j = 0; j < k; ++j) {
      cent.row(j) = Xnorm.row(uni(rng));
    }
    std::vector<int> labels(static_cast<std::size_t>(n), 0);
    for (int it = 0; it < 100; ++it) {
      bool changed = false;
      for (int i = 0; i < n; ++i) {
        double best = 1e300;
        int bi = 0;
        for (int j = 0; j < k; ++j) {
          double dist = (Xnorm.row(i) - cent.row(j)).squaredNorm();
          if (dist < best) {
            best = dist;
            bi = j;
          }
        }
        if (labels[static_cast<std::size_t>(i)] != bi) {
          labels[static_cast<std::size_t>(i)] = bi;
          changed = true;
        }
      }
      cent.setZero();
      std::vector<int> cnt(static_cast<std::size_t>(k), 0);
      for (int i = 0; i < n; ++i) {
        const int b = labels[static_cast<std::size_t>(i)];
        cent.row(b) += Xnorm.row(i);
        cnt[static_cast<std::size_t>(b)] += 1;
      }
      for (int j = 0; j < k; ++j) {
        if (cnt[static_cast<std::size_t>(j)] > 0) {
          cent.row(j) /= static_cast<double>(cnt[static_cast<std::size_t>(j)]);
        }
      }
      if (!changed) {
        break;
      }
    }
    double inertia = 0.0;
    for (int i = 0; i < n; ++i) {
      const int b = labels[static_cast<std::size_t>(i)];
      inertia += (Xnorm.row(i) - cent.row(b)).squaredNorm();
    }
    if (inertia < best_inertia) {
      best_inertia = inertia;
      best_labs = labels;
    }
  }
  return best_labs;
}

Eigen::MatrixXd centroids_from_labels(const Eigen::MatrixXd& train,
                                      const std::vector<int>& lab, int k) {
  Eigen::MatrixXd cent(k, train.cols());
  cent.setZero();
  std::vector<int> cnt(static_cast<std::size_t>(k), 0);
  for (int i = 0; i < train.rows(); ++i) {
    const int b = lab[static_cast<std::size_t>(i)];
    cent.row(b) += train.row(i);
    cnt[static_cast<std::size_t>(b)] += 1;
  }
  for (int j = 0; j < k; ++j) {
    if (cnt[static_cast<std::size_t>(j)] > 0) {
      cent.row(j) /= static_cast<double>(cnt[static_cast<std::size_t>(j)]);
    }
  }
  return cent;
}

void hungarian_maximize(const Eigen::MatrixXd& score,
                        std::vector<int>& assign_row_to_col) {
  const int r = static_cast<int>(score.rows());
  const int c = static_cast<int>(score.cols());
  const int m = std::max(r, c);
  double mx = score.maxCoeff();
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      mx = std::max(mx, score(i, j));
    }
  }
  const double pad = mx + 1.0;
  std::vector<std::vector<double>> cost(
      static_cast<std::size_t>(m),
      std::vector<double>(static_cast<std::size_t>(m), pad));
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      cost[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
          mx - score(i, j);
    }
  }
  auto res = hungarian::min_cost_assignment(cost);
  assign_row_to_col.assign(static_cast<std::size_t>(r), -1);
  for (int i = 0; i < r; ++i) {
    const int j = res.second[static_cast<std::size_t>(i)];
    if (j < c && j >= 0) {
      assign_row_to_col[static_cast<std::size_t>(i)] = j;
    }
  }
}

}  // namespace

void vbx_clustering_hard(const plda_vbx::PldaModel& plda,
                         const VbxClusteringParams& pr, int num_chunks,
                         int num_frames, int num_speakers, int dim,
                         const float* embeddings, const float* binarized,
                         std::vector<std::int8_t>& hard_clusters) {
  hard_clusters.assign(static_cast<std::size_t>(num_chunks * num_speakers), 0);
  std::vector<int> c_idx;
  std::vector<int> s_idx;
  Eigen::MatrixXd train;
  filter_train::filter_embeddings_train(
      num_chunks, num_frames, num_speakers, dim, embeddings, binarized,
      pr.min_active_ratio, c_idx, s_idx, train);
  const int T = static_cast<int>(train.rows());
  if (T < 2) {
    if (parity::env_parity_level() >= 1) {
      std::ostringstream oss;
      oss << "vbxi skip: filtered_train_T=" << T << " chunks=" << num_chunks
          << " frames=" << num_frames << " speakers=" << num_speakers
          << " dim=" << dim;
      parity::log_light(oss.str());
    }
    return;
  }

  Eigen::MatrixXd train_n = train;
  row_normalize(train_n);
  std::vector<double> xflat(
      static_cast<std::size_t>(train_n.rows() * train_n.cols()));
  for (int i = 0; i < train_n.rows(); ++i) {
    for (int j = 0; j < train_n.cols(); ++j) {
      xflat[static_cast<std::size_t>(i * train_n.cols() + j)] = train_n(i, j);
    }
  }
  std::vector<double> pd;
  scipy_linkage::pdist_euclidean(xflat, T, dim, pd);
  std::vector<double> Z;
  scipy_linkage::linkage_centroid_naive(pd, T, Z);
  std::vector<int> fc;
  scipy_linkage::fcluster_distance(Z, T, pr.threshold, fc);
  std::vector<int> ahc;
  scipy_linkage::remap_labels_contiguous(fc, ahc);

  Eigen::MatrixXd fea = plda(train);
  Eigen::VectorXd Phi = plda.phi_between;
  if (Phi.size() < fea.cols()) {
    throw std::runtime_error("PLDA phi shorter than PLDA feature dimension");
  }
  if (Phi.size() > fea.cols()) {
    Phi.conservativeResize(fea.cols());
  }
  Eigen::MatrixXd q;
  Eigen::VectorXd sp_pi;
  int vbx_iters_done = 0;
  double vbx_last_elbo_delta = 0.0;
  plda_vbx::cluster_vbx(ahc, fea, Phi, pr.Fa, pr.Fb, pr.max_vbx_iters,
                        pr.init_smoothing, q, sp_pi, 1e-4, nullptr, nullptr,
                        &vbx_iters_done, &vbx_last_elbo_delta);
  std::vector<int> keep_cols;
  for (int j = 0; j < sp_pi.size(); ++j) {
    if (sp_pi(j) > 1e-7) {
      keep_cols.push_back(j);
    }
  }
  const int Kvb = static_cast<int>(keep_cols.size());
  if (Kvb <= 0) {
    throw std::runtime_error("VBx produced no speakers");
  }
  Eigen::MatrixXd W(T, Kvb);
  for (int t = 0; t < T; ++t) {
    for (int j = 0; j < Kvb; ++j) {
      W(t, j) = q(t, keep_cols[static_cast<std::size_t>(j)]);
    }
  }
  Eigen::VectorXd wsum = W.colwise().sum();
  Eigen::MatrixXd centroids(Kvb, dim);
  for (int k = 0; k < Kvb; ++k) {
    if (wsum(k) < 1e-12) {
      centroids.row(k).setZero();
    } else {
      centroids.row(k) = (W.col(k).transpose() * train) / wsum(k);
    }
  }
  int auto_k = Kvb;
  int nc = pr.num_clusters;
  if (auto_k < pr.min_clusters) {
    nc = pr.min_clusters;
  } else if (auto_k > pr.max_clusters) {
    nc = pr.max_clusters;
  }
  bool constrained = pr.constrained_assignment;
  if (nc > 0 && nc != auto_k) {
    constrained = false;
    const std::vector<int> km = kmeans_fit_predict(train_n, nc, 3, 42u);
    centroids = centroids_from_labels(train, km, nc);
    auto_k = nc;
  }

  Eigen::MatrixXd all(static_cast<int>(num_chunks * num_speakers), dim);
  for (int c = 0; c < num_chunks; ++c) {
    for (int s = 0; s < num_speakers; ++s) {
      for (int t = 0; t < dim; ++t) {
        all(c * num_speakers + s, t) =
            static_cast<double>(embeddings[((c * num_speakers) + s) * dim + t]);
      }
    }
  }
  Eigen::MatrixXd dist;
  if (pr.metric_is_cosine) {
    cdist_cosine(all, centroids, dist);
  } else {
    throw std::runtime_error("only cosine metric supported in cpp clustering");
  }
  Eigen::MatrixXd soft = (-dist.array() + 2.0).matrix();

  double finite_min = std::numeric_limits<double>::infinity();
  for (int i = 0; i < soft.rows(); ++i) {
    for (int j = 0; j < soft.cols(); ++j) {
      const double v = soft(i, j);
      if (std::isfinite(v)) {
        finite_min = std::min(finite_min, v);
      }
    }
  }
  if (!(finite_min < std::numeric_limits<double>::infinity())) {
    finite_min = 0.0;
  }
  const double const_score = finite_min - 1.0;

  for (int c = 0; c < num_chunks; ++c) {
    Eigen::MatrixXd blk = soft.middleRows(c * num_speakers, num_speakers);
    for (int s = 0; s < num_speakers; ++s) {
      float ssum = 0.f;
      for (int f = 0; f < num_frames; ++f) {
        ssum += binarized[((c * num_frames) + f) * num_speakers + s];
      }
      if (ssum == 0.f) {
        for (int k = 0; k < blk.cols(); ++k) {
          blk(s, k) = const_score;
        }
      }
    }
  }
  // Match ``np.nan_to_num(..., nan=np.nanmin(soft_clusters))`` before
  // ``linear_sum_assignment``.
  for (int i = 0; i < soft.rows(); ++i) {
    for (int j = 0; j < soft.cols(); ++j) {
      if (!std::isfinite(soft(i, j))) {
        soft(i, j) = finite_min;
      }
    }
  }

  for (int c = 0; c < num_chunks; ++c) {
    Eigen::MatrixXd blk = soft.middleRows(c * num_speakers, num_speakers);
    if (constrained) {
      std::vector<int> assign;
      hungarian_maximize(blk, assign);
      for (int s = 0; s < num_speakers; ++s) {
        const int j = assign[static_cast<std::size_t>(s)];
        hard_clusters[static_cast<std::size_t>(c * num_speakers + s)] =
            j >= 0 ? static_cast<std::int8_t>(j) : static_cast<std::int8_t>(-2);
      }
    } else {
      for (int s = 0; s < num_speakers; ++s) {
        Eigen::Index mj = 0;
        blk.row(s).maxCoeff(&mj);
        hard_clusters[static_cast<std::size_t>(c * num_speakers + s)] =
            static_cast<std::int8_t>(mj);
      }
    }
  }

  for (int c = 0; c < num_chunks; ++c) {
    for (int s = 0; s < num_speakers; ++s) {
      float ssum = 0.f;
      for (int f = 0; f < num_frames; ++f) {
        ssum += binarized[((c * num_frames) + f) * num_speakers + s];
      }
      if (ssum == 0.f) {
        hard_clusters[static_cast<std::size_t>(c * num_speakers + s)] = -2;
      }
    }
  }

  int max_hc = -1;
  for (std::int8_t v : hard_clusters) {
    if (v > max_hc) {
      max_hc = static_cast<int>(v);
    }
  }
  if (parity::env_parity_level() >= 1) {
    std::vector<float> fea_f(static_cast<size_t>(fea.rows() * fea.cols()));
    size_t k = 0;
    for (int i = 0; i < static_cast<int>(fea.rows()); ++i) {
      for (int j = 0; j < static_cast<int>(fea.cols()); ++j) {
        fea_f[k++] = static_cast<float>(fea(i, j));
      }
    }
    std::ostringstream oss;
    oss << "vbxi C=" << num_chunks << " F=" << num_frames
        << " S=" << num_speakers << " dim=" << dim << " T_train=" << T
        << " Kvb=" << Kvb << " max_hard_cluster=" << max_hc
        << " vbx_iters=" << vbx_iters_done
        << " vbx_last_elbo_delta=" << vbx_last_elbo_delta << " fea_fp="
        << parity::fingerprint_float32(fea_f.data(), fea_f.size());
    parity::log_light(oss.str());
  }

  // Note: the upstream heavy-dump NPZ block (cnpy::npz_save of VBx
  // intermediates when PYANNOTE_CPP_PARITY=2) has been removed in this
  // vendored copy to avoid a cnpy/zlib dependency.
}

}  // namespace cppannote::clustering_vbx
