#ifndef PRORATIO_H
#define PRORATIO_H

#include "PROconfig.h"
#include <Eigen/Eigen>
#include <string>
#include <vector>

namespace PROfit {

    // numerator channel over denominator channel, each named
    // <mode>_<detector>_<channel>; same-channel/two-detector and two-channel/
    // one-detector are the same operation on the collapsed vector.
    struct RatioSpec {
        std::string num_name;
        std::string den_name;
    };

    // maps a collapsed spectrum onto a vector of channel ratios, and builds the
    // ratio-space covariance from an event-space one by linear propagation,
    // C_R = J C J^T with J = d(ratio)/d(collapsed bin).
    class PROratio {
        public:
            PROratio() = default;
            // Throws std::runtime_error on unresolvable names or incompatible binning.
            PROratio(const PROconfig &config, const std::vector<RatioSpec> &specs, int var_index);

            bool   Empty()  const { return m_slots.empty(); }
            size_t NBins()  const { return m_nbins_total; }
            const std::vector<std::string> &BinLabels() const { return m_labels; }

            Eigen::VectorXf Apply(const Eigen::VectorXf &collapsed) const;
            Eigen::VectorXf Numerators(const Eigen::VectorXf &collapsed) const;
            Eigen::VectorXf Denominators(const Eigen::VectorXf &collapsed) const;

            Eigen::MatrixXf Jacobian(const Eigen::VectorXf &collapsed) const;

            Eigen::MatrixXf PropagateCovariance(const Eigen::VectorXf &collapsed,
                                                const Eigen::MatrixXf &C) const;

            void SetValidBins(const Eigen::VectorXf &data_coll,
                              const Eigen::VectorXf &cv_coll,
                              float min_denominator = 1e-3f);
            const std::vector<Eigen::Index> &ValidBins() const { return m_valid; }

        private:
            struct Slot {
                size_t off_num, ny_num;
                size_t off_den, ny_den;
                size_t nx;
                size_t out_start;
            };
            struct Term { Eigen::Index idx; float c; };

            std::vector<Term> RowTerms(const Eigen::VectorXf &collapsed, size_t r) const;

            std::vector<Slot> m_slots;
            std::vector<std::string> m_labels;
            std::vector<Eigen::Index> m_valid;
            size_t m_nbins_total = 0;
            size_t m_ncollapsed  = 0;
    };

    // covariance, built like PROchi
    Eigen::MatrixXf BuildCollapsedSystCovariance(const PROconfig &config,
                                                 const Eigen::VectorXf &spec_full,
                                                 const Eigen::MatrixXf &frac_cov);

    // chi2 on the ratio observable. M is the full covariance, stat
    // included, so whichever statistical treatment the calling metric uses
    // (Neyman, Pearson, CNP) is carried 
    float PROratioChi2(const PROratio &rmap,
                       const Eigen::VectorXf &pred_coll,
                       const Eigen::VectorXf &data_coll,
                       const Eigen::MatrixXf &M);
}
#endif