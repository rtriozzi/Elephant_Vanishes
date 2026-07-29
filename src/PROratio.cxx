#include "PROratio.h"
#include "PROtocall.h"
#include "PROlog.h"
#include <stdexcept>

namespace PROfit {

    namespace {
        // <mode>_<detector>_<channel>
        std::vector<std::string> ChannelNames(const PROconfig &config) {
            std::vector<std::string> names;
            for(size_t mode = 0; mode < config.m_num_modes; ++mode)
                for(size_t det = 0; det < config.m_num_detectors; ++det)
                    for(size_t ch = 0; ch < config.m_num_channels; ++ch)
                        names.push_back(config.m_mode_names[mode] + "_" +
                                        config.m_detector_names[det] + "_" +
                                        config.m_channel_names[ch]);
            return names;
        }

        size_t ResolveChannel(const std::vector<std::string> &names, const std::string &pattern) {
            for(size_t i = 0; i < names.size(); ++i)
                if(names[i] == pattern) return i;
            size_t found = names.size();
            for(size_t i = 0; i < names.size(); ++i) {
                if(names[i].find(pattern) == std::string::npos) continue;
                if(found != names.size())
                    throw std::runtime_error("--fit-ratio channel '" + pattern + "' is ambiguous");
                found = i;
            }
            if(found == names.size())
                throw std::runtime_error("--fit-ratio channel '" + pattern + "' matched nothing");
            return found;
        }

        bool SameEdges(const std::vector<float> &a, const std::vector<float> &b) {
            if(a.size() != b.size()) return false;
            for(size_t i = 0; i < a.size(); ++i) {
                const float sc = std::max(1.0f, std::max(std::fabs(a[i]), std::fabs(b[i])));
                if(std::fabs(a[i] - b[i]) > 1e-4f*sc) return false;
            }
            return true;
        }
    }

    PROratio::PROratio(const PROconfig &config, const std::vector<RatioSpec> &specs, int var_index) {
        const std::vector<std::string> names = ChannelNames(config);
        m_ncollapsed = config.m_num_variable_bins_total_collapsed[var_index];

        for(const RatioSpec &rs : specs) {
            const size_t g_num = ResolveChannel(names, rs.num_name);
            const size_t g_den = ResolveChannel(names, rs.den_name);
            if(g_num == g_den)
                throw std::runtime_error("--fit-ratio numerator and denominator are the same channel");

            const size_t ch_num = g_num % config.m_num_channels;
            const size_t ch_den = g_den % config.m_num_channels;

            const size_t nx_n = config.m_channel_variable_bins[ch_num][var_index].NBinsAlong(0);
            const size_t nx_d = config.m_channel_variable_bins[ch_den][var_index].NBinsAlong(0);
            const std::vector<float> e_n = config.m_channel_variable_bins[ch_num][var_index].Edges(0);
            const std::vector<float> e_d = config.m_channel_variable_bins[ch_den][var_index].Edges(0);
            if(nx_n != nx_d || !SameEdges(e_n, e_d))
                throw std::runtime_error("--fit-ratio requires identical x binning in " +
                                         names[g_num] + " and " + names[g_den]);

            Slot s;
            s.nx      = nx_n;
            s.off_num = config.GetCollapsedGlobalVariableBinStart(g_num, var_index);
            s.off_den = config.GetCollapsedGlobalVariableBinStart(g_den, var_index);
            s.ny_num  = config.m_channel_variable_dims[ch_num][var_index] == 2
                      ? config.m_channel_variable_bins[ch_num][var_index].NBinsAlong(1) : 1;
            s.ny_den  = config.m_channel_variable_dims[ch_den][var_index] == 2
                      ? config.m_channel_variable_bins[ch_den][var_index].NBinsAlong(1) : 1;
            s.out_start = m_nbins_total;
            m_slots.push_back(s);

            for(size_t bx = 0; bx < s.nx; ++bx)
                m_labels.push_back(names[g_num] + "/" + names[g_den] + "_bin" + std::to_string(bx));
            m_nbins_total += s.nx;

            log<LOG_INFO>(L"%1% || --fit-ratio: %2% / %3%, %4% bins, collapsed offsets %5% and %6%")
                % __func__ % names[g_num].c_str() % names[g_den].c_str() % s.nx % s.off_num % s.off_den;
        }

        m_valid.resize(m_nbins_total);
        std::iota(m_valid.begin(), m_valid.end(), 0);
    }

    Eigen::VectorXf PROratio::Numerators(const Eigen::VectorXf &v) const {
        Eigen::VectorXf out = Eigen::VectorXf::Zero(m_nbins_total);
        for(const Slot &s : m_slots)
            for(size_t bx = 0; bx < s.nx; ++bx) {
                float a = 0.0f;
                for(size_t by = 0; by < s.ny_num; ++by) a += v(s.off_num + bx*s.ny_num + by);
                out(s.out_start + bx) = a;
            }
        return out;
    }

    Eigen::VectorXf PROratio::Denominators(const Eigen::VectorXf &v) const {
        Eigen::VectorXf out = Eigen::VectorXf::Zero(m_nbins_total);
        for(const Slot &s : m_slots)
            for(size_t bx = 0; bx < s.nx; ++bx) {
                float b = 0.0f;
                for(size_t by = 0; by < s.ny_den; ++by) b += v(s.off_den + bx*s.ny_den + by);
                out(s.out_start + bx) = b;
            }
        return out;
    }

    Eigen::VectorXf PROratio::Apply(const Eigen::VectorXf &v) const {
        const Eigen::VectorXf a = Numerators(v);
        const Eigen::VectorXf b = Denominators(v);
        Eigen::VectorXf out = Eigen::VectorXf::Zero(m_nbins_total);
        for(size_t i = 0; i < m_nbins_total; ++i)
            out(i) = (b(i) != 0.0f) ? a(i)/b(i) : 0.0f;
        return out;
    }

    // dR/da_k = 1/B for every k in the numerator sum, dR/db_k = -A/B^2 for the
    // denominator. Rows for different x bins have disjoint support.
    std::vector<PROratio::Term> PROratio::RowTerms(const Eigen::VectorXf &v, size_t r) const {
        std::vector<Term> terms;
        for(const Slot &s : m_slots) {
            if(r < s.out_start || r >= s.out_start + s.nx) continue;
            const size_t bx = r - s.out_start;
            float a = 0.0f, b = 0.0f;
            for(size_t by = 0; by < s.ny_num; ++by) a += v(s.off_num + bx*s.ny_num + by);
            for(size_t by = 0; by < s.ny_den; ++by) b += v(s.off_den + bx*s.ny_den + by);
            if(b == 0.0f) return terms;   // empty row, bin will have been dropped
            terms.reserve(s.ny_num + s.ny_den);
            for(size_t by = 0; by < s.ny_num; ++by)
                terms.push_back({(Eigen::Index)(s.off_num + bx*s.ny_num + by), 1.0f/b});
            for(size_t by = 0; by < s.ny_den; ++by)
                terms.push_back({(Eigen::Index)(s.off_den + bx*s.ny_den + by), -a/(b*b)});
            break;
        }
        return terms;
    }

    Eigen::MatrixXf PROratio::Jacobian(const Eigen::VectorXf &v) const {
        Eigen::MatrixXf J = Eigen::MatrixXf::Zero(m_nbins_total, m_ncollapsed);
        for(size_t r = 0; r < m_nbins_total; ++r)
            for(const Term &t : RowTerms(v, r)) J(r, t.idx) = t.c;
        return J;
    }

    Eigen::MatrixXf PROratio::PropagateCovariance(const Eigen::VectorXf &v, const Eigen::MatrixXf &C) const {
        std::vector<std::vector<Term>> rows(m_nbins_total);
        for(size_t r = 0; r < m_nbins_total; ++r) rows[r] = RowTerms(v, r);

        Eigen::MatrixXf CR = Eigen::MatrixXf::Zero(m_nbins_total, m_nbins_total);
        for(size_t r = 0; r < m_nbins_total; ++r) {
            for(size_t s = r; s < m_nbins_total; ++s) {
                float acc = 0.0f;
                for(const Term &t : rows[r])
                    for(const Term &u : rows[s])
                        acc += t.c * C(t.idx, u.idx) * u.c;
                CR(r,s) = acc;
                CR(s,r) = acc;
            }
        }
        return CR;
    }

    void PROratio::SetValidBins(const Eigen::VectorXf &data_coll,
                                   const Eigen::VectorXf &cv_coll,
                                   float min_denominator) {
        const Eigen::VectorXf bd = Denominators(data_coll);
        const Eigen::VectorXf bc = Denominators(cv_coll);
        m_valid.clear();
        for(size_t i = 0; i < m_nbins_total; ++i) {
            if(bd(i) <= 0.0f) {
                log<LOG_WARNING>(L"%1% || Dropping ratio bin %2% (%3%): denominator has no data.")
                    % __func__ % i % m_labels[i].c_str();
                continue;
            }
            if(bc(i) < min_denominator) {
                log<LOG_WARNING>(L"%1% || Dropping ratio bin %2% (%3%): CV denominator %4% below floor %5%. 1/B^2 in the Jacobian would amplify it.")
                    % __func__ % i % m_labels[i].c_str() % bc(i) % min_denominator;
                continue;
            }
            m_valid.push_back((Eigen::Index)i);
        }
        log<LOG_INFO>(L"%1% || Ratio fit uses %2% of %3% ratio bins.")
            % __func__ % m_valid.size() % m_nbins_total;
    }

    Eigen::MatrixXf BuildCollapsedSystCovariance(const PROconfig &config,
                                                 const Eigen::VectorXf &spec_full,
                                                 const Eigen::MatrixXf &frac_cov) {
        Eigen::MatrixXf full_cov = spec_full.asDiagonal() * frac_cov * spec_full.asDiagonal();
        return CollapseMatrix(config, full_cov);
    }

    float PROratioChi2(const PROratio &rmap,
                       const Eigen::VectorXf &pred_coll,
                       const Eigen::VectorXf &data_coll,
                       const Eigen::MatrixXf &M) {
        const Eigen::MatrixXf CR = rmap.PropagateCovariance(pred_coll, M);
        const Eigen::VectorXf Rp = rmap.Apply(pred_coll);
        const Eigen::VectorXf Rd = rmap.Apply(data_coll);

        const std::vector<Eigen::Index> &keep = rmap.ValidBins();
        const Eigen::Index n = (Eigen::Index)keep.size();
        if(n == 0) return 0.0f;

        Eigen::MatrixXf Mred(n,n);
        Eigen::VectorXf delta(n);
        for(Eigen::Index i = 0; i < n; ++i) {
            delta(i) = Rp(keep[i]) - Rd(keep[i]);
            for(Eigen::Index j = 0; j < n; ++j) Mred(i,j) = CR(keep[i], keep[j]);
        }
        return delta.dot(Mred.llt().solve(delta));
    }
}