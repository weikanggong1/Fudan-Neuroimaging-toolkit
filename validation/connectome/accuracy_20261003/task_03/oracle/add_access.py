from pathlib import Path
import sys
root=Path(sys.argv[1]); p=root/'src/dwi/tractography/algorithms/iFOD2.h'; s=p.read_text()
needle='          private:\n            const Shared& S;'
assert s.count(needle)==1
method='''            bool fnit_oracle_arc(const Eigen::Vector3f& point,
                                const Eigen::Vector3f& start_direction,
                                const Eigen::Vector3f& end_direction,
                                vector<Eigen::Vector3f>& out_positions,
                                vector<Eigen::Vector3f>& out_tangents,
                                float& start_amplitude, float& probability,
                                Eigen::Vector3f& metrics) {
              pos=point; dir=start_direction.normalized();
              if (!get_data(source)) return false;
              start_amplitude=FOD(dir);
              half_log_prob0=0.5f*std::log(start_amplitude);
              get_path(out_positions,out_tangents,end_direction.normalized());
              probability=path_prob(out_positions,out_tangents);
              return true;
            }
'''
p.write_text(s.replace(needle,method+'\n'+needle))
