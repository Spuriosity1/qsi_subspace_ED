#!/bin/bash
set -e

# Resolve the directory the script is in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

data_dir="$REPO_ROOT/test/data"
tmp_dir="$REPO_ROOT/test/tmp"
stem="pyro_2_2_2x0,4,4b4,0,4b4,4,0b1"
exec_dir="$REPO_ROOT/build"
infile="${tmp_dir}/${stem}.json"

ref_outfile_csv="${data_dir}/${stem}.reference.basis.csv"
ref_outfile_h5="${data_dir}/${stem}.reference.basis.h5"

mkdir -p "$tmp_dir"
cp "${data_dir}/${stem}.json" "$infile"

test_single() {
  ext_st=".test_basis_st"
  outfile="${tmp_dir}/${stem}.0${ext_st}.csv"
  h5file="${outfile%.csv}.h5"
  rm -f "$outfile" "$h5file"

  echo "[running] gen_spinon_basis (single-threaded)"
  "${exec_dir}/gen_spinon_basis" "$infile" 0 "$ext_st" --out_format both > "${tmp_dir}/output_st.txt"

  if diff "$outfile" "$ref_outfile_csv" > /dev/null; then
    echo -e "\033[32;1m[csv] single-threaded test passed\033[0m"
  else
    echo -e "\033[31;1m[csv] single-threaded test failed\033[0m"
  fi

  if h5diff "$h5file" "$ref_outfile_h5"; then
    echo -e "\033[32;1m[hdf5] single-threaded test passed\033[0m"
  else
    echo -e "\033[31;1m[hdf5] single-threaded test failed\033[0m"
    return 1
  fi
}

test_parallel() {
  ext_par=".test_basis_par"
  outfile="${tmp_dir}/${stem}.0${ext_par}.csv"
  h5file="${outfile%.csv}.h5"
  rm -f "$outfile" "$h5file"

  echo "[running] gen_spinon_basis in parallel mode"
  "${exec_dir}/gen_spinon_basis" "$infile" 0 --n_threads 4 "$ext_par" --out_format both > "${tmp_dir}/output_par.txt"

  if diff "$outfile" "$ref_outfile_csv" > /dev/null; then
    echo -e "\033[32;1m[csv] multi-threaded test passed\033[0m"
  else
    echo -e "\033[31;1m[csv] multi-threaded test failed\033[0m"
  fi

  if h5diff "$h5file" "$ref_outfile_h5" > /dev/null; then
    echo -e "\033[32;1m[hdf5] multi-threaded test passed\033[0m"
  else
    echo -e "\033[31;1m[hdf5] multi-threaded test failed\033[0m"
    return 1
  fi
}

test_spinorder() {
  ext_rand=".test_basis_par_rand"
  ext_default=".test_basis_par"
  outfile_rand="${tmp_dir}/${stem}.0${ext_rand}.h5"
  outfile_default="${tmp_dir}/${stem}.0${ext_default}.h5"
  rm -f "$outfile_rand" "$outfile_default"

  echo "[running] gen_spinon_basis --order_spins random"
  "${exec_dir}/gen_spinon_basis" "$infile" 0 --n_threads 4 "$ext_rand" --order_spins random > "${tmp_dir}/output_rand.txt"

  echo "[running] gen_spinon_basis --order_spins greedy"
  "${exec_dir}/gen_spinon_basis" "$infile" 0 --n_threads 4 "$ext_default" --order_spins greedy > "${tmp_dir}/output_rand.txt"

  if h5diff "$outfile_rand" "$outfile_default" > /dev/null; then
    echo -e "\033[32;1mspin order test passed\033[0m"
  else
    echo -e "\033[31;1mspin order test failed\033[0m"
    return 1
  fi
}

# entry point
MODE="${1:-all}"

case "$MODE" in
  single)
    test_single
    ;;
  parallel)
    test_parallel
    ;;
  spinorder)
    test_spinorder
    ;;
  all)
    test_single &&
    test_parallel &&
    test_spinorder
    ;;
  *)
    echo "Usage: $0 [single|parallel|hdf5|spinorder|all]"
    exit 1
    ;;
esac

