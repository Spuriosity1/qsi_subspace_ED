#pragma once
#include "operator.hpp"

// Local sparse methods for converting SymbolicOpSums to concrete matrices

template <RealOrCplx coeff_t>
struct LazyOpSum {
	using Scalar = coeff_t;
	explicit LazyOpSum(
			const ZBasis& basis_, const SymbolicOpSum<coeff_t>& ops_
			) : basis(basis_), ops(ops_) 
	{
	}


    LazyOpSum operator=(const LazyOpSum& other) = delete;

    LazyOpSum(const LazyOpSum& other) : basis(other.basis), ops(other.ops) {
		//tmp = new coeff_t[basis.dim()]; 
    }

	~LazyOpSum() {
		//delete[] tmp;
	}

	// Core evaluator 
    // Applies y = A x (sets y=0 first)
	void evaluate(const coeff_t* x, coeff_t* y) const
    {
		std::fill(y, y + basis.dim(), coeff_t(0));
        this->evaluate_add(x, y);
	}

    // performs y <- Ax + y
	void evaluate_add(const coeff_t* x, coeff_t* y) const; 

	// Eigen-compatible wrapper, evaluate y = this * x
	template <typename In, typename Out>
	void applyTo(const In& x, Out& y) const {
		Eigen::Map<const Eigen::VectorXd> x_vec(x.data(), basis.dim());
		Eigen::Map<Eigen::VectorXd> y_vec(y.data(), basis.dim());
		evaluate(x_vec.data(), y_vec.data());
	}
	
	// Eigen glue
	Eigen::Index rows() const { return basis.dim(); }
	Eigen::Index cols() const { return basis.dim(); }

	
	// Enable Eigen’s operator*
	template <typename Rhs>
	friend Eigen::Product<LazyOpSum, Rhs, Eigen::AliasFreeProduct>
	operator*(const LazyOpSum& lhs, const Eigen::MatrixBase<Rhs>& rhs) {
		return Eigen::Product<LazyOpSum, Rhs, Eigen::AliasFreeProduct>(lhs, rhs.derived());
	}	

	// Dense matrix matrialiser
	Eigen::Matrix<coeff_t, Eigen::Dynamic, Eigen::Dynamic>
	toDenseMatrix() const {
		Eigen::Index N = basis.dim();
		Eigen::Matrix<coeff_t, Eigen::Dynamic, Eigen::Dynamic> M(N, N);

		Eigen::VectorXd x = Eigen::VectorXd::Zero(N);
		Eigen::VectorXd y(N);
		for (Eigen::Index j = 0; j < N; ++j) {
			x[j] = coeff_t(1);
			applyTo(x, y);
			M.col(j) = y;
			x[j] = coeff_t(0);
		}
		return M;
	}

	// Sparse matrix materialiser
	Eigen::SparseMatrix<coeff_t>
	toSparseMatrix(coeff_t tol = 1e-14) const {
		Eigen::Index N = basis.dim();
		std::vector<Eigen::Triplet<coeff_t>> triplets;

		// Eigen::VectorXd x = Eigen::VectorXd::Zero(N);
		// Eigen::VectorXd y(N);

		for (Eigen::Index j = 0; j < N; ++j) {
            for (auto& [c, op] : ops.terms){
				ZBasis::idx_t J = j;
                coeff_t res = c * op.applyIndex(basis, J);
                if (std::abs(res) > tol )
                    triplets.emplace_back(J, j, res);
            }

		}
		Eigen::SparseMatrix<coeff_t> S(N, N);
		S.setFromTriplets(triplets.begin(), triplets.end());
		return S;
	}


	private:
	// coeff_t* tmp; // temp storage
	const ZBasis& basis;
	const SymbolicOpSum<coeff_t> ops;
};

template <typename coeff_t>
class LazyOpSumProd {
public:
	using Scalar = coeff_t;

	LazyOpSumProd(const LazyOpSum<coeff_t>& op_)
	    : op(op_), xdim(op_.rows()) {}

	// Spectra expects raw pointers
	void perform_op(const coeff_t* x_in, coeff_t* y_out) const {
		op.evaluate(x_in, y_out);
	}

	Eigen::Index rows() const { return xdim; }
	Eigen::Index cols() const { return xdim; }

private:
	const LazyOpSum<coeff_t>& op;
	Eigen::Index xdim;
};


namespace Eigen {
	template<typename coeff_t, typename Rhs>
	struct Product<LazyOpSum<coeff_t>, Rhs, AliasFreeProduct> :
		public Matrix<coeff_t, Dynamic, 1>
	{
		Product(const LazyOpSum<coeff_t>& op, const Rhs& rhs)
			: Matrix<coeff_t, Dynamic, 1>(op.rows())
		{
			op.applyTo(rhs, *this);
		}
	};
}



