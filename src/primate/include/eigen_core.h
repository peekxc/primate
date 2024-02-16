#ifndef __EIGEN_CORE_H
#define __EIGEN_CORE_H

#include <Eigen/Core>

using Eigen::Dynamic; 
using Eigen::Ref; 
using Eigen::MatrixXf;
using Eigen::MatrixXd;

template< typename F >
using Vector = Eigen::Matrix< F, Dynamic, 1 >;

template< typename F >
using Array = Eigen::Array< F, Dynamic, 1 >;

template< typename F >
using DenseMatrix = Eigen::Matrix< F, Dynamic, Dynamic >;

#endif