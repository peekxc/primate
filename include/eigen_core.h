#ifndef __EIGEN_CORE_H
#define __EIGEN_CORE_H

#include <Eigen/Core>

using Eigen::Dynamic; 
using Eigen::Ref; 
using Eigen::MatrixXf;
using Eigen::MatrixXd;

template< typename F >
using ColVector = Eigen::Matrix< F, Dynamic, 1 >;

template< typename F >
using RowVector = Eigen::Matrix< F, 1, Dynamic >;

template< typename F >
using DenseMatrix = Eigen::Matrix< F, Dynamic, Dynamic >;

#endif